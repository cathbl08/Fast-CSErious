#ifndef SIMD_HPP
#define SIMD_HPP

#include<iostream>
#include<string>
#include<memory>
#include <array>
#include <cstdint>

namespace ASC_HPC
{


#ifdef __AVX__
  constexpr size_t DefaultSimdSizeBytes = 32;
#else
  constexpr size_t DefaultSimdSizeBytes = 16;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

  
  template <typename T, size_t S = DefaultSimdSizeBytes/sizeof(T)> class SIMD;
  
  
  constexpr size_t largestPowerOfTwo (size_t x)
  {
    size_t y = 1;
    while (2*y <= x) y *= 2;
    return y;
  }

  static constexpr double sincof[] = {
      1.58962301576546568060E-10,
      -2.50507477628578072866E-8,
      2.75573136213857245213E-6,
      -1.98412698295895385996E-4,
      8.33333333332211858878E-3,
      -1.66666666666666307295E-1,
  };

  static constexpr double coscof[6] = {
      -1.13585365213876817300E-11,
      2.08757008419747316778E-9,
      -2.75573141792967388112E-7,
      2.48015872888517045348E-5,
      -1.38888888888730564116E-3,
      4.16666666666665929218E-2,
  };
  
  class mask64
  {
    int64_t m_mask;
  public:
    mask64 (bool b)
      : m_mask{ b ? -1 : 0 } { }
    auto val() const { return m_mask; }
    operator bool() { return bool(m_mask); }
  };
  
  inline std::ostream & operator<< (std::ostream & ost, mask64 m)
  {
    ost << (m ? 't' : 'f');
    return ost;
  }

  namespace detail {
    template <typename T, size_t N, size_t... I>
    auto array_range_impl(std::array<T, N> const& arr, size_t first,
                          std::index_sequence<I...>) {
      return std::array<T, sizeof...(I)>{arr[first + I]...};
    }
    
    template <size_t FIRST, size_t NEXT, typename T, size_t N>
    auto array_range(std::array<T, N> const& arr) {
      return array_range_impl(arr, FIRST, std::make_index_sequence<NEXT-FIRST>{});
    }
  } // namespace detail

  
  
  template <typename T, size_t S>
  class SIMD
  {
  protected:
    static constexpr size_t S1 = largestPowerOfTwo(S-1);
    static constexpr size_t S2 = S-S1;

    SIMD<T,S1> m_lo;
    SIMD<T,S2> m_hi;
  public:
    SIMD() = default;

    explicit SIMD (T val)
      : m_lo(val), m_hi(val) { }

    explicit SIMD (SIMD<T,S1> lo, SIMD<T,S2> hi)
      : m_lo(lo), m_hi(hi) { }
    
    explicit SIMD (std::array<T, S> arr)
      : m_lo(detail::array_range<0, S1>(arr)),
        m_hi(detail::array_range<S1, S>(arr))
      {}

    template <typename ...T2>
    explicit SIMD (T val0, T2... vals)
      : SIMD(std::array<T, S>{val0, vals...}) { }

    explicit SIMD (T * ptr)
      : m_lo(ptr), m_hi(ptr+S1) { }
    
    explicit SIMD (T * ptr, SIMD<mask64,S> mask)
      : m_lo(ptr, mask.lo()), m_hi(ptr+S1, mask.hi()) { }
    
    
    static constexpr int size() { return S; }    
    auto & lo() { return m_lo; }
    auto & hi() { return m_hi; }

    const T * ptr() const { return m_lo.ptr(); }
    T operator[] (size_t i) const { return ptr()[i]; }

    void store (T * ptr) const {
      m_lo.store(ptr);
      m_hi.store(ptr+S1);
    }

    void store (T * ptr, SIMD<mask64,S> mask) const {
      m_lo.store(ptr, mask.lo());
      m_hi.store(ptr+S1, mask.hi());
    }
  };


  
  template <typename T>
  class SIMD<T,1>
  {
    T m_val;
  public:
    SIMD() = default;
    SIMD(T val) : m_val(val) { }
    SIMD(std::array<T,1> vals) : m_val(vals[0]) { }
    explicit SIMD (T * ptr) : m_val{*ptr} { } 

    auto val() const { return m_val; }
    
    explicit SIMD (T * ptr, SIMD<mask64,1> mask)
      : m_val{ mask.val() ? *ptr : T(0)} { }

    static constexpr size_t size() { return 1; }
    const T * ptr() const { return &m_val; }    
    T operator[] (size_t i) const { return m_val; }

    void store (T * ptr) const { *ptr = m_val; }
    void store (T * ptr, SIMD<mask64,1> mask) const { if (mask.val()) *ptr = m_val; }
  };

  // template<> class SIMD<mask64,2>;
  // template<> class SIMD<double,2>;
  // template<> class SIMD<int64_t,2>;


  template <typename T, size_t S>  
  std::ostream & operator<< (std::ostream & ost, SIMD<T,S> simd)
  {
    for (size_t i = 0; i < S-1; i++)
      ost << simd[i] << ", ";
    ost << simd[S-1];
    return ost;
  }

  // ********************** Arithmetic operations ********************************

  template <typename T, size_t S>
  auto operator+ (SIMD<T,S> a, SIMD<T,S> b) { return SIMD<T,S> (a.lo()+b.lo(), a.hi()+b.hi()); }
  template <typename T>
  auto operator+ (SIMD<T,1> a, SIMD<T,1> b) { return SIMD<T,1> (a.val()+b.val()); }


  template <typename T, size_t S>
  auto operator* (SIMD<T,S> a, SIMD<T,S> b) { return SIMD<T,S> (a.lo()*b.lo(), a.hi()*b.hi()); }
  template <typename T>
  auto operator* (SIMD<T,1> a, SIMD<T,1> b) { return SIMD<T,1> (a.val()*b.val()); }
  
  template <typename T, size_t S>
  auto operator* (double a, SIMD<T,S> b) { return SIMD<T,S> (a*b.lo(), a*b.hi()); }
  template <typename T>
  auto operator* (double a, SIMD<T,1> b) { return SIMD<T,1> (a*b.val()); }

  template <typename T, size_t S>
  auto operator+= (SIMD<T,S> & a, SIMD<T,S> b) { a = a+b; return a; }
  
  template <typename T, size_t S>
  auto fma(SIMD<T,S> a, SIMD<T,S> b, SIMD<T,S> c)
  { return SIMD<T,S> (fma(a.lo(),b.lo(),c.lo()), fma(a.hi(),b.hi(),c.hi())); }    
  template <typename T>
  auto fma(SIMD<T,1> a, SIMD<T,1> b, SIMD<T,1> c)
  { return SIMD<T,1> (a.val()*b.val()+c.val()); }



  // ****************** Horizontal sums *****************************
  
  template <typename T, size_t S>
  auto hSum (SIMD<T,S> a) { return hSum(a.lo())+hSum(a.hi()); }

  template <typename T>
  auto hSum (SIMD<T,1> a) { return a.val(); }
  
  
  template <typename T, size_t S>
  auto hSum (SIMD<T,S> a0, SIMD<T,S> a1)
  { return hSum(a0.lo(), a1.lo())+hSum(a0.hi(), a1.hi()); }

  template <typename T>
  auto hSum(SIMD<T,1> a0, SIMD<T,1> a1)
  { return SIMD<T,2> (a0.val(), a1.val()); }


  
  // ******************  select   ***********************************

  template <typename T>
  auto select (SIMD<mask64,1> mask, SIMD<T,1> a, SIMD<T,1> b)
  { return mask.val() ? a : b; }
  
  template <typename T, size_t S>
  auto select (SIMD<mask64,S> mask, SIMD<T,S> a, SIMD<T,S> b)
  { return SIMD<T,S> (select (mask.lo(), a.lo(), b.lo()),
                      select (mask.hi(), a.hi(), b.hi())); }

  // ****************** IndexSequence ********************************
  
  template <typename T, size_t S, T first=0>
  class IndexSequence : public SIMD<T,S>
  {
    using SIMD<T,S>::S1;
    using SIMD<T,S>::S2;
  public:
    IndexSequence()
      : SIMD<T,S> (IndexSequence<T,S1,first>(),
                   IndexSequence<T,S2,first+S1>()) { }
  };

  template <typename T, T first>
  class IndexSequence<T,1,first> : public SIMD<T,1>
  {
  public:
    IndexSequence() : SIMD<T,1> (first) { }
  };


  template <typename T, size_t S>
  auto operator>= (SIMD<T,S> a, SIMD<T,S> b)
  { return SIMD<mask64,S>(a.lo()>=b.lo(), a.hi()>=b.hi()); }

  template <typename T>
  auto operator>= (SIMD<T,1> a, SIMD<T,1> b)
  { return SIMD<mask64,1>(a.val()>=b.val()); }

  template <typename TA, typename T, size_t S>
  auto operator>= (TA a, const SIMD<T,S> & b)
  { 
    // std::cout << "Warning: implicit conversion in operator>=" << std::endl;
    return SIMD<T,S>(a) >= b; }


  // new operator <=
  template <typename T, size_t S>
  auto operator<= (SIMD<T,S> a, SIMD<T,S> b)
  { return SIMD<mask64,S>(a.lo()<=b.lo(), a.hi()<=b.hi()); }

  template <typename T>
  auto operator<= (SIMD<T,1> a, SIMD<T,1> b)
  { return SIMD<mask64,1>(a.val()<=b.val()); }


  template <typename TA, typename T, size_t S>
  auto operator<= (TA a, const SIMD<T,S> & b)
  { 
    // std::cout << "Warning: implicit conversion in operator<=" << std::endl;
    return SIMD<T,S>(a) <= b; }


  // new operator == 
  template <typename T, size_t S>
  auto operator== (SIMD<T,S> a, SIMD<T,S> b)
  { return SIMD<mask64,S>(a.lo()==b.lo(), a.hi()==b.hi()); }

  template <typename T>
  auto operator== (SIMD<T,1> a, SIMD<T,1> b)
  { return SIMD<mask64,1>(a.val()==b.val()); }

  template <typename TA, typename T, size_t S>
  auto operator== (TA a, const SIMD<T,S> & b)
  { 
    // std::cout << "Warning: implicit conversion in operato== " << std::endl;
    return SIMD<T,S>(a) == b; }


  void transpose (SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
                SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3);


  // highly accurate on [-pi/4, pi/4]
  template <typename T>
  auto sincos_reduced(T x)
  {
    auto x2 = x * x;

    auto s = (((((sincof[0] * x2 + sincof[1]) * x2 + sincof[2]) * x2 + sincof[3]) * x2 + sincof[4]) * x2 + sincof[5]);
    s = x + x * x * x * s;

    auto c = (((((coscof[0] * x2 + coscof[1]) * x2 + coscof[2]) * x2 + coscof[3]) * x2 + coscof[4]) * x2 + coscof[5]);
    c = 1.0 - 0.5 * x2 + x2 * x2 * c;

    return std::tuple{s, c};
  }

  template <size_t N>
  auto sincos(SIMD<double, N> x)
  {
    SIMD<double, N> y = round((2 / M_PI) * x);
    SIMD<int64_t, N> q = lround(y);

    auto [s1, c1] = sincos_reduced(x - y * (M_PI / 2));

    auto s2 = select((q & SIMD<int64_t, N>(1)) == SIMD<int64_t, N>(0), s1, c1);
    auto s = select((q & SIMD<int64_t, N>(2)) == SIMD<int64_t, N>(0), s2, -s2);

    auto c2 = select((q & SIMD<int64_t, N>(1)) == SIMD<int64_t, N>(0), c1, -s1);
    auto c = select((q & SIMD<int64_t, N>(2)) == SIMD<int64_t, N>(0), c2, -c2);

    return std::tuple{s, c};
  }
}
  
  
#ifdef __AVX__
#include "simd_avx.hpp"
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include "simd_arm64.hpp"
#endif



#endif
