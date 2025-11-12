#ifndef SIMD_AVX_HPP
#define SIMD_AVX_HPP

#include <immintrin.h>
#include <array>


/*
  implementation of SIMDs for Intel-CPUs with AVX support:
  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */


namespace ASC_HPC
{

  // SIMD <T,2>
  template<>
  class SIMD<mask64,2>
  {
    __m128i m_mask;
  public:

    SIMD (__m128i mask) : m_mask(mask) { };
    SIMD (__m128d mask) : m_mask(_mm_castpd_si128(mask)) { ; }
    SIMD (SIMD<mask64,1> lo, SIMD<mask64,1> hi) : m_mask(_mm_set_epi64x(hi[0] ? -1 : 0, lo[0] ? -1 : 0)) { }
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }
    
    SIMD<mask64, 1> lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> hi() const { return SIMD<mask64,1>((*this)[1]); }
  };


  template<>
  class SIMD<mask64,4>
  {
    __m256i m_mask;
  public:

    SIMD (__m256i mask) : m_mask(mask) { };
    SIMD (__m256d mask) : m_mask(_mm256_castpd_si256(mask)) { ; }
    SIMD (SIMD<mask64,2> lo, SIMD<mask64,2> hi) : m_mask(_mm256_insertf128_si256(_mm256_castsi128_si256(lo.val()), hi.val(), 1)) { }
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }

    // SIMD<mask64, 2> lo() const { return SIMD<mask64,2>((*this)[0], (*this)[1]); }
    // SIMD<mask64, 2> hi() const { return SIMD<mask64,2>((*this)[2], (*this)[3]); }

    SIMD<mask64,2> lo() const { return SIMD<mask64,2>(_mm256_castsi256_si128(m_mask)); }
    SIMD<mask64,2> hi() const { return SIMD<mask64,2>(_mm256_extractf128_si256(m_mask, 1)); }
  };


  template<>
  class SIMD<int64_t,2>
  {
    __m128i m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t val) : m_val{_mm_set1_epi64x(val)} {};
    SIMD(__m128i val) : m_val{val} {};
    SIMD (int64_t v0, int64_t v1) : m_val{_mm_set_epi64x(v1,v0) } { } 
    SIMD (SIMD<int64_t,1> v0, SIMD<int64_t,1> v1) : SIMD(v0[0], v1[0]) { }  // can do better !
    // SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    // SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    // SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    // const double * Ptr() const { return (double*)&val; }
    SIMD<int64_t, 1> lo() const { return SIMD<int64_t,1>((*this)[0]); }
    SIMD<int64_t, 1> hi() const { return SIMD<int64_t,1>((*this)[1]); }
    int64_t operator[](size_t i) const { return ((int64_t*)&m_val)[i]; }
  };

  
  template<>
  class SIMD<int64_t,4>
  {
    __m256i m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t val) : m_val{_mm256_set1_epi64x(val)} {};
    SIMD(__m256i val) : m_val{val} {};
    SIMD (int64_t v0, int64_t v1, int64_t v2, int64_t v3) : m_val{_mm256_set_epi64x(v3,v2,v1,v0) } { } 
    SIMD (SIMD<int64_t,2> v0, SIMD<int64_t,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // can do better !
    // SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    // SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    // SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    // const double * Ptr() const { return (double*)&val; }
    SIMD<int64_t, 2> lo() const { return SIMD<int64_t, 2>(_mm256_extractf128_si256(m_val, 0)); }
    SIMD<int64_t, 2> hi() const { return SIMD<int64_t, 2>(_mm256_extractf128_si256(m_val, 1)); }
    int64_t operator[](size_t i) const { return ((int64_t*)&m_val)[i]; }
  };
  

  template<>
  class SIMD<double,2>
  {
    __m128d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double val) : m_val{_mm_set1_pd(val)} {};
    SIMD(__m128d val) : m_val{val} {};
    SIMD (double v0, double v1) : m_val{_mm_set_pd(v1,v0)} {  }
    // SIMD (SIMD<double,2> v0, SIMD<double,2> v1) : SIMD(v0[0], v0[1]) { }  // better with _mm256_set_m128d
    SIMD(SIMD<double,1> v0, SIMD<double,1> v1) : m_val(_mm_set_pd(v1.val(), v0.val())) {}
    SIMD (std::array<double,2> a) : SIMD(a[0],a[1]) { }
    SIMD (double const * p) { m_val = _mm_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,2> mask) { m_val = _mm_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 1> lo() const { return SIMD<double,1>((*this)[0]); }
    SIMD<double, 1> hi() const { return SIMD<double,1>((*this)[1]); }

    // better:
    // SIMD<double, 2> lo() const { return _mm256_extractf128_pd(m_val, 0); }
    // SIMD<double, 2> hi() const { return _mm256_extractf128_pd(m_val, 1); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64,2> mask) const { _mm_maskstore_pd(p, mask.val(), m_val); }
  };
  
  template<>
  class SIMD<double,4>
  {
    __m256d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double val) : m_val{_mm256_set1_pd(val)} {};
    SIMD(__m256d val) : m_val{val} {};
    SIMD (double v0, double v1, double v2, double v3) : m_val{_mm256_set_pd(v3,v2,v1,v0)} {  }
    SIMD (SIMD<double,2> v0, SIMD<double,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // better with _mm256_set_m128d
    SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    SIMD (double const * p) { m_val = _mm256_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,4> mask) { m_val = _mm256_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    // SIMD<double, 2> lo() const { return SIMD<double,2>((*this)[0], (*this)[1]); }
    // SIMD<double, 2> hi() const { return SIMD<double,2>((*this)[2], (*this)[3]); }

    SIMD<double,2> lo() const { return SIMD<double,2>(_mm256_extractf128_pd(m_val, 0)); }
    SIMD<double,2> hi() const { return SIMD<double,2>(_mm256_extractf128_pd(m_val, 1)); }


    // better:
    // SIMD<double, 2> lo() const { return _mm256_extractf128_pd(m_val, 0); }
    // SIMD<double, 2> hi() const { return _mm256_extractf128_pd(m_val, 1); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm256_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64,4> mask) const { _mm256_maskstore_pd(p, mask.val(), m_val); }
  };
  

  SIMD<int64_t,4> lround(SIMD<double,4> val)
  {
    __m128i tmp32 = _mm256_cvtpd_epi32(_mm256_round_pd(val.val(), 0));
    __m256i tmp64 = _mm256_cvtepi32_epi64(tmp32);
    return SIMD<int64_t,4>(tmp64);
  }

  SIMD<double,4> round(SIMD<double,4> val)
  {
    return SIMD<double,4>(_mm256_round_pd(val.val(), 2));
  }


  template <int64_t first>
  class IndexSequence<int64_t, 4, first> : public SIMD<int64_t,4>
  {
  public:
    IndexSequence()
      : SIMD<int64_t,4> (first, first+1, first+2, first+3) { }
  };
  


  
  inline auto operator+ (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_add_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_sub_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,4> a) { return SIMD<double,4>(_mm256_sub_pd(_mm256_setzero_pd(), a.val())); }
  
  inline auto operator* (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_mul_pd(a.val(), b.val())); }
  inline auto operator* (double a, SIMD<double,4> b) { return SIMD<double,4>(a)*b; }

  inline auto operator&(SIMD<int64_t,4> a, SIMD<int64_t,4> b) { return SIMD<int64_t,4>(_mm256_and_si256(a.val(), b.val())); }

  inline auto operator+(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2>(_mm_add_pd(a.val(), b.val())); }
  inline auto operator-(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2>(_mm_sub_pd(a.val(), b.val())); }
  inline auto operator*(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2>(_mm_mul_pd(a.val(), b.val())); }
  inline auto operator*(double s, SIMD<double,2> b) { return SIMD<double,2>(_mm_mul_pd(_mm_set1_pd(s), b.val())); }
  inline auto operator*(SIMD<double,2> a, double s) { return SIMD<double,2>(_mm_mul_pd(a.val(), _mm_set1_pd(s))); }

  inline auto operator&(SIMD<int64_t,2> a, SIMD<int64_t,2> b) { return SIMD<int64_t,2>(_mm_and_si128(a.val(), b.val())); }

#ifdef __FMA__
  inline SIMD<double,4> fma (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
  { return _mm256_fmadd_pd (a.val(), b.val(), c.val()); }
#endif

#ifdef __FMA__
inline SIMD<double,2> fma(SIMD<double,2> a, SIMD<double,2> b, SIMD<double,2> c) {
  return SIMD<double,2>(_mm_fmadd_pd(a.val(), b.val(), c.val()));
}
// #else
// inline SIMD<double,2> fma(SIMD<double,2> a, SIMD<double,2> b, SIMD<double,2> c) {
//   return a*b + c;
// }
#endif

  inline SIMD<mask64,4> operator>= (SIMD<int64_t,4> a , SIMD<int64_t,4> b)
  { // there is no a>=b, so we return !(b>a)
    return  _mm256_xor_si256(_mm256_cmpgt_epi64(b.val(),a.val()),_mm256_set1_epi32(-1)); }
  
  inline auto operator>= (SIMD<double,4> a, SIMD<double,4> b)
  { return SIMD<mask64,4>(_mm256_cmp_pd (a.val(), b.val(), _CMP_GE_OQ)); }

  inline auto operator== (SIMD<double,4> a, SIMD<double,4> b) 
  { return SIMD<mask64,4>(_mm256_cmp_pd(a.val(), b.val(), _CMP_EQ_OQ)); }

  inline auto operator<= (SIMD<double,4> a, SIMD<double,4> b) 
  { return SIMD<mask64,4>(_mm256_cmp_pd(a.val(), b.val(), _CMP_LE_OQ)); }

  

  inline auto operator==(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(_mm_castpd_si128(_mm_cmpeq_pd(a.val(), b.val()))); }
  inline auto operator<=(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(_mm_castpd_si128(_mm_cmple_pd(a.val(), b.val()))); }
  inline auto operator>=(SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(_mm_castpd_si128(_mm_cmpge_pd(a.val(), b.val()))); }


  inline void transpose(
    SIMD<double,4> a0, SIMD<double,4> a1,
    SIMD<double,4> a2, SIMD<double,4> a3,
    SIMD<double,4> &b0, SIMD<double,4> &b1,
    SIMD<double,4> &b2, SIMD<double,4> &b3)
{
    // load 4 rows 
    __m256d r0 = a0.val();
    __m256d r1 = a1.val();
    __m256d r2 = a2.val();
    __m256d r3 = a3.val();

    __m256d t0 = _mm256_unpacklo_pd(r0, r1); 
    __m256d t1 = _mm256_unpackhi_pd(r0, r1); 
    __m256d t2 = _mm256_unpacklo_pd(r2, r3); 
    __m256d t3 = _mm256_unpackhi_pd(r2, r3); 

    // combine two 128 halves
    __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20); 
    __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20); 
    __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31); 

    // store into SIMD<double,4>
    b0 = SIMD<double,4>(c0);
    b1 = SIMD<double,4>(c1);
    b2 = SIMD<double,4>(c2);
    b3 = SIMD<double,4>(c3);
}
  
}

#endif
