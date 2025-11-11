#ifndef SIMD_ARM64_HPP
#define SIMD_ARM64_HPP

#include "arm_neon.h"

#include <cstdint>
#include <array>

/*
  implementation of SIMDs for ARM-Neon CPUs:
  https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

  // neon coding:
  https://developer.arm.com/documentation/102159/0400
*/


namespace ASC_HPC
{

  template<>
  class SIMD<mask64,2>
  {
    int64x2_t m_val;
  public:
    SIMD (int64x2_t val) : m_val(val) { };
    SIMD (SIMD<mask64,1> v0, SIMD<mask64,1> v1)
      : m_val{vcombine_s64(int64x1_t{v0.val().val()}, int64x1_t{v1.val().val()})} { } 

    auto val() const { return m_val; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_val)[i] != 0; }

    SIMD<mask64, 1> lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> hi() const { return SIMD<mask64,1>((*this)[1]); }
    const mask64 * ptr() const { return (mask64*)&m_val; }
  };




  
  template<>
  class SIMD<double,2>
  {
    float64x2_t m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD (double val) : m_val{val,val} { }
    SIMD (float64x2_t val) : m_val(val) { }
    SIMD (double v0, double v1) : m_val{vcombine_f64(float64x1_t{v0}, float64x1_t{v1})} { }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) : SIMD(v0.val(), v1.val()) { }
    SIMD (std::array<double, 2> arr) : m_val{arr[0], arr[1]} { }

    SIMD (double const * p) : m_val{vld1q_f64(p)} { }
    SIMD (double const * p, SIMD<mask64,2> mask)
      {
	m_val[0] = mask[0] ? p[0] : 0;
	m_val[1] = mask[1] ? p[1] : 0;
      }

    static constexpr int size() { return 2; }    
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }

    auto lo() const { return SIMD<double,1> (m_val[0]); }
    auto hi() const { return SIMD<double,1> (m_val[1]); }
    double operator[] (int i) const { return m_val[i]; }

    void store (double * p) const
    {
      vst1q_f64(p, m_val);
    }
    
    void store (double * p, SIMD<mask64,2> mask) const
    {
      if (mask[0]) p[0] = m_val[0];
      if (mask[1]) p[1] = m_val[1];
    }
  };

  inline auto operator+ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()+b.val()); }
  inline auto operator- (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()-b.val()); }
  
  inline auto operator* (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()*b.val()); }
  inline auto operator* (double a, SIMD<double,2> b) { return SIMD<double,2> (a*b.val()); }   
  

  //
  inline SIMD<mask64,2> operator<= (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcleq_f64(a.val(), b.val())));}  
  inline SIMD<mask64,2> operator== (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(vreinterpretq_s64_u64(vceqq_f64(a.val(), b.val())));}
  inline SIMD<mask64,2> operator>= (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcgeq_f64(a.val(), b.val())));}
  
  // a*b+c
  inline SIMD<double,2> fma (SIMD<double,2> a, SIMD<double,2> b, SIMD<double,2> c) 
  { return vmlaq_f64(c.val(), a.val(), b.val()); }



  inline SIMD<double,2> select (SIMD<mask64,2> mask, SIMD<double,2> b, SIMD<double,2> c)
  { return vbslq_f64(mask.val(), b.val(), c.val()); }
  
  inline SIMD<double,2> hSum (SIMD<double,2> a, SIMD<double,2> b)
  { return vpaddq_f64(a.val(), b.val()); }

  // greater than or equal to
  // uint64x2_t vcgeq_f64(float64x2_t a, float64x2_t b)
  /*inline SIMD<double,2> operator>=(SIMD<double,2> a, SIMD<double,2> b)
  {
    // std::cout << "using >= operator for ARM" << std::endl;
    return SIMD<double,2>(vcgeq_f64(a.val(), b.val()));
  }
*/
//   inline SIMD<mask64,2> operator>=(SIMD<double,2> a, SIMD<double,2> b)
// {
//     uint64x2_t mask = vcgeq_f64(a.val(), b.val());

//     // Convert mask to doubles: true -> 1.0, false -> 0.0
//     float64x2_t result = vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));

//     return SIMD<double,2>(result);
// }

//   inline SIMD<mask64,4> operator>=(SIMD<double,4> a, SIMD<double,4> b)
//   {
//     // std::cout << "using >= operator for ARM 4-wide" << std::endl;
//     return SIMD<mask64,4>(a.lo()>=b.lo(), a.hi()>=b.hi());
//   }


  // less than or equal to
  // uint64x2_t vcleq_f64(float64x2_t a, float64x2_t b)
  /*inline SIMD<double,2> operator<=(SIMD<double,2> a, SIMD<double,2> b)
  {
    // std::cout << "using <= operator for ARM" << std::endl;
    return SIMD<double,2>(vcleq_f64(a.val(), b.val()));
  }*/

//  inline SIMD<mask64,2> operator<=(SIMD<double,2> a, SIMD<double,2> b)
// {
//     // Get mask: each lane is all-ones if a[i] <= b[i], else 0
//     uint64x2_t mask = vcleq_f64(a.val(), b.val());

//     // Convert mask to doubles: true -> 1.0, false -> 0.0
//     float64x2_t result = vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));

//     return SIMD<mask64,2>(result);
// }


//   inline SIMD<mask64,4> operator<=(SIMD<double,4> a, SIMD<double,4> b)
//   {
//     // std::cout << "using <= operator for ARM 4-wide" << std::endl;
//     return SIMD<mask64,4>(a.lo()<=b.lo(), a.hi()<=b.hi());
//   }

//   // greater than
//   // uint64x2_t vcgtq_f64(float64x2_t a, float64x2_t b)
//   inline SIMD<mask64,2> operator>(SIMD<double,2> a, SIMD<double,2> b)
//   {
//     uint64x2_t mask = vcgtq_f64(a.val(), b.val());
//     float64x2_t result = vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));
//     return SIMD<mask64,2>(result);
//   }

//   inline SIMD<mask64,4> operator>(SIMD<double,4> a, SIMD<double,4> b)
//   {
//     return SIMD<mask64,4>(a.lo()>b.lo(), a.hi()>b.hi());
//   }

  // less than
  // uint64x2_t vcltq_f64(float64x2_t a, float64x2_t b)
  // inline SIMD<mask64,2> operator<(SIMD<double,2> a, SIMD<double,2> b)
  // {
  //   uint64x2_t mask = vcltq_f64(a.val(), b.val());
  //   float64x2_t result = vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));
  //   return SIMD<mask64,2>(result);
  // }

  // inline SIMD<mask64,4> operator<(SIMD<double,4> a, SIMD<double,4> b)
  // {
  //   return SIMD<mask64,4>(a.lo()<b.lo(), a.hi()<b.hi());
  // }

  // // is equal to
  // inline SIMD<mask64,2> operator==(SIMD<double,2> a, SIMD<double,2> b)
  // {
  //   uint64x2_t mask = vceqq_f64(a.val(), b.val());
  //   float64x2_t result = vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));
  //   return SIMD<mask64,2>(result);
  // }

  // inline SIMD<mask64,4> operator==(SIMD<double,4> a, SIMD<double,4> b)
  // {
  //   return SIMD<mask64,4>(a.lo()==b.lo(), a.hi()==b.hi());
  // }

  inline SIMD<mask64,2> operator> (SIMD<double,2> a, SIMD<double,2> b) {
    return SIMD<mask64,2>(vreinterpretq_s64_u64(vcgtq_f64(a.val(), b.val())));
  }

  inline SIMD<mask64,2> operator< (SIMD<double,2> a, SIMD<double,2> b) {
    return SIMD<mask64,2>(vreinterpretq_s64_u64(vcltq_f64(a.val(), b.val())));
  }


  inline void transpose(
    SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
    SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3)
{
    auto a0l = a0.lo().val(), a0h = a0.hi().val();
    auto a1l = a1.lo().val(), a1h = a1.hi().val();
    auto a2l = a2.lo().val(), a2h = a2.hi().val();
    auto a3l = a3.lo().val(), a3h = a3.hi().val();


    float64x2_t t0l = vzip1q_f64(a0l, a1l); 
    float64x2_t t1l = vzip2q_f64(a0l, a1l); 
    float64x2_t t2l = vzip1q_f64(a2l, a3l); 
    float64x2_t t3l = vzip2q_f64(a2l, a3l); 

    float64x2_t t0h = vzip1q_f64(a0h, a1h); 
    float64x2_t t1h = vzip2q_f64(a0h, a1h); 
    float64x2_t t2h = vzip1q_f64(a2h, a3h); 
    float64x2_t t3h = vzip2q_f64(a2h, a3h); 

    b0 = SIMD<double,4>(SIMD<double,2>(t0l), SIMD<double,2>(t2l)); // [a00 a10 a20 a30]
    b1 = SIMD<double,4>(SIMD<double,2>(t1l), SIMD<double,2>(t3l)); // [a01 a11 a21 a31]
    b2 = SIMD<double,4>(SIMD<double,2>(t0h), SIMD<double,2>(t2h)); // [a02 a12 a22 a32]
    b3 = SIMD<double,4>(SIMD<double,2>(t1h), SIMD<double,2>(t3h)); // [a03 a13 a23 a33]
}


}


#endif
