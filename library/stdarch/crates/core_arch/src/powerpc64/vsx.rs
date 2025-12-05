//! PowerPC Vector Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use crate::core_arch::powerpc::macros::*;
use crate::core_arch::powerpc::*;

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem::transmute;

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.ppc.vsx.lxvl"]
    fn lxvl(a: *const u8, l: usize) -> vector_signed_int;

    #[link_name = "llvm.ppc.vsx.stxvl"]
    fn stxvl(v: vector_signed_int, a: *mut u8, l: usize);
}

mod sealed {
    use super::*;

    #[inline]
    #[target_feature(enable = "power9-vector")]
    #[cfg_attr(test, assert_instr(lxvl))]
    unsafe fn vec_lxvl(p: *const u8, l: usize) -> vector_signed_int {
        lxvl(p, l << 56)
    }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorXloads {
        type Result;
        unsafe fn vec_xl_len(self, l: usize) -> Self::Result;
    }

    macro_rules! impl_vsx_loads {
        ($ty:ident) => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorXloads for *const $ty {
                type Result = t_t_l!($ty);
                #[inline]
                #[target_feature(enable = "power9-vector")]
                unsafe fn vec_xl_len(self, l: usize) -> Self::Result {
                    transmute(vec_lxvl(self as *const u8, l))
                }
            }
        };
    }

    impl_vsx_loads! { i8 }
    impl_vsx_loads! { u8 }
    impl_vsx_loads! { i16 }
    impl_vsx_loads! { u16 }
    impl_vsx_loads! { i32 }
    impl_vsx_loads! { u32 }
    impl_vsx_loads! { f32 }

    #[inline]
    #[target_feature(enable = "power9-vector")]
    #[cfg_attr(test, assert_instr(stxvl))]
    unsafe fn vec_stxvl(v: vector_signed_int, a: *mut u8, l: usize) {
        stxvl(v, a, l << 56);
    }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorXstores {
        type Out;
        unsafe fn vec_xst_len(self, p: Self::Out, l: usize);
    }

    macro_rules! impl_stores {
        ($ty:ident) => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorXstores for t_t_l!($ty) {
                type Out = *mut $ty;
                #[inline]
                #[target_feature(enable = "power9-vector")]
                unsafe fn vec_xst_len(self, a: Self::Out, l: usize) {
                    stxvl(transmute(self), a as *mut u8, l)
                }
            }
        };
    }

    impl_stores! { i8 }
    impl_stores! { u8 }
    impl_stores! { i16 }
    impl_stores! { u16 }
    impl_stores! { i32 }
    impl_stores! { u32 }
    impl_stores! { f32 }
}

/// Vector Load with Length
///
/// ## Purpose
/// Loads a vector of a specified byte length.
///
/// ## Result value
/// Loads the number of bytes specified by b from the address specified in a.
/// Initializes elements in order from the byte stream (as defined by the endianness of the
/// target). Any bytes of elements that cannot be initialized from the number of loaded bytes have
/// a zero value.
///
/// Between 0 and 16 bytes, inclusive, will be loaded. The length is specified by the
/// least-significant byte of b, as min (b mod 256, 16). The behavior is undefined if the length
/// argument is outside of the range 0–255, or if it is not a multiple of the vector element size.
///
/// ## Notes
/// vec_xl_len should not be used to load from cache-inhibited memory.
#[inline]
#[target_feature(enable = "power9-vector")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_xl_len<T>(p: T, len: usize) -> <T as sealed::VectorXloads>::Result
where
    T: sealed::VectorXloads,
{
    p.vec_xl_len(len)
}

/// Vector Store with Length
///
/// ## Purpose
///
/// Stores a vector of a specified byte length.
///
/// ## Operation
///
/// Stores the number of bytes specified by c of the vector a to the address specified
/// in b. The bytes are obtained starting from the lowest-numbered byte of the lowest-numbered
/// element (as defined by the endianness of the target). All bytes of an element are accessed
/// before proceeding to the next higher element.
///
/// Between 0 and 16 bytes, inclusive, will be stored. The length is specified by the
/// least-significant byte of c, as min (c mod 256, 16). The behavior is undefined if the length
/// argument is outside of the range 0–255, or if it is not a multiple of the vector element size.
///
/// ## Notes
/// vec_xst_len should not be used to store to cache-inhibited memory.
#[inline]
#[target_feature(enable = "power9-vector")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_xst_len<T>(v: T, a: <T as sealed::VectorXstores>::Out, l: usize)
where
    T: sealed::VectorXstores,
{
    v.vec_xst_len(a, l)
}
