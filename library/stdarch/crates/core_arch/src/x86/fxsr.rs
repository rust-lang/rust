//! FXSR floating-point context fast save and restor.

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.fxsave"]
    fn fxsave(p: *mut u8);
    #[link_name = "llvm.x86.fxrstor"]
    fn fxrstor(p: *const u8);
}

/// Saves the `x87` FPU, `MMX` technology, `XMM`, and `MXCSR` registers to the
/// 512-byte-long 16-byte-aligned memory region `mem_addr`.
///
/// A misaligned destination operand raises a general-protection (#GP) or an
/// alignment check exception (#AC).
///
/// See [`FXSAVE`][fxsave] and [`FXRSTOR`][fxrstor].
///
/// [fxsave]: http://www.felixcloutier.com/x86/FXSAVE.html
/// [fxrstor]: http://www.felixcloutier.com/x86/FXRSTOR.html
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_fxsave)
#[inline]
#[target_feature(enable = "fxsr")]
#[cfg_attr(test, assert_instr(fxsave))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _fxsave(mem_addr: *mut u8) {
    fxsave(mem_addr)
}

/// Restores the `XMM`, `MMX`, `MXCSR`, and `x87` FPU registers from the
/// 512-byte-long 16-byte-aligned memory region `mem_addr`.
///
/// The contents of this memory region should have been written to by a
/// previous
/// `_fxsave` or `_fxsave64` intrinsic.
///
/// A misaligned destination operand raises a general-protection (#GP) or an
/// alignment check exception (#AC).
///
/// See [`FXSAVE`][fxsave] and [`FXRSTOR`][fxrstor].
///
/// [fxsave]: http://www.felixcloutier.com/x86/FXSAVE.html
/// [fxrstor]: http://www.felixcloutier.com/x86/FXRSTOR.html
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_fxrstor)
#[inline]
#[target_feature(enable = "fxsr")]
#[cfg_attr(test, assert_instr(fxrstor))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _fxrstor(mem_addr: *const u8) {
    fxrstor(mem_addr)
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use std::{cmp::PartialEq, fmt};
    use stdarch_test::simd_test;

    #[repr(align(16))]
    struct FxsaveArea {
        data: [u8; 512], // 512 bytes
    }

    impl FxsaveArea {
        fn new() -> FxsaveArea {
            FxsaveArea { data: [0; 512] }
        }
        fn ptr(&mut self) -> *mut u8 {
            &mut self.data[0] as *mut _ as *mut u8
        }
    }

    impl PartialEq<FxsaveArea> for FxsaveArea {
        fn eq(&self, other: &FxsaveArea) -> bool {
            for i in 0..self.data.len() {
                if self.data[i] != other.data[i] {
                    return false;
                }
            }
            true
        }
    }

    impl fmt::Debug for FxsaveArea {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "[")?;
            for i in 0..self.data.len() {
                write!(f, "{}", self.data[i])?;
                if i != self.data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")
        }
    }

    #[simd_test(enable = "fxsr")]
    unsafe fn fxsave() {
        let mut a = FxsaveArea::new();
        let mut b = FxsaveArea::new();

        fxsr::_fxsave(a.ptr());
        fxsr::_fxrstor(a.ptr());
        fxsr::_fxsave(b.ptr());
        assert_eq!(a, b);
    }
}
