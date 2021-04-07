//! `x86_64`'s `xsave` and `xsaveopt` target feature intrinsics

#![allow(clippy::module_name_repetitions)]

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.xsave64"]
    fn xsave64(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xrstor64"]
    fn xrstor64(p: *const u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsaveopt64"]
    fn xsaveopt64(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsavec64"]
    fn xsavec64(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsaves64"]
    fn xsaves64(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xrstors64"]
    fn xrstors64(p: *const u8, hi: u32, lo: u32);
}

/// Performs a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits `[62:0]` in `save_mask` and XCR0.
/// `mem_addr` must be aligned on a 64-byte boundary.
///
/// The format of the XSAVE area is detailed in Section 13.4, “XSAVE Area,” of
/// Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 1.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsave64)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xsave64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsave64(mem_addr: *mut u8, save_mask: u64) {
    xsave64(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial restore of the enabled processor states using
/// the state information stored in memory at `mem_addr`.
///
/// State is restored based on bits `[62:0]` in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xrstor64)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xrstor64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xrstor64(mem_addr: *const u8, rs_mask: u64) {
    xrstor64(mem_addr, (rs_mask >> 32) as u32, rs_mask as u32);
}

/// Performs a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits `[62:0]` in `save_mask` and `XCR0`.
/// `mem_addr` must be aligned on a 64-byte boundary. The hardware may optimize
/// the manner in which data is saved. The performance of this instruction will
/// be equal to or better than using the `XSAVE64` instruction.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsaveopt64)
#[inline]
#[target_feature(enable = "xsave,xsaveopt")]
#[cfg_attr(test, assert_instr(xsaveopt64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsaveopt64(mem_addr: *mut u8, save_mask: u64) {
    xsaveopt64(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial save of the enabled processor states to memory
/// at `mem_addr`.
///
/// `xsavec` differs from `xsave` in that it uses compaction and that it may
/// use init optimization. State is saved based on bits `[62:0]` in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsavec64)
#[inline]
#[target_feature(enable = "xsave,xsavec")]
#[cfg_attr(test, assert_instr(xsavec64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsavec64(mem_addr: *mut u8, save_mask: u64) {
    xsavec64(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial save of the enabled processor states to memory at
/// `mem_addr`
///
/// `xsaves` differs from xsave in that it can save state components
/// corresponding to bits set in `IA32_XSS` `MSR` and that it may use the
/// modified optimization. State is saved based on bits `[62:0]` in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsaves64)
#[inline]
#[target_feature(enable = "xsave,xsaves")]
#[cfg_attr(test, assert_instr(xsaves64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsaves64(mem_addr: *mut u8, save_mask: u64) {
    xsaves64(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial restore of the enabled processor states using the
/// state information stored in memory at `mem_addr`.
///
/// `xrstors` differs from `xrstor` in that it can restore state components
/// corresponding to bits set in the `IA32_XSS` `MSR`; `xrstors` cannot restore
/// from an `xsave` area in which the extended region is in the standard form.
/// State is restored based on bits `[62:0]` in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xrstors64)
#[inline]
#[target_feature(enable = "xsave,xsaves")]
#[cfg_attr(test, assert_instr(xrstors64))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xrstors64(mem_addr: *const u8, rs_mask: u64) {
    xrstors64(mem_addr, (rs_mask >> 32) as u32, rs_mask as u32);
}

// FIXME: https://github.com/rust-lang/stdarch/issues/209
// All these tests fail with Intel SDE.
/*
#[cfg(test)]
mod tests {
    use crate::core_arch::x86::x86_64::xsave;
    use stdarch_test::simd_test;
    use std::fmt;

    // FIXME: https://github.com/rust-lang/stdarch/issues/209
    #[repr(align(64))]
    struct XsaveArea {
        // max size for 256-bit registers is 800 bytes:
        // see https://software.intel.com/en-us/node/682996
        // max size for 512-bit registers is 2560 bytes:
        // FIXME: add source
        data: [u8; 2560],
    }

    impl XsaveArea {
        fn new() -> XsaveArea {
            XsaveArea { data: [0; 2560] }
        }
        fn ptr(&mut self) -> *mut u8 {
            &mut self.data[0] as *mut _ as *mut u8
        }
    }

    impl PartialEq<XsaveArea> for XsaveArea {
        fn eq(&self, other: &XsaveArea) -> bool {
            for i in 0..self.data.len() {
                if self.data[i] != other.data[i] {
                    return false;
                }
            }
            true
        }
    }

    impl fmt::Debug for XsaveArea {
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

    #[simd_test(enable = "xsave")]
    unsafe fn xsave64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        xsave::_xsave64(a.ptr(), m);
        xsave::_xrstor64(a.ptr(), m);
        xsave::_xsave64(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[simd_test(enable = "xsave,xsaveopt")]
    unsafe fn xsaveopt64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        xsave::_xsaveopt64(a.ptr(), m);
        xsave::_xrstor64(a.ptr(), m);
        xsave::_xsaveopt64(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[simd_test(enable = "xsave,xsavec")]
    unsafe fn xsavec64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        xsave::_xsavec64(a.ptr(), m);
        xsave::_xrstor64(a.ptr(), m);
        xsave::_xsavec64(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[simd_test(enable = "xsave,xsaves")]
    unsafe fn xsaves64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        xsave::_xsaves64(a.ptr(), m);
        xsave::_xrstors64(a.ptr(), m);
        xsave::_xsaves64(b.ptr(), m);
        assert_eq!(a, b);
    }
}
*/
