//! `i586`'s `xsave` and `xsaveopt` target feature intrinsics
#![allow(clippy::module_name_repetitions)]

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.xsave"]
    fn xsave(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xrstor"]
    fn xrstor(p: *const u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsetbv"]
    fn xsetbv(v: u32, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xgetbv"]
    fn xgetbv(v: u32) -> i64;
    #[link_name = "llvm.x86.xsaveopt"]
    fn xsaveopt(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsavec"]
    fn xsavec(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xsaves"]
    fn xsaves(p: *mut u8, hi: u32, lo: u32);
    #[link_name = "llvm.x86.xrstors"]
    fn xrstors(p: *const u8, hi: u32, lo: u32);
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
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsave)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xsave))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsave(mem_addr: *mut u8, save_mask: u64) {
    xsave(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial restore of the enabled processor states using
/// the state information stored in memory at `mem_addr`.
///
/// State is restored based on bits `[62:0]` in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xrstor)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xrstor))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xrstor(mem_addr: *const u8, rs_mask: u64) {
    xrstor(mem_addr, (rs_mask >> 32) as u32, rs_mask as u32);
}

/// `XFEATURE_ENABLED_MASK` for `XCR`
///
/// This intrinsic maps to `XSETBV` instruction.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _XCR_XFEATURE_ENABLED_MASK: u32 = 0;

/// Copies 64-bits from `val` to the extended control register (`XCR`) specified
/// by `a`.
///
/// Currently only `XFEATURE_ENABLED_MASK` `XCR` is supported.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsetbv)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xsetbv))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsetbv(a: u32, val: u64) {
    xsetbv(a, (val >> 32) as u32, val as u32);
}

/// Reads the contents of the extended control register `XCR`
/// specified in `xcr_no`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xgetbv)
#[inline]
#[target_feature(enable = "xsave")]
#[cfg_attr(test, assert_instr(xgetbv))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xgetbv(xcr_no: u32) -> u64 {
    xgetbv(xcr_no) as u64
}

/// Performs a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits `[62:0]` in `save_mask` and `XCR0`.
/// `mem_addr` must be aligned on a 64-byte boundary. The hardware may optimize
/// the manner in which data is saved. The performance of this instruction will
/// be equal to or better than using the `XSAVE` instruction.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsaveopt)
#[inline]
#[target_feature(enable = "xsave,xsaveopt")]
#[cfg_attr(test, assert_instr(xsaveopt))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsaveopt(mem_addr: *mut u8, save_mask: u64) {
    xsaveopt(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial save of the enabled processor states to memory
/// at `mem_addr`.
///
/// `xsavec` differs from `xsave` in that it uses compaction and that it may
/// use init optimization. State is saved based on bits `[62:0]` in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsavec)
#[inline]
#[target_feature(enable = "xsave,xsavec")]
#[cfg_attr(test, assert_instr(xsavec))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsavec(mem_addr: *mut u8, save_mask: u64) {
    xsavec(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
}

/// Performs a full or partial save of the enabled processor states to memory at
/// `mem_addr`
///
/// `xsaves` differs from xsave in that it can save state components
/// corresponding to bits set in `IA32_XSS` `MSR` and that it may use the
/// modified optimization. State is saved based on bits `[62:0]` in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xsaves)
#[inline]
#[target_feature(enable = "xsave,xsaves")]
#[cfg_attr(test, assert_instr(xsaves))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xsaves(mem_addr: *mut u8, save_mask: u64) {
    xsaves(mem_addr, (save_mask >> 32) as u32, save_mask as u32);
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
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xrstors)
#[inline]
#[target_feature(enable = "xsave,xsaves")]
#[cfg_attr(test, assert_instr(xrstors))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _xrstors(mem_addr: *const u8, rs_mask: u64) {
    xrstors(mem_addr, (rs_mask >> 32) as u32, rs_mask as u32);
}

#[cfg(test)]
mod tests {
    use std::{fmt, prelude::v1::*};

    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

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

    // FIXME: https://github.com/rust-lang/stdarch/issues/209
    /*
    #[simd_test(enable = "xsave")]
    unsafe fn xsave() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        _xsave(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsave(b.ptr(), m);
        assert_eq!(a, b);
    }
    */

    #[simd_test(enable = "xsave")]
    unsafe fn xgetbv_xsetbv() {
        let xcr_n: u32 = _XCR_XFEATURE_ENABLED_MASK;

        let xcr: u64 = _xgetbv(xcr_n);
        // FIXME: XSETBV is a privileged instruction we should only test this
        // when running in privileged mode:
        //
        // _xsetbv(xcr_n, xcr);
        let xcr_cpy: u64 = _xgetbv(xcr_n);
        assert_eq!(xcr, xcr_cpy);
    }

    // FIXME: https://github.com/rust-lang/stdarch/issues/209
    /*
    #[simd_test(enable = "xsave,xsaveopt")]
    unsafe fn xsaveopt() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        _xsaveopt(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsaveopt(b.ptr(), m);
        assert_eq!(a, b);
    }
    */

    // FIXME: this looks like a bug in Intel's SDE:
    #[cfg(not(stdarch_intel_sde))]
    #[simd_test(enable = "xsave,xsavec")]
    unsafe fn xsavec() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        _xsavec(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsavec(b.ptr(), m);
        assert_eq!(a, b);
    }

    // FIXME: https://github.com/rust-lang/stdarch/issues/209
    /*
    #[simd_test(enable = "xsave,xsaves")]
    unsafe fn xsaves() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = XsaveArea::new();
        let mut b = XsaveArea::new();

        _xsaves(a.ptr(), m);
        _xrstors(a.ptr(), m);
        _xsaves(b.ptr(), m);
        assert_eq!(a, b);
    }
    */
}
