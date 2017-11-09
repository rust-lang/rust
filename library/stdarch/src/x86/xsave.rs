//! `xsave` and `xsaveopt` target feature intrinsics

#![cfg_attr(feature = "cargo-clippy", allow(stutter))]

#[cfg(test)]
use stdsimd_test::assert_instr;

use x86::c_void;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.xsave"]
    fn xsave(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xrstor"]
    fn xrstor(p: *const c_void, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsetbv"]
    fn xsetbv(v: i32, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xgetbv"]
    fn xgetbv(x: i32) -> i64;
    #[link_name = "llvm.x86.xsave64"]
    fn xsave64(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xrstor64"]
    fn xrstor64(p: *const c_void, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsaveopt"]
    fn xsaveopt(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsaveopt64"]
    fn xsaveopt64(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsavec"]
    fn xsavec(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsavec64"]
    fn xsavec64(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsaves"]
    fn xsaves(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xsaves64"]
    fn xsaves64(p: *mut i8, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xrstors"]
    fn xrstors(p: *const c_void, hi: i32, lo: i32) -> ();
    #[link_name = "llvm.x86.xrstors64"]
    fn xrstors64(p: *const c_void, hi: i32, lo: i32) -> ();
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits [62:0] in `save_mask` and XCR0.
/// `mem_addr` must be aligned on a 64-byte boundary.
///
/// The format of the XSAVE area is detailed in Section 13.4, “XSAVE Area,” of
/// Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 1.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xsave))]
pub unsafe fn _xsave(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsave(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial restore of the enabled processor states using
/// the state information stored in memory at `mem_addr`.
///
/// State is restored based on bits [62:0] in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xrstor))]
pub unsafe fn _xrstor(mem_addr: *const c_void, rs_mask: u64) -> () {
    xrstor(mem_addr, (rs_mask >> 32) as i32, rs_mask as i32);
}

/// `XFEATURE_ENABLED_MASK` for `XCR`
///
/// This intrinsic maps to `XSETBV` instruction.
const _XCR_XFEATURE_ENABLED_MASK: u32 = 0;

/// Copy 64-bits from `val` to the extended control register (`XCR`) specified
/// by `a`.
///
/// Currently only `XFEATURE_ENABLED_MASK` `XCR` is supported.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xsetbv))]
pub unsafe fn _xsetbv(a: u32, val: u64) -> () {
    xsetbv(a as i32, (val >> 32) as i32, val as i32);
}

/// Reads the contents of the extended control register `XCR`
/// specified in `xcr_no`.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xgetbv))]
pub unsafe fn _xgetbv(xcr_no: u32) -> u64 {
    xgetbv(xcr_no as i32) as u64
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits [62:0] in `save_mask` and XCR0.
/// `mem_addr` must be aligned on a 64-byte boundary.
///
/// The format of the XSAVE area is detailed in Section 13.4, “XSAVE Area,” of
/// Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 1.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xsave64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xsave64(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsave64(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial restore of the enabled processor states using
/// the state information stored in memory at `mem_addr`.
///
/// State is restored based on bits [62:0] in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
#[inline(always)]
#[target_feature = "+xsave"]
#[cfg_attr(test, assert_instr(xrstor64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xrstor64(mem_addr: *const c_void, rs_mask: u64) -> () {
    xrstor64(mem_addr, (rs_mask >> 32) as i32, rs_mask as i32);
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits [62:0] in `save_mask` and `XCR0`.
/// `mem_addr` must be aligned on a 64-byte boundary. The hardware may optimize
/// the manner in which data is saved. The performance of this instruction will
/// be equal to or better than using the `XSAVE` instruction.
#[inline(always)]
#[target_feature = "+xsave,+xsaveopt"]
#[cfg_attr(test, assert_instr(xsaveopt))]
pub unsafe fn _xsaveopt(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsaveopt(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`.
///
/// State is saved based on bits [62:0] in `save_mask` and `XCR0`.
/// `mem_addr` must be aligned on a 64-byte boundary. The hardware may optimize
/// the manner in which data is saved. The performance of this instruction will
/// be equal to or better than using the `XSAVE64` instruction.
#[inline(always)]
#[target_feature = "+xsave,+xsaveopt"]
#[cfg_attr(test, assert_instr(xsaveopt64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xsaveopt64(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsaveopt64(
        mem_addr as *mut i8,
        (save_mask >> 32) as i32,
        save_mask as i32,
    );
}

/// Perform a full or partial save of the enabled processor states to memory
/// at `mem_addr`.
///
/// `xsavec` differs from `xsave` in that it uses compaction and that it may
/// use init optimization. State is saved based on bits [62:0] in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsavec"]
#[cfg_attr(test, assert_instr(xsavec))]
pub unsafe fn _xsavec(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsavec(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial save of the enabled processor states to memory
/// at `mem_addr`.
///
/// `xsavec` differs from `xsave` in that it uses compaction and that it may
/// use init optimization. State is saved based on bits [62:0] in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsavec"]
#[cfg_attr(test, assert_instr(xsavec64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xsavec64(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsavec64(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`
///
/// `xsaves` differs from xsave in that it can save state components
/// corresponding to bits set in `IA32_XSS` `MSR` and that it may use the
/// modified optimization. State is saved based on bits [62:0] in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsaves"]
#[cfg_attr(test, assert_instr(xsaves))]
pub unsafe fn _xsaves(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsaves(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial save of the enabled processor states to memory at
/// `mem_addr`
///
/// `xsaves` differs from xsave in that it can save state components
/// corresponding to bits set in `IA32_XSS` `MSR` and that it may use the
/// modified optimization. State is saved based on bits [62:0] in `save_mask`
/// and `XCR0`. `mem_addr` must be aligned on a 64-byte boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsaves"]
#[cfg_attr(test, assert_instr(xsaves64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xsaves64(mem_addr: *mut c_void, save_mask: u64) -> () {
    xsaves64(mem_addr as *mut i8, (save_mask >> 32) as i32, save_mask as i32);
}

/// Perform a full or partial restore of the enabled processor states using the
/// state information stored in memory at `mem_addr`.
///
/// `xrstors` differs from `xrstor` in that it can restore state components
/// corresponding to bits set in the `IA32_XSS` `MSR`; `xrstors` cannot restore
/// from an `xsave` area in which the extended region is in the standard form.
/// State is restored based on bits [62:0] in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsaves"]
#[cfg_attr(test, assert_instr(xrstors))]
pub unsafe fn _xrstors(mem_addr: *const c_void, rs_mask: u64) -> () {
    xrstors(mem_addr, (rs_mask >> 32) as i32, rs_mask as i32);
}
/// Perform a full or partial restore of the enabled processor states using the
/// state information stored in memory at `mem_addr`.
///
/// `xrstors` differs from `xrstor` in that it can restore state components
/// corresponding to bits set in the `IA32_XSS` `MSR`; `xrstors` cannot restore
/// from an `xsave` area in which the extended region is in the standard form.
/// State is restored based on bits [62:0] in `rs_mask`, `XCR0`, and
/// `mem_addr.HEADER.XSTATE_BV`. `mem_addr` must be aligned on a 64-byte
/// boundary.
#[inline(always)]
#[target_feature = "+xsave,+xsaves"]
#[cfg_attr(test, assert_instr(xrstors64))]
#[cfg(not(target_arch = "x86"))]
pub unsafe fn _xrstors64(mem_addr: *const c_void, rs_mask: u64) -> () {
    xrstors64(mem_addr, (rs_mask >> 32) as i32, rs_mask as i32);
}


#[cfg(test)]
mod tests {
    use super::*;
    use stdsimd_test::simd_test;
    use std::fmt;

    #[repr(align(64))]
    struct Buffer {
        data: [u64; 1024], // 8192 bytes
    }

    impl Buffer {
        fn new() -> Buffer {
            Buffer { data: [0; 1024] }
        }
        fn ptr(&mut self) -> *mut c_void {
            &mut self.data[0] as *mut _ as *mut c_void
        }
    }

    impl PartialEq<Buffer> for Buffer {
        fn eq(&self, other: &Buffer) -> bool {
            for i in 0..1024 {
                if self.data[i] != other.data[i] {
                    return false;
                }
            }
            true
        }
    }

    impl fmt::Debug for Buffer {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "[")?;
            for i in 0..1024 {
                write!(f, "{}", self.data[i])?;
                if i != 1023 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")
        }
    }

    #[simd_test = "xsave"]
    unsafe fn xsave() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsave(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsave(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[cfg(not(target_arch = "x86"))]
    #[simd_test = "xsave"]
    unsafe fn xsave64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsave64(a.ptr(), m);
        _xrstor64(a.ptr(), m);
        _xsave64(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[simd_test = "xsave"]
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

    #[simd_test = "xsave,xsaveopt"]
    unsafe fn xsaveopt() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsaveopt(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsaveopt(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[cfg(not(target_arch = "x86"))]
    #[simd_test = "xsave,xsaveopt"]
    unsafe fn xsaveopt64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsaveopt64(a.ptr(), m);
        _xrstor64(a.ptr(), m);
        _xsaveopt64(b.ptr(), m);
        assert_eq!(a, b);
    }


    #[simd_test = "xsave,xsavec"]
    unsafe fn xsavec() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsavec(a.ptr(), m);
        _xrstor(a.ptr(), m);
        _xsavec(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[cfg(not(target_arch = "x86"))]
    #[simd_test = "xsave,xsavec"]
    unsafe fn xsavec64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsavec64(a.ptr(), m);
        _xrstor64(a.ptr(), m);
        _xsavec64(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[cfg(not(feature = "intel_sde"))]
    #[simd_test = "xsaves"]
    unsafe fn xsaves() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsaves(a.ptr(), m);
        _xrstors(a.ptr(), m);
        _xsaves(b.ptr(), m);
        assert_eq!(a, b);
    }

    #[cfg(not(any(target_arch = "x86", feature = "intel_sde")))]
    #[simd_test = "xsaves"]
    unsafe fn xsaves64() {
        let m = 0xFFFFFFFFFFFFFFFF_u64; //< all registers
        let mut a = Buffer::new();
        let mut b = Buffer::new();

        _xsaves64(a.ptr(), m);
        _xrstors64(a.ptr(), m);
        _xsaves64(b.ptr(), m);
        assert_eq!(a, b);
    }
}
