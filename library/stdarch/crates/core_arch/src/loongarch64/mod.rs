//! `LoongArch64` intrinsics

mod lasx;
mod lsx;

#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub use self::lasx::*;
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub use self::lsx::*;

use crate::arch::asm;

/// Reads the 64-bit stable counter value and the counter ID
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn rdtime_d() -> (i64, isize) {
    let (val, tid): (i64, isize);
    unsafe { asm!("rdtime.d {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack)) };
    (val, tid)
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.crc.w.d.w"]
    fn __crc_w_d_w(a: i64, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.d.w"]
    fn __crcc_w_d_w(a: i64, b: i32) -> i32;
    #[link_name = "llvm.loongarch.cacop.d"]
    fn __cacop(a: i64, b: i64, c: i64);
    #[link_name = "llvm.loongarch.csrrd.d"]
    fn __csrrd(a: i32) -> i64;
    #[link_name = "llvm.loongarch.csrwr.d"]
    fn __csrwr(a: i64, b: i32) -> i64;
    #[link_name = "llvm.loongarch.csrxchg.d"]
    fn __csrxchg(a: i64, b: i64, c: i32) -> i64;
    #[link_name = "llvm.loongarch.iocsrrd.d"]
    fn __iocsrrd_d(a: i32) -> i64;
    #[link_name = "llvm.loongarch.iocsrwr.d"]
    fn __iocsrwr_d(a: i64, b: i32);
    #[link_name = "llvm.loongarch.asrtle.d"]
    fn __asrtle(a: i64, b: i64);
    #[link_name = "llvm.loongarch.asrtgt.d"]
    fn __asrtgt(a: i64, b: i64);
    #[link_name = "llvm.loongarch.lddir.d"]
    fn __lddir(a: i64, b: i64) -> i64;
    #[link_name = "llvm.loongarch.ldpte.d"]
    fn __ldpte(a: i64, b: i64);
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crc_w_d_w(a: i64, b: i32) -> i32 {
    unsafe { __crc_w_d_w(a, b) }
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crcc_w_d_w(a: i64, b: i32) -> i32 {
    unsafe { __crcc_w_d_w(a, b) }
}

/// Generates the cache operation instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn cacop<const IMM5: i64, const IMM_S12: i64>(b: i64) {
    static_assert_uimm_bits!(IMM5, 5);
    static_assert_simm_bits!(IMM_S12, 12);
    __cacop(IMM5, b, IMM_S12);
}

/// Reads the CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrrd<const IMM14: i32>() -> i64 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrrd(IMM14)
}

/// Writes the CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrwr<const IMM14: i32>(a: i64) -> i64 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrwr(a, IMM14)
}

/// Exchanges the CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrxchg<const IMM14: i32>(a: i64, b: i64) -> i64 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrxchg(a, b, IMM14)
}

/// Reads the 64-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrrd_d(a: i32) -> i64 {
    __iocsrrd_d(a)
}

/// Writes the 64-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrwr_d(a: i64, b: i32) {
    __iocsrwr_d(a, b)
}

/// Generates the less-than-or-equal asseration instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn asrtle(a: i64, b: i64) {
    __asrtle(a, b);
}

/// Generates the greater-than asseration instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn asrtgt(a: i64, b: i64) {
    __asrtgt(a, b);
}

/// Loads the page table directory entry
#[inline]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lddir<const IMM8: i64>(a: i64) -> i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lddir(a, IMM8)
}

/// Loads the page table entry
#[inline]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn ldpte<const IMM8: i64>(a: i64) {
    static_assert_uimm_bits!(IMM8, 8);
    __ldpte(a, IMM8)
}
