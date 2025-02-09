//! `LoongArch` intrinsics

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
pub unsafe fn rdtime_d() -> (i64, isize) {
    let val: i64;
    let tid: isize;
    asm!("rdtime.d {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack));
    (val, tid)
}

/// Reads the lower 32-bit stable counter value and the counter ID
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn rdtimel_w() -> (i32, isize) {
    let val: i32;
    let tid: isize;
    asm!("rdtimel.w {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack));
    (val, tid)
}

/// Reads the upper 32-bit stable counter value and the counter ID
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn rdtimeh_w() -> (i32, isize) {
    let val: i32;
    let tid: isize;
    asm!("rdtimeh.w {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack));
    (val, tid)
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.crc.w.b.w"]
    fn __crc_w_b_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crc.w.h.w"]
    fn __crc_w_h_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crc.w.w.w"]
    fn __crc_w_w_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crc.w.d.w"]
    fn __crc_w_d_w(a: i64, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.b.w"]
    fn __crcc_w_b_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.h.w"]
    fn __crcc_w_h_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.w.w"]
    fn __crcc_w_w_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.d.w"]
    fn __crcc_w_d_w(a: i64, b: i32) -> i32;
    #[link_name = "llvm.loongarch.cacop.d"]
    fn __cacop(a: i64, b: i64, c: i64);
    #[link_name = "llvm.loongarch.dbar"]
    fn __dbar(a: i32);
    #[link_name = "llvm.loongarch.ibar"]
    fn __ibar(a: i32);
    #[link_name = "llvm.loongarch.movgr2fcsr"]
    fn __movgr2fcsr(a: i32, b: i32);
    #[link_name = "llvm.loongarch.movfcsr2gr"]
    fn __movfcsr2gr(a: i32) -> i32;
    #[link_name = "llvm.loongarch.csrrd.d"]
    fn __csrrd(a: i32) -> i64;
    #[link_name = "llvm.loongarch.csrwr.d"]
    fn __csrwr(a: i64, b: i32) -> i64;
    #[link_name = "llvm.loongarch.csrxchg.d"]
    fn __csrxchg(a: i64, b: i64, c: i32) -> i64;
    #[link_name = "llvm.loongarch.iocsrrd.b"]
    fn __iocsrrd_b(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.h"]
    fn __iocsrrd_h(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.w"]
    fn __iocsrrd_w(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.d"]
    fn __iocsrrd_d(a: i32) -> i64;
    #[link_name = "llvm.loongarch.iocsrwr.b"]
    fn __iocsrwr_b(a: i32, b: i32);
    #[link_name = "llvm.loongarch.iocsrwr.h"]
    fn __iocsrwr_h(a: i32, b: i32);
    #[link_name = "llvm.loongarch.iocsrwr.w"]
    fn __iocsrwr_w(a: i32, b: i32);
    #[link_name = "llvm.loongarch.iocsrwr.d"]
    fn __iocsrwr_d(a: i64, b: i32);
    #[link_name = "llvm.loongarch.break"]
    fn __break(a: i32);
    #[link_name = "llvm.loongarch.cpucfg"]
    fn __cpucfg(a: i32) -> i32;
    #[link_name = "llvm.loongarch.syscall"]
    fn __syscall(a: i32);
    #[link_name = "llvm.loongarch.asrtle.d"]
    fn __asrtle(a: i64, b: i64);
    #[link_name = "llvm.loongarch.asrtgt.d"]
    fn __asrtgt(a: i64, b: i64);
    #[link_name = "llvm.loongarch.lddir.d"]
    fn __lddir(a: i64, b: i64) -> i64;
    #[link_name = "llvm.loongarch.ldpte.d"]
    fn __ldpte(a: i64, b: i64);
    #[link_name = "llvm.loongarch.frecipe.s"]
    fn __frecipe_s(a: f32) -> f32;
    #[link_name = "llvm.loongarch.frecipe.d"]
    fn __frecipe_d(a: f64) -> f64;
    #[link_name = "llvm.loongarch.frsqrte.s"]
    fn __frsqrte_s(a: f32) -> f32;
    #[link_name = "llvm.loongarch.frsqrte.d"]
    fn __frsqrte_d(a: f64) -> f64;
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crc_w_b_w(a: i32, b: i32) -> i32 {
    __crc_w_b_w(a, b)
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crc_w_h_w(a: i32, b: i32) -> i32 {
    __crc_w_h_w(a, b)
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crc_w_w_w(a: i32, b: i32) -> i32 {
    __crc_w_w_w(a, b)
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crc_w_d_w(a: i64, b: i32) -> i32 {
    __crc_w_d_w(a, b)
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crcc_w_b_w(a: i32, b: i32) -> i32 {
    __crcc_w_b_w(a, b)
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crcc_w_h_w(a: i32, b: i32) -> i32 {
    __crcc_w_h_w(a, b)
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crcc_w_w_w(a: i32, b: i32) -> i32 {
    __crcc_w_w_w(a, b)
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn crcc_w_d_w(a: i64, b: i32) -> i32 {
    __crcc_w_d_w(a, b)
}

/// Generates the cache operation instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn cacop<const IMM12: i64>(a: i64, b: i64) {
    static_assert_simm_bits!(IMM12, 12);
    __cacop(a, b, IMM12);
}

/// Generates the memory barrier instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn dbar<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    __dbar(IMM15);
}

/// Generates the instruction-fetch barrier instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn ibar<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    __ibar(IMM15);
}

/// Moves data from a GPR to the FCSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn movgr2fcsr<const IMM5: i32>(a: i32) {
    static_assert_uimm_bits!(IMM5, 5);
    __movgr2fcsr(IMM5, a);
}

/// Moves data from a FCSR to the GPR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn movfcsr2gr<const IMM5: i32>() -> i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __movfcsr2gr(IMM5)
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

/// Reads the 8-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrrd_b(a: i32) -> i32 {
    __iocsrrd_b(a)
}

/// Reads the 16-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrrd_h(a: i32) -> i32 {
    __iocsrrd_h(a)
}

/// Reads the 32-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrrd_w(a: i32) -> i32 {
    __iocsrrd_w(a)
}

/// Reads the 64-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrrd_d(a: i32) -> i64 {
    __iocsrrd_d(a)
}

/// Writes the 8-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrwr_b(a: i32, b: i32) {
    __iocsrwr_b(a, b)
}

/// Writes the 16-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrwr_h(a: i32, b: i32) {
    __iocsrwr_h(a, b)
}

/// Writes the 32-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrwr_w(a: i32, b: i32) {
    __iocsrwr_w(a, b)
}

/// Writes the 64-bit IO-CSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn iocsrwr_d(a: i64, b: i32) {
    __iocsrwr_d(a, b)
}

/// Generates the breakpoint instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn brk<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    __break(IMM15);
}

/// Reads the CPU configuration register
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn cpucfg(a: i32) -> i32 {
    __cpucfg(a)
}

/// Generates the syscall instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn syscall<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    __syscall(IMM15);
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
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lddir(a: i64, b: i64) -> i64 {
    __lddir(a, b)
}

/// Loads the page table entry
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn ldpte(a: i64, b: i64) {
    __ldpte(a, b)
}

/// Calculate the approximate single-precision result of 1.0 divided
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn frecipe_s(a: f32) -> f32 {
    __frecipe_s(a)
}

/// Calculate the approximate double-precision result of 1.0 divided
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn frecipe_d(a: f64) -> f64 {
    __frecipe_d(a)
}

/// Calculate the approximate single-precision result of dividing 1.0 by the square root
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn frsqrte_s(a: f32) -> f32 {
    __frsqrte_s(a)
}

/// Calculate the approximate double-precision result of dividing 1.0 by the square root
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn frsqrte_d(a: f64) -> f64 {
    __frsqrte_d(a)
}
