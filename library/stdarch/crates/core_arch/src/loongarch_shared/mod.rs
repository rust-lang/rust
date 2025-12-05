//! `Shared LoongArch` intrinsics

use crate::arch::asm;

/// Reads the lower 32-bit stable counter value and the counter ID
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn rdtimel_w() -> (i32, isize) {
    let (val, tid): (i32, isize);
    unsafe { asm!("rdtimel.w {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack)) };
    (val, tid)
}

/// Reads the upper 32-bit stable counter value and the counter ID
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn rdtimeh_w() -> (i32, isize) {
    let (val, tid): (i32, isize);
    unsafe { asm!("rdtimeh.w {}, {}", out(reg) val, out(reg) tid, options(readonly, nostack)) };
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
    #[link_name = "llvm.loongarch.crcc.w.b.w"]
    fn __crcc_w_b_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.h.w"]
    fn __crcc_w_h_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.crcc.w.w.w"]
    fn __crcc_w_w_w(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.dbar"]
    fn __dbar(a: i32);
    #[link_name = "llvm.loongarch.ibar"]
    fn __ibar(a: i32);
    #[link_name = "llvm.loongarch.movgr2fcsr"]
    fn __movgr2fcsr(a: i32, b: i32);
    #[link_name = "llvm.loongarch.movfcsr2gr"]
    fn __movfcsr2gr(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.b"]
    fn __iocsrrd_b(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.h"]
    fn __iocsrrd_h(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrrd.w"]
    fn __iocsrrd_w(a: i32) -> i32;
    #[link_name = "llvm.loongarch.iocsrwr.b"]
    fn __iocsrwr_b(a: i32, b: i32);
    #[link_name = "llvm.loongarch.iocsrwr.h"]
    fn __iocsrwr_h(a: i32, b: i32);
    #[link_name = "llvm.loongarch.iocsrwr.w"]
    fn __iocsrwr_w(a: i32, b: i32);
    #[link_name = "llvm.loongarch.break"]
    fn __break(a: i32);
    #[link_name = "llvm.loongarch.cpucfg"]
    fn __cpucfg(a: i32) -> i32;
    #[link_name = "llvm.loongarch.syscall"]
    fn __syscall(a: i32);
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
pub fn crc_w_b_w(a: i32, b: i32) -> i32 {
    unsafe { __crc_w_b_w(a, b) }
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crc_w_h_w(a: i32, b: i32) -> i32 {
    unsafe { __crc_w_h_w(a, b) }
}

/// Calculate the CRC value using the IEEE 802.3 polynomial (0xEDB88320)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crc_w_w_w(a: i32, b: i32) -> i32 {
    unsafe { __crc_w_w_w(a, b) }
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crcc_w_b_w(a: i32, b: i32) -> i32 {
    unsafe { __crcc_w_b_w(a, b) }
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crcc_w_h_w(a: i32, b: i32) -> i32 {
    unsafe { __crcc_w_h_w(a, b) }
}

/// Calculate the CRC value using the Castagnoli polynomial (0x82F63B78)
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn crcc_w_w_w(a: i32, b: i32) -> i32 {
    unsafe { __crcc_w_w_w(a, b) }
}

/// Generates the memory barrier instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn dbar<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    unsafe { __dbar(IMM15) };
}

/// Generates the instruction-fetch barrier instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn ibar<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    unsafe { __ibar(IMM15) };
}

/// Moves data from a GPR to the FCSR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn movgr2fcsr<const IMM2: i32>(a: i32) {
    static_assert_uimm_bits!(IMM2, 2);
    __movgr2fcsr(IMM2, a);
}

/// Moves data from a FCSR to the GPR
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn movfcsr2gr<const IMM2: i32>() -> i32 {
    static_assert_uimm_bits!(IMM2, 2);
    unsafe { __movfcsr2gr(IMM2) }
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
pub fn cpucfg(a: i32) -> i32 {
    unsafe { __cpucfg(a) }
}

/// Generates the syscall instruction
#[inline]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn syscall<const IMM15: i32>() {
    static_assert_uimm_bits!(IMM15, 15);
    __syscall(IMM15);
}

/// Calculate the approximate single-precision result of 1.0 divided
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn frecipe_s(a: f32) -> f32 {
    unsafe { __frecipe_s(a) }
}

/// Calculate the approximate double-precision result of 1.0 divided
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn frecipe_d(a: f64) -> f64 {
    unsafe { __frecipe_d(a) }
}

/// Calculate the approximate single-precision result of dividing 1.0 by the square root
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn frsqrte_s(a: f32) -> f32 {
    unsafe { __frsqrte_s(a) }
}

/// Calculate the approximate double-precision result of dividing 1.0 by the square root
#[inline]
#[target_feature(enable = "frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn frsqrte_d(a: f64) -> f64 {
    unsafe { __frsqrte_d(a) }
}
