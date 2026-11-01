//! RISC-V RV64 specific intrinsics
use crate::arch::asm;

mod zk;

#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub use zk::*;

/// Loads virtual machine memory by unsigned word integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// This operation is not available under RV32 base instruction set.
///
/// This function is unsafe for it accesses the virtual supervisor or user via a `HLV.WU`
/// instruction which is effectively a dereference to any memory address.
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub unsafe fn hlv_wu(src: *const u32) -> u32 {
    let value: u32;
    asm!(
        ".insn i 0x73, 0x4, {}, {}, 0x681",
        lateout(reg) value,
        in(reg) src,
        options(readonly, nostack, preserves_flags)
    );
    value
}

/// Loads virtual machine memory by double integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// This operation is not available under RV32 base instruction set.
///
/// This function is unsafe for it accesses the virtual supervisor or user via a `HLV.D`
/// instruction which is effectively a dereference to any memory address.
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub unsafe fn hlv_d(src: *const i64) -> i64 {
    let value: i64;
    asm!(
        ".insn i 0x73, 0x4, {}, {}, 0x6C0",
        lateout(reg) value,
        in(reg) src,
        options(readonly, nostack, preserves_flags)
    );
    value
}

/// Stores virtual machine memory by double integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// This function is unsafe for it accesses the virtual supervisor or user via a `HSV.D`
/// instruction which is effectively a dereference to any memory address.
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub unsafe fn hsv_d(dst: *mut i64, src: i64) {
    asm!(
        ".insn r 0x73, 0x4, 0x37, x0, {}, {}",
        in(reg) dst,
        in(reg) src,
        options(nostack, preserves_flags)
    );
}
