#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.riscv.orc.b.i32"]
    fn _orc_b_32(rs: u32) -> u32;

    #[link_name = "llvm.riscv.clmul.i32"]
    fn _clmul_32(rs1: u32, rs2: u32) -> u32;
}

/// Bitwise OR-Combine, byte granule
///
/// Combines the bits within every byte through a reciprocal bitwise logical OR. This sets the bits of each byte in
/// the result rd to all zeros if no bit within the respective byte of rs is set, or to all ones if any bit within the
/// respective byte of rs is set.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.24
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbb")]
#[cfg_attr(test, assert_instr(orc.b))]
#[inline]
pub fn orc_b_32(rs: u32) -> u32 {
    unsafe { _orc_b_32(rs) }
}

/// Carry-less multiply (low-part)
///
/// clmul produces the lower half of the 32-bit carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.11
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbkc")]
#[cfg_attr(test, assert_instr(clmul))]
#[inline]
pub fn clmul_32(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmul_32(rs1, rs2) }
}
