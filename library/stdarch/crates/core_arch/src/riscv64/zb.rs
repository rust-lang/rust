#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.riscv.xperm8.i64"]
    fn _xperm8_64(rs1: u64, rs2: u64) -> u64;

    #[link_name = "llvm.riscv.xperm4.i64"]
    fn _xperm4_64(rs1: u64, rs2: u64) -> u64;

    #[link_name = "llvm.riscv.orc.b.i64"]
    fn _orc_b_64(rs: u64) -> u64;

    #[link_name = "llvm.riscv.clmul.i64"]
    fn _clmul_64(rs1: u64, rs2: u64) -> u64;

    #[link_name = "llvm.riscv.clmulh.i64"]
    fn _clmulh_64(rs1: u64, rs2: u64) -> u64;

    #[link_name = "llvm.riscv.clmulr.i64"]
    fn _clmulr_64(rs1: u64, rs2: u64) -> u64;
}

/// Byte-wise lookup of indices into a vector in registers.
///
/// The xperm8 instruction operates on bytes. The rs1 register contains a vector of 8
/// 8-bit elements. The rs2 register contains a vector of 8 8-bit indexes. The result is
/// each element in rs2 replaced by the indexed element in rs1, or zero if the index into rs2
/// is out of bounds.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.47
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbkx")]
#[cfg_attr(test, assert_instr(xperm8))]
#[inline]
pub fn xperm8_64(rs1: u64, rs2: u64) -> u64 {
    unsafe { _xperm8_64(rs1, rs2) }
}

/// Nibble-wise lookup of indices into a vector.
///
/// The xperm4 instruction operates on nibbles. The rs1 register contains a vector of 16
/// 4-bit elements. The rs2 register contains a vector of 16 4-bit indexes. The result is
/// each element in rs2 replaced by the indexed element in rs1, or zero if the index into rs2
/// is out of bounds.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.48
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbkx")]
#[cfg_attr(test, assert_instr(xperm4))]
#[inline]
pub fn xperm4_64(rs1: u64, rs2: u64) -> u64 {
    unsafe { _xperm4_64(rs1, rs2) }
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
pub fn orc_b_64(rs: u64) -> u64 {
    unsafe { _orc_b_64(rs) }
}

/// Carry-less multiply (low-part)
///
/// clmul produces the lower half of the 128-bit carry-less product.
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
pub fn clmul_64(rs1: u64, rs2: u64) -> u64 {
    unsafe { _clmul_64(rs1, rs2) }
}

/// Carry-less multiply (high-part)
///
/// clmulh produces the upper half of the 128-bit carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.12
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbkc")]
#[cfg_attr(test, assert_instr(clmulh))]
#[inline]
pub fn clmulh_64(rs1: u64, rs2: u64) -> u64 {
    unsafe { _clmulh_64(rs1, rs2) }
}

/// Carry-less multiply (reversed)
///
/// clmulr produces bits 126:63 of the 128-bit carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.13
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
#[target_feature(enable = "zbc")]
#[cfg_attr(test, assert_instr(clmulr))]
#[inline]
pub fn clmulr_64(rs1: u64, rs2: u64) -> u64 {
    unsafe { _clmulr_64(rs1, rs2) }
}
