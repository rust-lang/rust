#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.riscv.xperm8.i32"]
    fn _xperm8_32(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.xperm4.i32"]
    fn _xperm4_32(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.clmulh.i32"]
    fn _clmulh_32(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.clmulr.i32"]
    fn _clmulr_32(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.zip.i32"]
    fn _zip_32(rs1: i32) -> i32;

    #[link_name = "llvm.riscv.unzip.i32"]
    fn _unzip_32(rs1: i32) -> i32;
}

/// Byte-wise lookup of indices into a vector in registers.
///
/// The xperm8 instruction operates on bytes. The rs1 register contains a vector of 4
/// 8-bit elements. The rs2 register contains a vector of 4 8-bit indexes. The result is
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
pub fn xperm8_32(rs1: u32, rs2: u32) -> u32 {
    unsafe { _xperm8_32(rs1 as i32, rs2 as i32) as u32 }
}

/// Nibble-wise lookup of indices into a vector.
///
/// The xperm4 instruction operates on nibbles. The rs1 register contains a vector of 8
/// 4-bit elements. The rs2 register contains a vector of 8 4-bit indexes. The result is
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
pub fn xperm4_32(rs1: u32, rs2: u32) -> u32 {
    unsafe { _xperm4_32(rs1 as i32, rs2 as i32) as u32 }
}

/// Carry-less multiply (high-part)
///
/// clmulh produces the upper half of the 64-bit carry-less product.
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
pub fn clmulh_32(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmulh_32(rs1 as i32, rs2 as i32) as u32 }
}

/// Carry-less multiply (reversed)
///
/// clmulr produces bits 62:31 of the 64-bit carry-less product.
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
pub fn clmulr_32(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmulr_32(rs1 as i32, rs2 as i32) as u32 }
}

/// Place upper/lower halves of the source register into odd/even bits of the destination
/// respectivley.
///
/// This instruction places bits in the low half of the source register into the even bit
/// positions of the destination, and bits in the high half of the source register into the odd
/// bit positions of the destination. It is the inverse of the unzip instruction. This
/// instruction is available only on RV32.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.49
#[target_feature(enable = "zbkb")]
// See #1464
// #[cfg_attr(test, assert_instr(zip))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn zip_32(rs: u32) -> u32 {
    unsafe { _zip_32(rs as i32) as u32 }
}

/// Place odd and even bits of the source word into upper/lower halves of the destination.
///
/// This instruction places the even bits of the source register into the low half of the
/// destination, and the odd bits of the source into the high bits of the destination. It is
/// the inverse of the zip instruction. This instruction is available only on RV32.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.45
#[target_feature(enable = "zbkb")]
#[cfg_attr(test, assert_instr(unzip))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn unzip_32(rs: u32) -> u32 {
    unsafe { _unzip_32(rs as i32) as u32 }
}
