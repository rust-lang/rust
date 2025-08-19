#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "unadjusted" {
    #[link_name = "llvm.riscv.aes32esi"]
    fn _aes32esi(rs1: i32, rs2: i32, bs: i32) -> i32;

    #[link_name = "llvm.riscv.aes32esmi"]
    fn _aes32esmi(rs1: i32, rs2: i32, bs: i32) -> i32;

    #[link_name = "llvm.riscv.aes32dsi"]
    fn _aes32dsi(rs1: i32, rs2: i32, bs: i32) -> i32;

    #[link_name = "llvm.riscv.aes32dsmi"]
    fn _aes32dsmi(rs1: i32, rs2: i32, bs: i32) -> i32;

    #[link_name = "llvm.riscv.zip.i32"]
    fn _zip(rs1: i32) -> i32;

    #[link_name = "llvm.riscv.unzip.i32"]
    fn _unzip(rs1: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sig0h"]
    fn _sha512sig0h(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sig0l"]
    fn _sha512sig0l(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sig1h"]
    fn _sha512sig1h(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sig1l"]
    fn _sha512sig1l(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sum0r"]
    fn _sha512sum0r(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.sha512sum1r"]
    fn _sha512sum1r(rs1: i32, rs2: i32) -> i32;
}

/// AES final round encryption instruction for RV32.
///
/// This instruction sources a single byte from rs2 according to bs. To this it applies the
/// forward AES SBox operation, before XOR’ing the result with rs1. This instruction must
/// always be implemented such that its execution latency does not depend on the data being
/// operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.3
///
/// # Note
///
/// The `BS` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
#[target_feature(enable = "zkne")]
#[rustc_legacy_const_generics(2)]
// See #1464
// #[cfg_attr(test, assert_instr(aes32esi, BS = 0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes32esi<const BS: u8>(rs1: u32, rs2: u32) -> u32 {
    static_assert!(BS < 4);

    unsafe { _aes32esi(rs1 as i32, rs2 as i32, BS as i32) as u32 }
}

/// AES middle round encryption instruction for RV32 with.
///
/// This instruction sources a single byte from rs2 according to bs. To this it applies the
/// forward AES SBox operation, and a partial forward MixColumn, before XOR’ing the result with
/// rs1. This instruction must always be implemented such that its execution latency does not
/// depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.4
///
/// # Note
///
/// The `bs` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
#[target_feature(enable = "zkne")]
#[rustc_legacy_const_generics(2)]
// See #1464
// #[cfg_attr(test, assert_instr(aes32esmi, BS = 0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes32esmi<const BS: u8>(rs1: u32, rs2: u32) -> u32 {
    static_assert!(BS < 4);

    unsafe { _aes32esmi(rs1 as i32, rs2 as i32, BS as i32) as u32 }
}

/// AES final round decryption instruction for RV32.
///
/// This instruction sources a single byte from rs2 according to bs. To this it applies the
/// inverse AES SBox operation, and XOR’s the result with rs1. This instruction must always be
/// implemented such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.1
///
/// # Note
///
/// The `BS` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
#[target_feature(enable = "zknd")]
#[rustc_legacy_const_generics(2)]
// See #1464
// #[cfg_attr(test, assert_instr(aes32dsi, BS = 0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes32dsi<const BS: u8>(rs1: u32, rs2: u32) -> u32 {
    static_assert!(BS < 4);

    unsafe { _aes32dsi(rs1 as i32, rs2 as i32, BS as i32) as u32 }
}

/// AES middle round decryption instruction for RV32.
///
/// This instruction sources a single byte from rs2 according to bs. To this it applies the
/// inverse AES SBox operation, and a partial inverse MixColumn, before XOR’ing the result with
/// rs1. This instruction must always be implemented such that its execution latency does not
/// depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.2
///
/// # Note
///
/// The `BS` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
#[target_feature(enable = "zknd")]
#[rustc_legacy_const_generics(2)]
// See #1464
// #[cfg_attr(test, assert_instr(aes32dsmi, BS = 0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes32dsmi<const BS: u8>(rs1: u32, rs2: u32) -> u32 {
    static_assert!(BS < 4);

    unsafe { _aes32dsmi(rs1 as i32, rs2 as i32, BS as i32) as u32 }
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
pub fn zip(rs: u32) -> u32 {
    unsafe { _zip(rs as i32) as u32 }
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
pub fn unzip(rs: u32) -> u32 {
    unsafe { _unzip(rs as i32) as u32 }
}

/// Implements the high half of the Sigma0 transformation, as used in the SHA2-512 hash
/// function \[49\] (Section 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sigma0 transform of the
/// SHA2-512 hash function in conjunction with the sha512sig0l instruction. The transform is a
/// 64-bit to 64-bit function, so the input and output are each represented by two 32-bit
/// registers. This instruction must always be implemented such that its execution latency does
/// not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.31
#[target_feature(enable = "zknh")]
// See #1464
// #[cfg_attr(test, assert_instr(sha512sig0h))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig0h(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sig0h(rs1 as i32, rs2 as i32) as u32 }
}

/// Implements the low half of the Sigma0 transformation, as used in the SHA2-512 hash function
/// \[49\] (Section 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sigma0 transform of the
/// SHA2-512 hash function in conjunction with the sha512sig0h instruction. The transform is a
/// 64-bit to 64-bit function, so the input and output are each represented by two 32-bit
/// registers. This instruction must always be implemented such that its execution latency does
/// not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.32
#[target_feature(enable = "zknh")]
// See #1464
// #[cfg_attr(test, assert_instr(sha512sig0l))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig0l(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sig0l(rs1 as i32, rs2 as i32) as u32 }
}

/// Implements the high half of the Sigma1 transformation, as used in the SHA2-512 hash
/// function \[49\] (Section 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sigma1 transform of the
/// SHA2-512 hash function in conjunction with the sha512sig1l instruction. The transform is a
/// 64-bit to 64-bit function, so the input and output are each represented by two 32-bit
/// registers. This instruction must always be implemented such that its execution latency does
/// not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.33
#[target_feature(enable = "zknh")]
// See #1464
// #[cfg_attr(test, assert_instr(sha512sig1h))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig1h(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sig1h(rs1 as i32, rs2 as i32) as u32 }
}

/// Implements the low half of the Sigma1 transformation, as used in the SHA2-512 hash function
/// \[49\] (Section 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sigma1 transform of the
/// SHA2-512 hash function in conjunction with the sha512sig1h instruction. The transform is a
/// 64-bit to 64-bit function, so the input and output are each represented by two 32-bit
/// registers. This instruction must always be implemented such that its execution latency does
/// not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.34
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig1l))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig1l(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sig1l(rs1 as i32, rs2 as i32) as u32 }
}

/// Implements the Sum0 transformation, as used in the SHA2-512 hash function \[49\] (Section
/// 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sum0 transform of the
/// SHA2-512 hash function. The transform is a 64-bit to 64-bit function, so the input and
/// output is represented by two 32-bit registers. This instruction must always be implemented
/// such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.35
#[target_feature(enable = "zknh")]
// See #1464
// #[cfg_attr(test, assert_instr(sha512sum0r))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sum0r(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sum0r(rs1 as i32, rs2 as i32) as u32 }
}

/// Implements the Sum1 transformation, as used in the SHA2-512 hash function \[49\] (Section
/// 4.1.3).
///
/// This instruction is implemented on RV32 only. Used to compute the Sum1 transform of the
/// SHA2-512 hash function. The transform is a 64-bit to 64-bit function, so the input and
/// output is represented by two 32-bit registers. This instruction must always be implemented
/// such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.36
#[target_feature(enable = "zknh")]
// See #1464
// #[cfg_attr(test, assert_instr(sha512sum1r))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sum1r(rs1: u32, rs2: u32) -> u32 {
    unsafe { _sha512sum1r(rs1 as i32, rs2 as i32) as u32 }
}
