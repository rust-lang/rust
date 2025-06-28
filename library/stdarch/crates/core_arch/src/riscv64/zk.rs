#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "unadjusted" {
    #[link_name = "llvm.riscv.aes64es"]
    fn _aes64es(rs1: i64, rs2: i64) -> i64;

    #[link_name = "llvm.riscv.aes64esm"]
    fn _aes64esm(rs1: i64, rs2: i64) -> i64;

    #[link_name = "llvm.riscv.aes64ds"]
    fn _aes64ds(rs1: i64, rs2: i64) -> i64;

    #[link_name = "llvm.riscv.aes64dsm"]
    fn _aes64dsm(rs1: i64, rs2: i64) -> i64;

    #[link_name = "llvm.riscv.aes64ks1i"]
    fn _aes64ks1i(rs1: i64, rnum: i32) -> i64;

    #[link_name = "llvm.riscv.aes64ks2"]
    fn _aes64ks2(rs1: i64, rs2: i64) -> i64;

    #[link_name = "llvm.riscv.aes64im"]
    fn _aes64im(rs1: i64) -> i64;

    #[link_name = "llvm.riscv.sha512sig0"]
    fn _sha512sig0(rs1: i64) -> i64;

    #[link_name = "llvm.riscv.sha512sig1"]
    fn _sha512sig1(rs1: i64) -> i64;

    #[link_name = "llvm.riscv.sha512sum0"]
    fn _sha512sum0(rs1: i64) -> i64;

    #[link_name = "llvm.riscv.sha512sum1"]
    fn _sha512sum1(rs1: i64) -> i64;
}

/// AES final round encryption instruction for RV64.
///
/// Uses the two 64-bit source registers to represent the entire AES state, and produces half
/// of the next round output, applying the ShiftRows and SubBytes steps. This instruction must
/// always be implemented such that its execution latency does not depend on the data being
/// operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.7
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes64es))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64es(rs1: u64, rs2: u64) -> u64 {
    unsafe { _aes64es(rs1 as i64, rs2 as i64) as u64 }
}

/// AES middle round encryption instruction for RV64.
///
/// Uses the two 64-bit source registers to represent the entire AES state, and produces half
/// of the next round output, applying the ShiftRows, SubBytes and MixColumns steps. This
/// instruction must always be implemented such that its execution latency does not depend on
/// the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.8
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes64esm))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64esm(rs1: u64, rs2: u64) -> u64 {
    unsafe { _aes64esm(rs1 as i64, rs2 as i64) as u64 }
}

/// AES final round decryption instruction for RV64.
///
/// Uses the two 64-bit source registers to represent the entire AES state, and produces half
/// of the next round output, applying the Inverse ShiftRows and SubBytes steps. This
/// instruction must always be implemented such that its execution latency does not depend on
/// the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.5
#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64ds))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64ds(rs1: u64, rs2: u64) -> u64 {
    unsafe { _aes64ds(rs1 as i64, rs2 as i64) as u64 }
}

/// AES middle round decryption instruction for RV64.
///
/// Uses the two 64-bit source registers to represent the entire AES state, and produces half
/// of the next round output, applying the Inverse ShiftRows, SubBytes and MixColumns steps.
/// This instruction must always be implemented such that its execution latency does not depend
/// on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.6
#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64dsm))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64dsm(rs1: u64, rs2: u64) -> u64 {
    unsafe { _aes64dsm(rs1 as i64, rs2 as i64) as u64 }
}

/// This instruction implements part of the KeySchedule operation for the AES Block cipher
/// involving the SBox operation.
///
/// This instruction implements the rotation, SubBytes and Round Constant addition steps of the
/// AES block cipher Key Schedule. This instruction must always be implemented such that its
/// execution latency does not depend on the data being operated on. Note that rnum must be in
/// the range 0x0..0xA. The values 0xB..0xF are reserved.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.10
///
/// # Note
///
/// The `RNUM` parameter is expected to be a constant value inside the range of `0..=10`.
#[target_feature(enable = "zkne", enable = "zknd")]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(aes64ks1i, RNUM = 0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64ks1i<const RNUM: u8>(rs1: u64) -> u64 {
    static_assert!(RNUM <= 10);

    unsafe { _aes64ks1i(rs1 as i64, RNUM as i32) as u64 }
}

/// This instruction implements part of the KeySchedule operation for the AES Block cipher.
///
/// This instruction implements the additional XOR’ing of key words as part of the AES block
/// cipher Key Schedule. This instruction must always be implemented such that its execution
/// latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.11
#[target_feature(enable = "zkne", enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64ks2))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64ks2(rs1: u64, rs2: u64) -> u64 {
    unsafe { _aes64ks2(rs1 as i64, rs2 as i64) as u64 }
}

/// This instruction accelerates the inverse MixColumns step of the AES Block Cipher, and is used to aid creation of
/// the decryption KeySchedule.
///
/// The instruction applies the inverse MixColumns transformation to two columns of the state array, packed
/// into a single 64-bit register. It is used to create the inverse cipher KeySchedule, according to the equivalent
/// inverse cipher construction in (Page 23, Section 5.3.5). This instruction must always be implemented
/// such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.9
#[target_feature(enable = "zkne", enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64im))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn aes64im(rs1: u64) -> u64 {
    unsafe { _aes64im(rs1 as i64) as u64 }
}

/// Implements the Sigma0 transformation function as used in the SHA2-512 hash function \[49\]
/// (Section 4.1.3).
///
/// This instruction is supported for the RV64 base architecture. It implements the Sigma0
/// transform of the SHA2-512 hash function. \[49\]. This instruction must always be
/// implemented such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.37
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig0(rs1: u64) -> u64 {
    unsafe { _sha512sig0(rs1 as i64) as u64 }
}

/// Implements the Sigma1 transformation function as used in the SHA2-512 hash function \[49\]
/// (Section 4.1.3).
///
/// This instruction is supported for the RV64 base architecture. It implements the Sigma1
/// transform of the SHA2-512 hash function. \[49\]. This instruction must always be
/// implemented such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.38
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig1))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sig1(rs1: u64) -> u64 {
    unsafe { _sha512sig1(rs1 as i64) as u64 }
}

/// Implements the Sum0 transformation function as used in the SHA2-512 hash function \[49\]
/// (Section 4.1.3).
///
/// This instruction is supported for the RV64 base architecture. It implements the Sum0
/// transform of the SHA2-512 hash function. \[49\]. This instruction must always be
/// implemented such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.39
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum0))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sum0(rs1: u64) -> u64 {
    unsafe { _sha512sum0(rs1 as i64) as u64 }
}

/// Implements the Sum1 transformation function as used in the SHA2-512 hash function \[49\]
/// (Section 4.1.3).
///
/// This instruction is supported for the RV64 base architecture. It implements the Sum1
/// transform of the SHA2-512 hash function. \[49\]. This instruction must always be
/// implemented such that its execution latency does not depend on the data being operated on.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.40
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum1))]
#[inline]
#[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
pub fn sha512sum1(rs1: u64) -> u64 {
    unsafe { _sha512sum1(rs1 as i64) as u64 }
}
