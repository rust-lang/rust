#[allow(unused)]
use core::arch::asm;

#[allow(unused)]
macro_rules! constify_imm_0_until_10 {
    ($imm2:expr, $expand:ident) => {
        match $imm2 {
            1 => $expand!(1),
            2 => $expand!(2),
            3 => $expand!(3),
            4 => $expand!(4),
            5 => $expand!(5),
            6 => $expand!(6),
            7 => $expand!(7),
            8 => $expand!(8),
            9 => $expand!(9),
            10 => $expand!(10),
            _ => $expand!(0),
        }
    };
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
///
/// # Safety
///
/// This function is safe to use if the `zkne` target feature is present.
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes64es))]
#[inline]
pub unsafe fn aes64es(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "aes64es {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zkne` target feature is present.
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes64esm))]
#[inline]
pub unsafe fn aes64esm(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "aes64esm {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknd` target feature is present.
#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64ds))]
#[inline]
pub unsafe fn aes64ds(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "aes64ds {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknd` target feature is present.
#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64dsm))]
#[inline]
pub unsafe fn aes64dsm(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "aes64dsm {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
/// The `rnum` parameter is expected to be a constant value inside the range of `0..=10`, if a
/// value outside the valid range is given it uses `rnum=0`.
///
/// # Safety
///
/// This function is safe to use if the `zkne` or `zknd` target feature is present.
#[target_feature(enable = "zkne", enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64ks1i))]
#[inline]
pub unsafe fn aes64ks1i(rs1: u64, rnum: u8) -> u64 {
    macro_rules! aes64ks1i {
            ($imm_0_until_10:expr) => {{
                let value: u64;
                unsafe {
                    asm!(
                        concat!("aes64ks1i {rd},{rs1},", $imm_0_until_10),
                        rd = lateout(reg) value,
                        rs1 = in(reg) rs1,
                        options(pure, nomem, nostack),
                    )
                }
                value
            }}
        }
    constify_imm_0_until_10!(rnum, aes64ks1i)
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
///
/// # Safety
///
/// This function is safe to use if the `zkne` or `zknd` target feature is present.
#[target_feature(enable = "zkne", enable = "zknd")]
#[cfg_attr(test, assert_instr(aes64ks2))]
#[inline]
pub unsafe fn aes64ks2(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "aes64ks2 {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
}

/// Pack the low 16-bits of rs1 and rs2 into rd on RV64
///
/// This instruction packs the low 16 bits of rs1 and rs2 into the 32 least-significant bits of
/// rd, sign extending the 32-bit result to the rest of rd. This instruction only exists on
/// RV64 based systems.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.26
///
/// # Safety
///
/// This function is safe to use if the `zbkb` target feature is present.
#[target_feature(enable = "zbkb")]
#[cfg_attr(test, assert_instr(packw))]
#[inline]
pub unsafe fn packw(rs1: u64, rs2: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "packw {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig0))]
#[inline]
pub unsafe fn sha512sig0(rs1: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "sha512sig0 {rd},{rs1}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig1))]
#[inline]
pub unsafe fn sha512sig1(rs1: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "sha512sig1 {rd},{rs1}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum0))]
#[inline]
pub unsafe fn sha512sum0(rs1: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "sha512sum0 {rd},{rs1}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum1))]
#[inline]
pub unsafe fn sha512sum1(rs1: u64) -> u64 {
    let value: u64;
    unsafe {
        asm!(
            "sha512sum1 {rd},{rs1}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            options(pure, nomem, nostack),
        )
    }
    value
}
