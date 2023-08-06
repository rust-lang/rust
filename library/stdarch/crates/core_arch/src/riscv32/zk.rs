#[allow(unused)]
use core::arch::asm;

#[allow(unused)]
macro_rules! constify_imm2 {
    ($imm2:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match $imm2 & 0b11 {
            0b00 => $expand!(0),
            0b01 => $expand!(1),
            0b10 => $expand!(2),
            _ => $expand!(3),
        }
    };
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
/// The `bs` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
///
/// # Safety
///
/// This function is safe to use if the `zkne` target feature is present.
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes32esi))]
#[inline]
pub unsafe fn aes32esi(rs1: u32, rs2: u32, bs: u8) -> u32 {
    macro_rules! aes32esi {
            ($imm2:expr) => {{
                let value: u32;
                unsafe {
                    asm!(
                        concat!("aes32esi {rd},{rs1},{rs2},", $imm2),
                        rd = lateout(reg) value,
                        rs1 = in(reg) rs1,
                        rs2 = in(reg) rs2,
                        options(pure, nomem, nostack),
                    );
                }
                value
            }}
        }
    constify_imm2!(bs, aes32esi)
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
///
/// # Safety
///
/// This function is safe to use if the `zkne` target feature is present.
#[target_feature(enable = "zkne")]
#[cfg_attr(test, assert_instr(aes32esmi))]
#[inline]
pub unsafe fn aes32esmi(rs1: u32, rs2: u32, bs: u8) -> u32 {
    macro_rules! aes32esmi {
            ($imm2:expr) => {{
                let value: u32;
                unsafe {
                    asm!(
                        concat!("aes32esmi {rd},{rs1},{rs2},", $imm2),
                        rd = lateout(reg) value,
                        rs1 = in(reg) rs1,
                        rs2 = in(reg) rs2,
                        options(pure, nomem, nostack),
                    );
                }
                value
            }}
        }
    constify_imm2!(bs, aes32esmi)
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
/// The `bs` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
///
/// # Safety
///
/// This function is safe to use if the `zknd` target feature is present.
#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes32dsi))]
#[inline]
pub unsafe fn aes32dsi(rs1: u32, rs2: u32, bs: u8) -> u32 {
    macro_rules! aes32dsi {
            ($imm2:expr) => {{
                let value: u32;
                unsafe {
                    asm!(
                        concat!("aes32dsi {rd},{rs1},{rs2},", $imm2),
                        rd = lateout(reg) value,
                        rs1 = in(reg) rs1,
                        rs2 = in(reg) rs2,
                        options(pure, nomem, nostack),
                    );
                }
                value
            }}
        }
    constify_imm2!(bs, aes32dsi)
}

#[target_feature(enable = "zknd")]
#[cfg_attr(test, assert_instr(aes32dsmi))]
#[inline]
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
/// The `bs` parameter is expected to be a constant value and only the bottom 2 bits of `bs` are
/// used.
///
/// # Safety
///
/// This function is safe to use if the `zknd` target feature is present.
pub unsafe fn aes32dsmi(rs1: u32, rs2: u32, bs: u8) -> u32 {
    macro_rules! aes32dsmi {
            ($imm2:expr) => {{
                let value: u32;
                unsafe {
                    asm!(
                        concat!("aes32dsmi {rd},{rs1},{rs2},", $imm2),
                        rd = lateout(reg) value,
                        rs1 = in(reg) rs1,
                        rs2 = in(reg) rs2,
                        options(pure, nomem, nostack),
                    );
                }
                value
            }}
        }
    constify_imm2!(bs, aes32dsmi)
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
///
/// # Safety
///
/// This function is safe to use if the `zbkb` target feature is present.
#[target_feature(enable = "zbkb")]
#[cfg_attr(test, assert_instr(zip))]
#[inline]
pub unsafe fn zip(rs: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(
            "zip {rd},{rs}",
            rd = lateout(reg) value,
            rs = in(reg) rs,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zbkb` target feature is present.
#[target_feature(enable = "zbkb")]
#[cfg_attr(test, assert_instr(unzip))]
#[inline]
pub unsafe fn unzip(rs: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(
            "unzip {rd},{rs}",
            rd = lateout(reg) value,
            rs = in(reg) rs,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig0h))]
#[inline]
pub unsafe fn sha512sig0h(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sig0h {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig0l))]
#[inline]
pub unsafe fn sha512sig0l(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sig0l {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig1h))]
#[inline]
pub unsafe fn sha512sig1h(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sig1h {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sig1l))]
#[inline]
pub unsafe fn sha512sig1l(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sig1l {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum0r))]
#[inline]
pub unsafe fn sha512sum0r(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sum0r {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
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
///
/// # Safety
///
/// This function is safe to use if the `zknh` target feature is present.
#[target_feature(enable = "zknh")]
#[cfg_attr(test, assert_instr(sha512sum1r))]
#[inline]
pub unsafe fn sha512sum1r(rs1: u32, rs2: u32) -> u32 {
    let value: u32;
    unsafe {
        asm!(
            "sha512sum1r {rd},{rs1},{rs2}",
            rd = lateout(reg) value,
            rs1 = in(reg) rs1,
            rs2 = in(reg) rs2,
            options(pure, nomem, nostack),
        )
    }
    value
}
