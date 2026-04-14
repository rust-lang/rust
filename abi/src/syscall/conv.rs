//! Syscall Calling Convention Logic
//!
//! This module defines the register mapping for syscall arguments on supported architectures.
//! It serves as the source of truth for both `stem` (userspace) and `bran` (kernel) to ensure
//! they agree on which register holds which argument.

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    /// Syscall Number
    pub const REG_N: &str = "rax";

    /// Argument 0
    pub const REG_A0: &str = "rdi";

    /// Argument 1
    pub const REG_A1: &str = "rsi";

    /// Argument 2
    pub const REG_A2: &str = "rdx";

    /// Argument 3
    /// Note: Linux/SystemV ABI uses RCX for 4th arg in C, but `syscall` instruction clobbers RCX.
    /// So the kernel ABI expects the 4th argument in R10.
    pub const REG_A3: &str = "r10";

    /// Argument 4
    pub const REG_A4: &str = "r8";

    /// Argument 5
    pub const REG_A5: &str = "r9";
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    pub const REG_N: &str = "x8";
    pub const REG_A0: &str = "x0";
    pub const REG_A1: &str = "x1";
    pub const REG_A2: &str = "x2";
    pub const REG_A3: &str = "x3";
    pub const REG_A4: &str = "x4";
    pub const REG_A5: &str = "x5";
}
