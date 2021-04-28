use std::fmt;

use rustc_macros::HashStable_Generic;

#[cfg(test)]
mod tests;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum Abi {
    // N.B., this ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Multiplatform / generic ABIs
    //
    // These ABIs come first because every time we add a new ABI, we
    // have to re-bless all the hashing tests. These are used in many
    // places, so giving them stable values reduces test churn. The
    // specific values are meaningless.
    Rust = 0,
    C = 1,

    // Single platform ABIs
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Thiscall,
    Aapcs,
    Win64,
    SysV64,
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,
    EfiApi,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    CCmseNonSecureCall,

    // Multiplatform / generic ABIs
    System,
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
    Unadjusted,
}

#[derive(Copy, Clone)]
pub struct AbiData {
    abi: Abi,

    /// Name of this ABI as we like it called.
    name: &'static str,

    /// A generic ABI is supported on all platforms.
    generic: bool,
}

#[allow(non_upper_case_globals)]
const AbiDatas: &[AbiData] = &[
    // Cross-platform ABIs
    AbiData { abi: Abi::Rust, name: "Rust", generic: true },
    AbiData { abi: Abi::C, name: "C", generic: true },
    // Platform-specific ABIs
    AbiData { abi: Abi::Cdecl, name: "cdecl", generic: false },
    AbiData { abi: Abi::Stdcall, name: "stdcall", generic: false },
    AbiData { abi: Abi::Fastcall, name: "fastcall", generic: false },
    AbiData { abi: Abi::Vectorcall, name: "vectorcall", generic: false },
    AbiData { abi: Abi::Thiscall, name: "thiscall", generic: false },
    AbiData { abi: Abi::Aapcs, name: "aapcs", generic: false },
    AbiData { abi: Abi::Win64, name: "win64", generic: false },
    AbiData { abi: Abi::SysV64, name: "sysv64", generic: false },
    AbiData { abi: Abi::PtxKernel, name: "ptx-kernel", generic: false },
    AbiData { abi: Abi::Msp430Interrupt, name: "msp430-interrupt", generic: false },
    AbiData { abi: Abi::X86Interrupt, name: "x86-interrupt", generic: false },
    AbiData { abi: Abi::AmdGpuKernel, name: "amdgpu-kernel", generic: false },
    AbiData { abi: Abi::EfiApi, name: "efiapi", generic: false },
    AbiData { abi: Abi::AvrInterrupt, name: "avr-interrupt", generic: false },
    AbiData {
        abi: Abi::AvrNonBlockingInterrupt,
        name: "avr-non-blocking-interrupt",
        generic: false,
    },
    AbiData { abi: Abi::CCmseNonSecureCall, name: "C-cmse-nonsecure-call", generic: false },
    // Cross-platform ABIs
    AbiData { abi: Abi::System, name: "system", generic: true },
    AbiData { abi: Abi::RustIntrinsic, name: "rust-intrinsic", generic: true },
    AbiData { abi: Abi::RustCall, name: "rust-call", generic: true },
    AbiData { abi: Abi::PlatformIntrinsic, name: "platform-intrinsic", generic: true },
    AbiData { abi: Abi::Unadjusted, name: "unadjusted", generic: true },
];

/// Returns the ABI with the given name (if any).
pub fn lookup(name: &str) -> Option<Abi> {
    AbiDatas.iter().find(|abi_data| name == abi_data.name).map(|&x| x.abi)
}

pub fn all_names() -> Vec<&'static str> {
    AbiDatas.iter().map(|d| d.name).collect()
}

impl Abi {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn data(self) -> &'static AbiData {
        &AbiDatas[self.index()]
    }

    pub fn name(self) -> &'static str {
        self.data().name
    }

    pub fn generic(self) -> bool {
        self.data().generic
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}
