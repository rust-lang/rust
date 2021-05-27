use std::fmt;

use rustc_macros::HashStable_Generic;

#[cfg(test)]
mod tests;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum Abi {
    // Multiplatform / generic ABIs
    //
    // These ABIs come first because every time we add a new ABI, we
    // have to re-bless all the hashing tests. These are used in many
    // places, so giving them stable values reduces test churn. The
    // specific values are meaningless.
    Rust,
    C { unwind: bool },

    // Single platform ABIs
    Cdecl,
    Stdcall { unwind: bool },
    Fastcall,
    Vectorcall,
    Thiscall { unwind: bool },
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
    Wasm,

    // Multiplatform / generic ABIs
    System { unwind: bool },
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
    AbiData { abi: Abi::C { unwind: false }, name: "C", generic: true },
    AbiData { abi: Abi::C { unwind: true }, name: "C-unwind", generic: true },
    // Platform-specific ABIs
    AbiData { abi: Abi::Cdecl, name: "cdecl", generic: false },
    AbiData { abi: Abi::Stdcall { unwind: false }, name: "stdcall", generic: false },
    AbiData { abi: Abi::Stdcall { unwind: true }, name: "stdcall-unwind", generic: false },
    AbiData { abi: Abi::Fastcall, name: "fastcall", generic: false },
    AbiData { abi: Abi::Vectorcall, name: "vectorcall", generic: false },
    AbiData { abi: Abi::Thiscall { unwind: false }, name: "thiscall", generic: false },
    AbiData { abi: Abi::Thiscall { unwind: true }, name: "thiscall-unwind", generic: false },
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
    AbiData { abi: Abi::Wasm, name: "wasm", generic: false },
    // Cross-platform ABIs
    AbiData { abi: Abi::System { unwind: false }, name: "system", generic: true },
    AbiData { abi: Abi::System { unwind: true }, name: "system-unwind", generic: true },
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
        // N.B., this ordering MUST match the AbiDatas array above.
        // (This is ensured by the test indices_are_correct().)
        use Abi::*;
        let i = match self {
            // Cross-platform ABIs
            Rust => 0,
            C { unwind: false } => 1,
            C { unwind: true } => 2,
            // Platform-specific ABIs
            Cdecl => 3,
            Stdcall { unwind: false } => 4,
            Stdcall { unwind: true } => 5,
            Fastcall => 6,
            Vectorcall => 7,
            Thiscall { unwind: false } => 8,
            Thiscall { unwind: true } => 9,
            Aapcs => 10,
            Win64 => 11,
            SysV64 => 12,
            PtxKernel => 13,
            Msp430Interrupt => 14,
            X86Interrupt => 15,
            AmdGpuKernel => 16,
            EfiApi => 17,
            AvrInterrupt => 18,
            AvrNonBlockingInterrupt => 19,
            CCmseNonSecureCall => 20,
            Wasm => 21,
            // Cross-platform ABIs
            System { unwind: false } => 22,
            System { unwind: true } => 23,
            RustIntrinsic => 24,
            RustCall => 25,
            PlatformIntrinsic => 26,
            Unadjusted => 27,
        };
        debug_assert!(
            AbiDatas
                .iter()
                .enumerate()
                .find(|(_, AbiData { abi, .. })| *abi == self)
                .map(|(index, _)| index)
                .expect("abi variant has associated data")
                == i,
            "Abi index did not match `AbiDatas` ordering"
        );
        i
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
        match self {
            abi => write!(f, "\"{}\"", abi.name()),
        }
    }
}
