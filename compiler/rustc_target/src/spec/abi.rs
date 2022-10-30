use std::fmt;

use rustc_macros::HashStable_Generic;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

#[cfg(test)]
mod tests;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum Abi {
    // Some of the ABIs come first because every time we add a new ABI, we have to re-bless all the
    // hashing tests. These are used in many places, so giving them stable values reduces test
    // churn. The specific values are meaningless.
    Rust,
    C { unwind: bool },
    Cdecl { unwind: bool },
    Stdcall { unwind: bool },
    Fastcall { unwind: bool },
    Vectorcall { unwind: bool },
    Thiscall { unwind: bool },
    Aapcs { unwind: bool },
    Win64 { unwind: bool },
    SysV64 { unwind: bool },
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,
    EfiApi,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    CCmseNonSecureCall,
    Wasm,
    System { unwind: bool },
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
    Unadjusted,
    RustCold,
}

#[derive(Copy, Clone)]
pub struct AbiData {
    abi: Abi,

    /// Name of this ABI as we like it called.
    name: &'static str,
}

#[allow(non_upper_case_globals)]
const AbiDatas: &[AbiData] = &[
    AbiData { abi: Abi::Rust, name: "Rust" },
    AbiData { abi: Abi::C { unwind: false }, name: "C" },
    AbiData { abi: Abi::C { unwind: true }, name: "C-unwind" },
    AbiData { abi: Abi::Cdecl { unwind: false }, name: "cdecl" },
    AbiData { abi: Abi::Cdecl { unwind: true }, name: "cdecl-unwind" },
    AbiData { abi: Abi::Stdcall { unwind: false }, name: "stdcall" },
    AbiData { abi: Abi::Stdcall { unwind: true }, name: "stdcall-unwind" },
    AbiData { abi: Abi::Fastcall { unwind: false }, name: "fastcall" },
    AbiData { abi: Abi::Fastcall { unwind: true }, name: "fastcall-unwind" },
    AbiData { abi: Abi::Vectorcall { unwind: false }, name: "vectorcall" },
    AbiData { abi: Abi::Vectorcall { unwind: true }, name: "vectorcall-unwind" },
    AbiData { abi: Abi::Thiscall { unwind: false }, name: "thiscall" },
    AbiData { abi: Abi::Thiscall { unwind: true }, name: "thiscall-unwind" },
    AbiData { abi: Abi::Aapcs { unwind: false }, name: "aapcs" },
    AbiData { abi: Abi::Aapcs { unwind: true }, name: "aapcs-unwind" },
    AbiData { abi: Abi::Win64 { unwind: false }, name: "win64" },
    AbiData { abi: Abi::Win64 { unwind: true }, name: "win64-unwind" },
    AbiData { abi: Abi::SysV64 { unwind: false }, name: "sysv64" },
    AbiData { abi: Abi::SysV64 { unwind: true }, name: "sysv64-unwind" },
    AbiData { abi: Abi::PtxKernel, name: "ptx-kernel" },
    AbiData { abi: Abi::Msp430Interrupt, name: "msp430-interrupt" },
    AbiData { abi: Abi::X86Interrupt, name: "x86-interrupt" },
    AbiData { abi: Abi::AmdGpuKernel, name: "amdgpu-kernel" },
    AbiData { abi: Abi::EfiApi, name: "efiapi" },
    AbiData { abi: Abi::AvrInterrupt, name: "avr-interrupt" },
    AbiData { abi: Abi::AvrNonBlockingInterrupt, name: "avr-non-blocking-interrupt" },
    AbiData { abi: Abi::CCmseNonSecureCall, name: "C-cmse-nonsecure-call" },
    AbiData { abi: Abi::Wasm, name: "wasm" },
    AbiData { abi: Abi::System { unwind: false }, name: "system" },
    AbiData { abi: Abi::System { unwind: true }, name: "system-unwind" },
    AbiData { abi: Abi::RustIntrinsic, name: "rust-intrinsic" },
    AbiData { abi: Abi::RustCall, name: "rust-call" },
    AbiData { abi: Abi::PlatformIntrinsic, name: "platform-intrinsic" },
    AbiData { abi: Abi::Unadjusted, name: "unadjusted" },
    AbiData { abi: Abi::RustCold, name: "rust-cold" },
];

/// Returns the ABI with the given name (if any).
pub fn lookup(name: &str) -> Option<Abi> {
    AbiDatas.iter().find(|abi_data| name == abi_data.name).map(|&x| x.abi)
}

pub fn all_names() -> Vec<&'static str> {
    AbiDatas.iter().map(|d| d.name).collect()
}

pub fn enabled_names(features: &rustc_feature::Features, span: Span) -> Vec<&'static str> {
    AbiDatas
        .iter()
        .map(|d| d.name)
        .filter(|name| is_enabled(features, span, name).is_ok())
        .collect()
}

pub enum AbiDisabled {
    Unstable { feature: Symbol, explain: &'static str },
    Unrecognized,
}

fn gate_feature_post(
    features: &rustc_feature::Features,
    feature: Symbol,
    span: Span,
    explain: &'static str,
) -> Result<(), AbiDisabled> {
    if !features.enabled(feature) && !span.allows_unstable(feature) {
        Err(AbiDisabled::Unstable { feature, explain })
    } else {
        Ok(())
    }
}

pub fn is_enabled(
    features: &rustc_feature::Features,
    span: Span,
    name: &str,
) -> Result<(), AbiDisabled> {
    match name {
        // Stable
        "Rust" | "C" | "cdecl" | "stdcall" | "fastcall" | "aapcs" | "win64" | "sysv64"
        | "system" => Ok(()),
        "rust-intrinsic" => {
            gate_feature_post(features, sym::intrinsics, span, "intrinsics are subject to change")
        }
        "platform-intrinsic" => gate_feature_post(
            features,
            sym::platform_intrinsics,
            span,
            "platform intrinsics are experimental and possibly buggy",
        ),
        "vectorcall" => gate_feature_post(
            features,
            sym::abi_vectorcall,
            span,
            "vectorcall is experimental and subject to change",
        ),
        "thiscall" => gate_feature_post(
            features,
            sym::abi_thiscall,
            span,
            "thiscall is experimental and subject to change",
        ),
        "rust-call" => gate_feature_post(
            features,
            sym::unboxed_closures,
            span,
            "rust-call ABI is subject to change",
        ),
        "rust-cold" => gate_feature_post(
            features,
            sym::rust_cold_cc,
            span,
            "rust-cold is experimental and subject to change",
        ),
        "ptx-kernel" => gate_feature_post(
            features,
            sym::abi_ptx,
            span,
            "PTX ABIs are experimental and subject to change",
        ),
        "unadjusted" => gate_feature_post(
            features,
            sym::abi_unadjusted,
            span,
            "unadjusted ABI is an implementation detail and perma-unstable",
        ),
        "msp430-interrupt" => gate_feature_post(
            features,
            sym::abi_msp430_interrupt,
            span,
            "msp430-interrupt ABI is experimental and subject to change",
        ),
        "x86-interrupt" => gate_feature_post(
            features,
            sym::abi_x86_interrupt,
            span,
            "x86-interrupt ABI is experimental and subject to change",
        ),
        "amdgpu-kernel" => gate_feature_post(
            features,
            sym::abi_amdgpu_kernel,
            span,
            "amdgpu-kernel ABI is experimental and subject to change",
        ),
        "avr-interrupt" | "avr-non-blocking-interrupt" => gate_feature_post(
            features,
            sym::abi_avr_interrupt,
            span,
            "avr-interrupt and avr-non-blocking-interrupt ABIs are experimental and subject to change",
        ),
        "efiapi" => gate_feature_post(
            features,
            sym::abi_efiapi,
            span,
            "efiapi ABI is experimental and subject to change",
        ),
        "C-cmse-nonsecure-call" => gate_feature_post(
            features,
            sym::abi_c_cmse_nonsecure_call,
            span,
            "C-cmse-nonsecure-call ABI is experimental and subject to change",
        ),
        "C-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "C-unwind ABI is experimental and subject to change",
        ),
        "stdcall-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "stdcall-unwind ABI is experimental and subject to change",
        ),
        "system-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "system-unwind ABI is experimental and subject to change",
        ),
        "thiscall-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "thiscall-unwind ABI is experimental and subject to change",
        ),
        "cdecl-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "cdecl-unwind ABI is experimental and subject to change",
        ),
        "fastcall-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "fastcall-unwind ABI is experimental and subject to change",
        ),
        "vectorcall-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "vectorcall-unwind ABI is experimental and subject to change",
        ),
        "aapcs-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "aapcs-unwind ABI is experimental and subject to change",
        ),
        "win64-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "win64-unwind ABI is experimental and subject to change",
        ),
        "sysv64-unwind" => gate_feature_post(
            features,
            sym::c_unwind,
            span,
            "sysv64-unwind ABI is experimental and subject to change",
        ),
        "wasm" => gate_feature_post(
            features,
            sym::wasm_abi,
            span,
            "wasm ABI is experimental and subject to change",
        ),
        _ => Err(AbiDisabled::Unrecognized),
    }
}

impl Abi {
    /// Default ABI chosen for `extern fn` declarations without an explicit ABI.
    pub const FALLBACK: Abi = Abi::C { unwind: false };

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
            Cdecl { unwind: false } => 3,
            Cdecl { unwind: true } => 4,
            Stdcall { unwind: false } => 5,
            Stdcall { unwind: true } => 6,
            Fastcall { unwind: false } => 7,
            Fastcall { unwind: true } => 8,
            Vectorcall { unwind: false } => 9,
            Vectorcall { unwind: true } => 10,
            Thiscall { unwind: false } => 11,
            Thiscall { unwind: true } => 12,
            Aapcs { unwind: false } => 13,
            Aapcs { unwind: true } => 14,
            Win64 { unwind: false } => 15,
            Win64 { unwind: true } => 16,
            SysV64 { unwind: false } => 17,
            SysV64 { unwind: true } => 18,
            PtxKernel => 19,
            Msp430Interrupt => 20,
            X86Interrupt => 21,
            AmdGpuKernel => 22,
            EfiApi => 23,
            AvrInterrupt => 24,
            AvrNonBlockingInterrupt => 25,
            CCmseNonSecureCall => 26,
            Wasm => 27,
            // Cross-platform ABIs
            System { unwind: false } => 28,
            System { unwind: true } => 29,
            RustIntrinsic => 30,
            RustCall => 31,
            PlatformIntrinsic => 32,
            Unadjusted => 33,
            RustCold => 34,
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
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            abi => write!(f, "\"{}\"", abi.name()),
        }
    }
}
