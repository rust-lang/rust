use std::fmt;

use rustc_macros::{Decodable, Encodable, HashStable_Generic};
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
    C {
        unwind: bool,
    },
    Cdecl {
        unwind: bool,
    },
    Stdcall {
        unwind: bool,
    },
    Fastcall {
        unwind: bool,
    },
    Vectorcall {
        unwind: bool,
    },
    Thiscall {
        unwind: bool,
    },
    Aapcs {
        unwind: bool,
    },
    Win64 {
        unwind: bool,
    },
    SysV64 {
        unwind: bool,
    },
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    EfiApi,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    CCmseNonSecureCall,
    CCmseNonSecureEntry,
    System {
        unwind: bool,
    },
    RustIntrinsic,
    RustCall,
    /// *Not* a stable ABI, just directly use the Rust types to describe the ABI for LLVM. Even
    /// normally ABI-compatible Rust types can become ABI-incompatible with this ABI!
    Unadjusted,
    /// For things unlikely to be called, where reducing register pressure in
    /// `extern "Rust"` callers is worth paying extra cost in the callee.
    /// Stronger than just `#[cold]` because `fn` pointers might be incompatible.
    RustCold,
    RiscvInterruptM,
    RiscvInterruptS,
}

impl Abi {
    pub fn supports_varargs(self) -> bool {
        // * C and Cdecl obviously support varargs.
        // * C can be based on Aapcs, SysV64 or Win64, so they must support varargs.
        // * EfiApi is based on Win64 or C, so it also supports it.
        // * System falls back to C for functions with varargs.
        //
        // * Stdcall does not, because it would be impossible for the callee to clean
        //   up the arguments. (callee doesn't know how many arguments are there)
        // * Same for Fastcall, Vectorcall and Thiscall.
        // * Other calling conventions are related to hardware or the compiler itself.
        match self {
            Self::C { .. }
            | Self::Cdecl { .. }
            | Self::System { .. }
            | Self::Aapcs { .. }
            | Self::Win64 { .. }
            | Self::SysV64 { .. }
            | Self::EfiApi => true,
            _ => false,
        }
    }
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
    AbiData { abi: Abi::EfiApi, name: "efiapi" },
    AbiData { abi: Abi::AvrInterrupt, name: "avr-interrupt" },
    AbiData { abi: Abi::AvrNonBlockingInterrupt, name: "avr-non-blocking-interrupt" },
    AbiData { abi: Abi::CCmseNonSecureCall, name: "C-cmse-nonsecure-call" },
    AbiData { abi: Abi::CCmseNonSecureEntry, name: "C-cmse-nonsecure-entry" },
    AbiData { abi: Abi::System { unwind: false }, name: "system" },
    AbiData { abi: Abi::System { unwind: true }, name: "system-unwind" },
    AbiData { abi: Abi::RustIntrinsic, name: "rust-intrinsic" },
    AbiData { abi: Abi::RustCall, name: "rust-call" },
    AbiData { abi: Abi::Unadjusted, name: "unadjusted" },
    AbiData { abi: Abi::RustCold, name: "rust-cold" },
    AbiData { abi: Abi::RiscvInterruptM, name: "riscv-interrupt-m" },
    AbiData { abi: Abi::RiscvInterruptS, name: "riscv-interrupt-s" },
];

#[derive(Copy, Clone, Debug)]
pub enum AbiUnsupported {
    Unrecognized,
    Reason { explain: &'static str },
}

/// Returns the ABI with the given name (if any).
pub fn lookup(name: &str) -> Result<Abi, AbiUnsupported> {
    AbiDatas.iter().find(|abi_data| name == abi_data.name).map(|&x| x.abi).ok_or_else(|| match name {
        "riscv-interrupt" => AbiUnsupported::Reason {
            explain: "please use one of riscv-interrupt-m or riscv-interrupt-s for machine- or supervisor-level interrupts, respectively",
        },
        "riscv-interrupt-u" => AbiUnsupported::Reason {
            explain: "user-mode interrupt handlers have been removed from LLVM pending standardization, see: https://reviews.llvm.org/D149314",
        },
        "wasm" => AbiUnsupported::Reason {
            explain: "non-standard wasm ABI is no longer supported",
        },

        _ => AbiUnsupported::Unrecognized,

    })
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

pub fn is_enabled(
    features: &rustc_feature::Features,
    span: Span,
    name: &str,
) -> Result<(), AbiDisabled> {
    let s = is_stable(name);
    if let Err(AbiDisabled::Unstable { feature, .. }) = s {
        if features.active(feature) || span.allows_unstable(feature) {
            return Ok(());
        }
    }
    s
}

pub fn is_stable(name: &str) -> Result<(), AbiDisabled> {
    match name {
        // Stable
        "Rust" | "C" | "C-unwind" | "cdecl" | "cdecl-unwind" | "stdcall" | "stdcall-unwind"
        | "fastcall" | "fastcall-unwind" | "aapcs" | "aapcs-unwind" | "win64" | "win64-unwind"
        | "sysv64" | "sysv64-unwind" | "system" | "system-unwind" | "efiapi" | "thiscall"
        | "thiscall-unwind" => Ok(()),
        "rust-intrinsic" => Err(AbiDisabled::Unstable {
            feature: sym::intrinsics,
            explain: "intrinsics are subject to change",
        }),
        "vectorcall" => Err(AbiDisabled::Unstable {
            feature: sym::abi_vectorcall,
            explain: "vectorcall is experimental and subject to change",
        }),
        "vectorcall-unwind" => Err(AbiDisabled::Unstable {
            feature: sym::abi_vectorcall,
            explain: "vectorcall-unwind ABI is experimental and subject to change",
        }),
        "rust-call" => Err(AbiDisabled::Unstable {
            feature: sym::unboxed_closures,
            explain: "rust-call ABI is subject to change",
        }),
        "rust-cold" => Err(AbiDisabled::Unstable {
            feature: sym::rust_cold_cc,
            explain: "rust-cold is experimental and subject to change",
        }),
        "ptx-kernel" => Err(AbiDisabled::Unstable {
            feature: sym::abi_ptx,
            explain: "PTX ABIs are experimental and subject to change",
        }),
        "unadjusted" => Err(AbiDisabled::Unstable {
            feature: sym::abi_unadjusted,
            explain: "unadjusted ABI is an implementation detail and perma-unstable",
        }),
        "msp430-interrupt" => Err(AbiDisabled::Unstable {
            feature: sym::abi_msp430_interrupt,
            explain: "msp430-interrupt ABI is experimental and subject to change",
        }),
        "x86-interrupt" => Err(AbiDisabled::Unstable {
            feature: sym::abi_x86_interrupt,
            explain: "x86-interrupt ABI is experimental and subject to change",
        }),
        "avr-interrupt" | "avr-non-blocking-interrupt" => Err(AbiDisabled::Unstable {
            feature: sym::abi_avr_interrupt,
            explain: "avr-interrupt and avr-non-blocking-interrupt ABIs are experimental and subject to change",
        }),
        "riscv-interrupt-m" | "riscv-interrupt-s" => Err(AbiDisabled::Unstable {
            feature: sym::abi_riscv_interrupt,
            explain: "riscv-interrupt ABIs are experimental and subject to change",
        }),
        "C-cmse-nonsecure-call" => Err(AbiDisabled::Unstable {
            feature: sym::abi_c_cmse_nonsecure_call,
            explain: "C-cmse-nonsecure-call ABI is experimental and subject to change",
        }),
        "C-cmse-nonsecure-entry" => Err(AbiDisabled::Unstable {
            feature: sym::cmse_nonsecure_entry,
            explain: "C-cmse-nonsecure-entry ABI is experimental and subject to change",
        }),
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
            EfiApi => 22,
            AvrInterrupt => 23,
            AvrNonBlockingInterrupt => 24,
            CCmseNonSecureCall => 25,
            CCmseNonSecureEntry => 26,
            // Cross-platform ABIs
            System { unwind: false } => 27,
            System { unwind: true } => 28,
            RustIntrinsic => 29,
            RustCall => 30,
            Unadjusted => 31,
            RustCold => 32,
            RiscvInterruptM => 33,
            RiscvInterruptS => 34,
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
        write!(f, "\"{}\"", self.name())
    }
}
