use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableOrd};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable};

use crate::AbiFromStrErr;

#[cfg(test)]
mod tests;

/// ABI we expect to see within `extern "{abi}"`
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable))]
pub enum ExternAbi {
    /* universal */
    /// presumed C ABI for the platform
    C {
        unwind: bool,
    },
    /// ABI of the "system" interface, e.g. the Win32 API, always "aliasing"
    System {
        unwind: bool,
    },

    /// that's us!
    Rust,
    /// the mostly-unused `unboxed_closures` ABI, effectively now an impl detail unless someone
    /// puts in the work to make it viable again... but would we need a special ABI?
    RustCall,
    /// For things unlikely to be called, where reducing register pressure in
    /// `extern "Rust"` callers is worth paying extra cost in the callee.
    /// Stronger than just `#[cold]` because `fn` pointers might be incompatible.
    RustCold,

    /// Unstable impl detail that directly uses Rust types to describe the ABI to LLVM.
    /// Even normally-compatible Rust types can become ABI-incompatible with this ABI!
    Unadjusted,

    /// An ABI that rustc does not know how to call or define. Functions with this ABI can
    /// only be created using `#[naked]` functions or `extern "custom"` blocks, and can only
    /// be called from inline assembly.
    Custom,

    /// UEFI ABI, usually an alias of C, but sometimes an arch-specific alias
    /// and only valid on platforms that have a UEFI standard
    EfiApi,

    /* arm */
    /// Arm Architecture Procedure Call Standard, sometimes `ExternAbi::C` is an alias for this
    Aapcs {
        unwind: bool,
    },
    /// extremely constrained barely-C ABI for TrustZone
    CCmseNonSecureCall,
    /// extremely constrained barely-C ABI for TrustZone
    CCmseNonSecureEntry,

    /* gpu */
    /// An entry-point function called by the GPU's host
    // FIXME: should not be callable from Rust on GPU targets, is for host's use only
    GpuKernel,
    /// An entry-point function called by the GPU's host
    // FIXME: why do we have two of these?
    PtxKernel,

    /* interrupt */
    AvrInterrupt,
    AvrNonBlockingInterrupt,
    Msp430Interrupt,
    RiscvInterruptM,
    RiscvInterruptS,
    X86Interrupt,

    /* x86 */
    /// `ExternAbi::C` but spelled funny because x86
    Cdecl {
        unwind: bool,
    },
    /// gnu-stdcall on "unix" and win-stdcall on "windows"
    Stdcall {
        unwind: bool,
    },
    /// gnu-fastcall on "unix" and win-fastcall on "windows"
    Fastcall {
        unwind: bool,
    },
    /// windows C++ ABI
    Thiscall {
        unwind: bool,
    },
    /// uses AVX and stuff
    Vectorcall {
        unwind: bool,
    },

    /* x86_64 */
    SysV64 {
        unwind: bool,
    },
    Win64 {
        unwind: bool,
    },
}

macro_rules! abi_impls {
    ($e_name:ident = {
        $($variant:ident $({ unwind: $uw:literal })? =><= $tok:literal,)*
    }) => {
        impl $e_name {
            pub const ALL_VARIANTS: &[Self] = &[
                $($e_name::$variant $({ unwind: $uw })*,)*
            ];
            pub const fn as_str(&self) -> &'static str {
                match self {
                    $($e_name::$variant $( { unwind: $uw } )* => $tok,)*
                }
            }
        }

        impl ::core::str::FromStr for $e_name {
            type Err = AbiFromStrErr;
            fn from_str(s: &str) -> Result<$e_name, Self::Err> {
                match s {
                    $($tok => Ok($e_name::$variant $({ unwind: $uw })*),)*
                    _ => Err(AbiFromStrErr::Unknown),
                }
            }
        }
    }
}

abi_impls! {
    ExternAbi = {
            C { unwind: false } =><= "C",
            CCmseNonSecureCall =><= "C-cmse-nonsecure-call",
            CCmseNonSecureEntry =><= "C-cmse-nonsecure-entry",
            C { unwind: true } =><= "C-unwind",
            Rust =><= "Rust",
            Aapcs { unwind: false } =><= "aapcs",
            Aapcs { unwind: true } =><= "aapcs-unwind",
            AvrInterrupt =><= "avr-interrupt",
            AvrNonBlockingInterrupt =><= "avr-non-blocking-interrupt",
            Cdecl { unwind: false } =><= "cdecl",
            Cdecl { unwind: true } =><= "cdecl-unwind",
            Custom =><= "custom",
            EfiApi =><= "efiapi",
            Fastcall { unwind: false } =><= "fastcall",
            Fastcall { unwind: true } =><= "fastcall-unwind",
            GpuKernel =><= "gpu-kernel",
            Msp430Interrupt =><= "msp430-interrupt",
            PtxKernel =><= "ptx-kernel",
            RiscvInterruptM =><= "riscv-interrupt-m",
            RiscvInterruptS =><= "riscv-interrupt-s",
            RustCall =><= "rust-call",
            RustCold =><= "rust-cold",
            Stdcall { unwind: false } =><= "stdcall",
            Stdcall { unwind: true } =><= "stdcall-unwind",
            System { unwind: false } =><= "system",
            System { unwind: true } =><= "system-unwind",
            SysV64 { unwind: false } =><= "sysv64",
            SysV64 { unwind: true } =><= "sysv64-unwind",
            Thiscall { unwind: false } =><= "thiscall",
            Thiscall { unwind: true } =><= "thiscall-unwind",
            Unadjusted =><= "unadjusted",
            Vectorcall { unwind: false } =><= "vectorcall",
            Vectorcall { unwind: true } =><= "vectorcall-unwind",
            Win64 { unwind: false } =><= "win64",
            Win64 { unwind: true } =><= "win64-unwind",
            X86Interrupt =><= "x86-interrupt",
    }
}

impl Ord for ExternAbi {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.as_str().cmp(rhs.as_str())
    }
}

impl PartialOrd for ExternAbi {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl PartialEq for ExternAbi {
    fn eq(&self, rhs: &Self) -> bool {
        self.cmp(rhs) == Ordering::Equal
    }
}

impl Eq for ExternAbi {}

impl Hash for ExternAbi {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
        // double-assurance of a prefix breaker
        u32::from_be_bytes(*b"ABI\0").hash(state);
    }
}

#[cfg(feature = "nightly")]
impl<C> HashStable<C> for ExternAbi {
    #[inline]
    fn hash_stable(&self, _: &mut C, hasher: &mut StableHasher) {
        Hash::hash(self, hasher);
    }
}

#[cfg(feature = "nightly")]
impl StableOrd for ExternAbi {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // because each ABI is hashed like a string, there is no possible instability
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
}

impl ExternAbi {
    /// An ABI "like Rust"
    ///
    /// These ABIs are fully controlled by the Rust compiler, which means they
    /// - support unwinding with `-Cpanic=unwind`, unlike `extern "C"`
    /// - often diverge from the C ABI
    /// - are subject to change between compiler versions
    pub fn is_rustic_abi(self) -> bool {
        use ExternAbi::*;
        matches!(self, Rust | RustCall | RustCold)
    }

    pub fn supports_varargs(self) -> bool {
        // * C and Cdecl obviously support varargs.
        // * C can be based on Aapcs, SysV64 or Win64, so they must support varargs.
        // * EfiApi is based on Win64 or C, so it also supports it.
        //
        // * Stdcall does not, because it would be impossible for the callee to clean
        //   up the arguments. (callee doesn't know how many arguments are there)
        // * Same for Fastcall, Vectorcall and Thiscall.
        // * Other calling conventions are related to hardware or the compiler itself.
        match self {
            Self::C { .. }
            | Self::Cdecl { .. }
            | Self::Aapcs { .. }
            | Self::Win64 { .. }
            | Self::SysV64 { .. }
            | Self::EfiApi => true,
            _ => false,
        }
    }
}

pub fn all_names() -> Vec<&'static str> {
    ExternAbi::ALL_VARIANTS.iter().map(|abi| abi.as_str()).collect()
}

impl ExternAbi {
    /// Default ABI chosen for `extern fn` declarations without an explicit ABI.
    pub const FALLBACK: ExternAbi = ExternAbi::C { unwind: false };

    pub fn name(self) -> &'static str {
        self.as_str()
    }
}

impl fmt::Display for ExternAbi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.as_str())
    }
}
