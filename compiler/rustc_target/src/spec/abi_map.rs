use rustc_abi::{ArmCall, CanonAbi, ExternAbi, InterruptKind, X86Call};

use crate::spec::Target;

/// Mapping for ExternAbi to CanonAbi according to a Target
///
/// A maybe-transitional structure circa 2025 for hosting future experiments in
/// encapsulating arch-specific ABI lowering details to make them more testable.
#[derive(Clone, Debug)]
pub struct AbiMap {
    arch: Arch,
    os: OsKind,
}

/// result from trying to map an ABI
#[derive(Copy, Clone, Debug)]
pub enum AbiMapping {
    /// this ABI is exactly mapped for this platform
    Direct(CanonAbi),
    /// we don't yet warn on this, but we will
    Deprecated(CanonAbi),
    /// ABI we do not map for this platform: it must not reach codegen
    Invalid,
}

impl AbiMapping {
    /// optionally get a [CanonAbi], even if Deprecated
    pub fn into_option(self) -> Option<CanonAbi> {
        match self {
            Self::Direct(abi) | Self::Deprecated(abi) => Some(abi),
            Self::Invalid => None,
        }
    }

    /// get a [CanonAbi] even if Deprecated, panicking if Invalid
    #[track_caller]
    pub fn unwrap(self) -> CanonAbi {
        self.into_option().unwrap()
    }

    pub fn is_mapped(self) -> bool {
        self.into_option().is_some()
    }
}

impl AbiMap {
    /// create an AbiMap according to arbitrary fields on the [Target]
    pub fn from_target(target: &Target) -> Self {
        // the purpose of this little exercise is to force listing what affects these mappings
        let arch = match &*target.arch {
            "aarch64" => Arch::Aarch64,
            "amdgpu" => Arch::Amdgpu,
            "arm" if target.llvm_target.starts_with("thumbv8m") => Arch::Arm(ArmVer::ThumbV8M),
            "arm" => Arch::Arm(ArmVer::Other),
            "avr" => Arch::Avr,
            "msp430" => Arch::Msp430,
            "nvptx64" => Arch::Nvptx,
            "riscv32" | "riscv64" => Arch::Riscv,
            "x86" => Arch::X86,
            "x86_64" => Arch::X86_64,
            _ => Arch::Other,
        };

        let os = if target.is_like_windows {
            OsKind::Windows
        } else if target.is_like_vexos {
            OsKind::VEXos
        } else {
            OsKind::Other
        };

        AbiMap { arch, os }
    }

    /// lower an [ExternAbi] to a [CanonAbi] if this AbiMap allows
    pub fn canonize_abi(&self, extern_abi: ExternAbi, has_c_varargs: bool) -> AbiMapping {
        let AbiMap { os, arch } = *self;

        let canon_abi = match (extern_abi, arch) {
            // infallible lowerings
            (ExternAbi::C { .. }, _) => CanonAbi::C,
            (ExternAbi::Rust | ExternAbi::RustCall, _) => CanonAbi::Rust,
            (ExternAbi::Unadjusted, _) => CanonAbi::C,

            (ExternAbi::RustCold, _) if self.os == OsKind::Windows => CanonAbi::Rust,
            (ExternAbi::RustCold, _) => CanonAbi::RustCold,

            (ExternAbi::Custom, _) => CanonAbi::Custom,

            (ExternAbi::System { .. }, Arch::X86) if os == OsKind::Windows && !has_c_varargs => {
                CanonAbi::X86(X86Call::Stdcall)
            }
            (ExternAbi::System { .. }, Arch::Arm(..)) if self.os == OsKind::VEXos => {
                // Calls to VEXos APIs do not use VFP registers.
                CanonAbi::Arm(ArmCall::Aapcs)
            }
            (ExternAbi::System { .. }, _) => CanonAbi::C,

            // fallible lowerings
            /* multi-platform */
            // always and forever
            (ExternAbi::RustInvalid, _) => return AbiMapping::Invalid,

            (ExternAbi::EfiApi, Arch::Arm(..)) => CanonAbi::Arm(ArmCall::Aapcs),
            (ExternAbi::EfiApi, Arch::X86_64) => CanonAbi::X86(X86Call::Win64),
            (ExternAbi::EfiApi, Arch::Aarch64 | Arch::Riscv | Arch::X86) => CanonAbi::C,
            (ExternAbi::EfiApi, _) => return AbiMapping::Invalid,

            /* arm */
            (ExternAbi::Aapcs { .. }, Arch::Arm(..)) => CanonAbi::Arm(ArmCall::Aapcs),
            (ExternAbi::Aapcs { .. }, _) => return AbiMapping::Invalid,

            (ExternAbi::CmseNonSecureCall, Arch::Arm(ArmVer::ThumbV8M)) => {
                CanonAbi::Arm(ArmCall::CCmseNonSecureCall)
            }
            (ExternAbi::CmseNonSecureEntry, Arch::Arm(ArmVer::ThumbV8M)) => {
                CanonAbi::Arm(ArmCall::CCmseNonSecureEntry)
            }
            (ExternAbi::CmseNonSecureCall | ExternAbi::CmseNonSecureEntry, ..) => {
                return AbiMapping::Invalid;
            }

            /* gpu */
            (ExternAbi::PtxKernel, Arch::Nvptx) => CanonAbi::GpuKernel,
            (ExternAbi::GpuKernel, Arch::Amdgpu | Arch::Nvptx) => CanonAbi::GpuKernel,
            (ExternAbi::PtxKernel | ExternAbi::GpuKernel, _) => return AbiMapping::Invalid,

            /* x86 */
            (ExternAbi::Cdecl { .. }, Arch::X86) => CanonAbi::C,
            (ExternAbi::Cdecl { .. }, _) => return AbiMapping::Deprecated(CanonAbi::C),

            (ExternAbi::Fastcall { .. }, Arch::X86) => CanonAbi::X86(X86Call::Fastcall),
            (ExternAbi::Fastcall { .. }, _) if os == OsKind::Windows => {
                return AbiMapping::Deprecated(CanonAbi::C);
            }
            (ExternAbi::Fastcall { .. }, _) => return AbiMapping::Invalid,

            (ExternAbi::Stdcall { .. }, Arch::X86) => CanonAbi::X86(X86Call::Stdcall),
            (ExternAbi::Stdcall { .. }, _) if os == OsKind::Windows => {
                return AbiMapping::Deprecated(CanonAbi::C);
            }
            (ExternAbi::Stdcall { .. }, _) => return AbiMapping::Invalid,

            (ExternAbi::Thiscall { .. }, Arch::X86) => CanonAbi::X86(X86Call::Thiscall),
            (ExternAbi::Thiscall { .. }, _) => return AbiMapping::Invalid,

            (ExternAbi::Vectorcall { .. }, Arch::X86 | Arch::X86_64) => {
                CanonAbi::X86(X86Call::Vectorcall)
            }
            (ExternAbi::Vectorcall { .. }, _) => return AbiMapping::Invalid,

            (ExternAbi::SysV64 { .. }, Arch::X86_64) => CanonAbi::X86(X86Call::SysV64),
            (ExternAbi::Win64 { .. }, Arch::X86_64) => CanonAbi::X86(X86Call::Win64),
            (ExternAbi::SysV64 { .. } | ExternAbi::Win64 { .. }, _) => return AbiMapping::Invalid,

            /* interrupts */
            (ExternAbi::AvrInterrupt, Arch::Avr) => CanonAbi::Interrupt(InterruptKind::Avr),
            (ExternAbi::AvrNonBlockingInterrupt, Arch::Avr) => {
                CanonAbi::Interrupt(InterruptKind::AvrNonBlocking)
            }
            (ExternAbi::Msp430Interrupt, Arch::Msp430) => {
                CanonAbi::Interrupt(InterruptKind::Msp430)
            }
            (ExternAbi::RiscvInterruptM, Arch::Riscv) => {
                CanonAbi::Interrupt(InterruptKind::RiscvMachine)
            }
            (ExternAbi::RiscvInterruptS, Arch::Riscv) => {
                CanonAbi::Interrupt(InterruptKind::RiscvSupervisor)
            }
            (ExternAbi::X86Interrupt, Arch::X86 | Arch::X86_64) => {
                CanonAbi::Interrupt(InterruptKind::X86)
            }
            (
                ExternAbi::AvrInterrupt
                | ExternAbi::AvrNonBlockingInterrupt
                | ExternAbi::Msp430Interrupt
                | ExternAbi::RiscvInterruptM
                | ExternAbi::RiscvInterruptS
                | ExternAbi::X86Interrupt,
                _,
            ) => return AbiMapping::Invalid,
        };

        AbiMapping::Direct(canon_abi)
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Arch {
    Aarch64,
    Amdgpu,
    Arm(ArmVer),
    Avr,
    Msp430,
    Nvptx,
    Riscv,
    X86,
    X86_64,
    /// Architectures which don't need other considerations for ABI lowering
    Other,
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum OsKind {
    Windows,
    VEXos,
    Other,
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum ArmVer {
    ThumbV8M,
    Other,
}
