use std::fmt;

#[cfg(feature = "nightly")]
use rustc_macros::HashStable_Generic;

use crate::ExternAbi;

/// Calling convention to determine codegen
///
/// CanonAbi erases certain distinctions ExternAbi preserves, but remains target-dependent.
/// There are still both target-specific variants and aliasing variants, though much fewer.
/// The reason for this step is the frontend may wish to show an ExternAbi but implement that ABI
/// using a different ABI than the string per se, or describe irrelevant differences, e.g.
/// - extern "system"
/// - extern "cdecl"
/// - extern "C-unwind"
/// In that sense, this erases mere syntactic distinctions to create a canonical *directive*,
/// rather than picking the "actual" ABI.
#[derive(Copy, Clone, Debug)]
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum CanonAbi {
    // NOTE: the use of nested variants for some ABIs is for many targets they don't matter,
    // and this pushes the complexity of their reasoning to target-specific code,
    // allowing a `match` to easily exhaustively ignore these subcategories of variants.
    // Otherwise it is very tempting to avoid matching exhaustively!
    C,
    Rust,
    RustCold,

    /// ABIs relevant to 32-bit Arm targets
    Arm(ArmCall),
    /// ABI relevant to GPUs: the entry point for a GPU kernel
    GpuKernel,

    /// ABIs relevant to bare-metal interrupt targets
    // FIXME(workingjubilee): a particular reason for this nesting is we might not need these?
    // interrupt ABIs should have the same properties:
    // - uncallable by Rust calls, as LLVM rejects it in most cases
    // - uses a preserve-all-registers *callee* convention
    // - should always return `-> !` (effectively... it can't use normal `ret`)
    // what differs between targets is
    // - allowed arguments: x86 differs slightly, having 2-3 arguments which are handled magically
    // - may need special prologues/epilogues for some interrupts, without affecting "call ABI"
    Interrupt(InterruptKind),

    /// ABIs relevant to Windows or x86 targets
    X86(X86Call),
}

impl fmt::Display for CanonAbi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_erased_extern_abi().as_str().fmt(f)
    }
}

impl CanonAbi {
    /// convert to the ExternAbi that *shares a string* with this CanonAbi
    ///
    /// A target-insensitive mapping of CanonAbi to ExternAbi, convenient for "forwarding" impls.
    /// Importantly, the set of CanonAbi values is a logical *subset* of ExternAbi values,
    /// so this is injective: if you take an ExternAbi to a CanonAbi and back, you have lost data.
    const fn to_erased_extern_abi(self) -> ExternAbi {
        match self {
            CanonAbi::C => ExternAbi::C { unwind: false },
            CanonAbi::Rust => ExternAbi::Rust,
            CanonAbi::RustCold => ExternAbi::RustCold,
            CanonAbi::Arm(arm_call) => match arm_call {
                ArmCall::Aapcs => ExternAbi::Aapcs { unwind: false },
                ArmCall::CCmseNonSecureCall => ExternAbi::CCmseNonSecureCall,
                ArmCall::CCmseNonSecureEntry => ExternAbi::CCmseNonSecureEntry,
            },
            CanonAbi::GpuKernel => ExternAbi::GpuKernel,
            CanonAbi::Interrupt(interrupt_kind) => match interrupt_kind {
                InterruptKind::Avr => ExternAbi::AvrInterrupt,
                InterruptKind::AvrNonBlocking => ExternAbi::AvrNonBlockingInterrupt,
                InterruptKind::Msp430 => ExternAbi::Msp430Interrupt,
                InterruptKind::RiscvMachine => ExternAbi::RiscvInterruptM,
                InterruptKind::RiscvSupervisor => ExternAbi::RiscvInterruptS,
                InterruptKind::X86 => ExternAbi::X86Interrupt,
            },
            CanonAbi::X86(x86_call) => match x86_call {
                X86Call::Fastcall => ExternAbi::Fastcall { unwind: false },
                X86Call::Stdcall => ExternAbi::Stdcall { unwind: false },
                X86Call::SysV64 => ExternAbi::SysV64 { unwind: false },
                X86Call::Thiscall => ExternAbi::Thiscall { unwind: false },
                X86Call::Vectorcall => ExternAbi::Vectorcall { unwind: false },
                X86Call::Win64 => ExternAbi::Win64 { unwind: false },
            },
        }
    }
}

/// Callee codegen for interrupts
///
/// This is named differently from the "Call" enums because it is different:
/// these "ABI" differences are not relevant to callers, since there is "no caller".
/// These only affect callee codegen. making their categorization as distinct ABIs a bit peculiar.
#[derive(Copy, Clone, Debug)]
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum InterruptKind {
    Avr,
    AvrNonBlocking,
    Msp430,
    RiscvMachine,
    RiscvSupervisor,
    X86,
}

/// ABIs defined for x86-{32,64}
///
/// One of SysV64 or Win64 may alias the C ABI, and arguably Win64 is cross-platform now?
#[derive(Clone, Copy, Debug)]
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum X86Call {
    /// "fastcall" has both GNU and Windows variants
    Fastcall,
    /// "stdcall" has both GNU and Windows variants
    Stdcall,
    SysV64,
    Thiscall,
    Vectorcall,
    Win64,
}

/// ABIs defined for 32-bit Arm
#[derive(Copy, Clone, Debug)]
#[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
pub enum ArmCall {
    Aapcs,
    CCmseNonSecureCall,
    CCmseNonSecureEntry,
}
