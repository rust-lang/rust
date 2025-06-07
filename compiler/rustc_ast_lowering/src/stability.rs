use std::fmt;

use rustc_abi::ExternAbi;
use rustc_feature::Features;
use rustc_session::Session;
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

pub(crate) fn enabled_names(features: &rustc_feature::Features, span: Span) -> Vec<&'static str> {
    ExternAbi::ALL_VARIANTS
        .into_iter()
        .filter(|abi| extern_abi_enabled(features, span, **abi).is_ok())
        .map(|abi| abi.as_str())
        .collect()
}

pub(crate) fn extern_abi_enabled(
    features: &rustc_feature::Features,
    span: Span,
    abi: ExternAbi,
) -> Result<(), UnstableAbi> {
    extern_abi_stability(abi).or_else(|unstable @ UnstableAbi { feature, .. }| {
        if features.enabled(feature) || span.allows_unstable(feature) {
            Ok(())
        } else {
            Err(unstable)
        }
    })
}

#[allow(rustc::untranslatable_diagnostic)]
pub(crate) fn gate_unstable_abi(sess: &Session, features: &Features, span: Span, abi: ExternAbi) {
    match extern_abi_enabled(features, span, abi) {
        Ok(_) => (),
        Err(unstable_abi) => {
            let explain = unstable_abi.to_string();
            feature_err(sess, unstable_abi.feature, span, explain).emit();
        }
    }
}

pub struct UnstableAbi {
    abi: ExternAbi,
    feature: Symbol,
    explain: GateReason,
}

enum GateReason {
    Experimental,
    ImplDetail,
}

impl fmt::Display for UnstableAbi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { abi, .. } = self;
        match self.explain {
            GateReason::Experimental => {
                write!(f, "the extern {abi} ABI is experimental and subject to change")
            }
            GateReason::ImplDetail => {
                write!(f, "the extern {abi} ABI is an implementation detail and perma-unstable")
            }
        }
    }
}

pub fn extern_abi_stability(abi: ExternAbi) -> Result<(), UnstableAbi> {
    match abi {
        // stable ABIs
        ExternAbi::Rust
        | ExternAbi::C { .. }
        | ExternAbi::Cdecl { .. }
        | ExternAbi::Stdcall { .. }
        | ExternAbi::Fastcall { .. }
        | ExternAbi::Thiscall { .. }
        | ExternAbi::Aapcs { .. }
        | ExternAbi::Win64 { .. }
        | ExternAbi::SysV64 { .. }
        | ExternAbi::System { .. }
        | ExternAbi::EfiApi => Ok(()),
        ExternAbi::Unadjusted => {
            Err(UnstableAbi { abi, feature: sym::abi_unadjusted, explain: GateReason::ImplDetail })
        }
        // experimental
        ExternAbi::Vectorcall { .. } => Err(UnstableAbi {
            abi,
            feature: sym::abi_vectorcall,
            explain: GateReason::Experimental,
        }),
        ExternAbi::RustCall => Err(UnstableAbi {
            abi,
            feature: sym::unboxed_closures,
            explain: GateReason::Experimental,
        }),
        ExternAbi::RustCold => {
            Err(UnstableAbi { abi, feature: sym::rust_cold_cc, explain: GateReason::Experimental })
        }
        ExternAbi::GpuKernel => Err(UnstableAbi {
            abi,
            feature: sym::abi_gpu_kernel,
            explain: GateReason::Experimental,
        }),
        ExternAbi::PtxKernel => {
            Err(UnstableAbi { abi, feature: sym::abi_ptx, explain: GateReason::Experimental })
        }
        ExternAbi::Msp430Interrupt => Err(UnstableAbi {
            abi,
            feature: sym::abi_msp430_interrupt,
            explain: GateReason::Experimental,
        }),
        ExternAbi::X86Interrupt => Err(UnstableAbi {
            abi,
            feature: sym::abi_x86_interrupt,
            explain: GateReason::Experimental,
        }),
        ExternAbi::AvrInterrupt | ExternAbi::AvrNonBlockingInterrupt => Err(UnstableAbi {
            abi,
            feature: sym::abi_avr_interrupt,
            explain: GateReason::Experimental,
        }),
        ExternAbi::RiscvInterruptM | ExternAbi::RiscvInterruptS => Err(UnstableAbi {
            abi,
            feature: sym::abi_riscv_interrupt,
            explain: GateReason::Experimental,
        }),
        ExternAbi::CCmseNonSecureCall => Err(UnstableAbi {
            abi,
            feature: sym::abi_c_cmse_nonsecure_call,
            explain: GateReason::Experimental,
        }),
        ExternAbi::CCmseNonSecureEntry => Err(UnstableAbi {
            abi,
            feature: sym::cmse_nonsecure_entry,
            explain: GateReason::Experimental,
        }),
        ExternAbi::Custom => {
            Err(UnstableAbi { abi, feature: sym::abi_custom, explain: GateReason::Experimental })
        }
    }
}
