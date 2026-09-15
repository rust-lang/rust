use rustc_feature::AttributeStability;
use rustc_target::spec::Arch;

use super::prelude::*;
use crate::diagnostics::BtfRelocatableOnNonBpfArch;

pub(crate) struct BtfRelocatableParser;

impl NoArgsAttributeParser for BtfRelocatableParser {
    const PATH: &[Symbol] = &[sym::btf_relocatable];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowList(&[Allow(Target::Struct), Allow(Target::Union)]);
    const STABILITY: AttributeStability = unstable!(btf_relocations);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::BtfRelocatable;

    fn finalize_check(cx: &FinalizeCheckContext<'_, '_>, attr_span: Span) {
        // `#[btf_relocatable]` may be only applied on BPF architecture.
        if cx.shared.cx.sess().target.arch != Arch::Bpf {
            cx.shared.cx.dcx().emit_err(BtfRelocatableOnNonBpfArch { span: attr_span });
        }
    }
}
