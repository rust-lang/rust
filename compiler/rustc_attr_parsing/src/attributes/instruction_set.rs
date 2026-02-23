use rustc_hir::attrs::InstructionSetAttr;

use super::prelude::*;
use crate::session_diagnostics;

pub(crate) struct InstructionSetParser;

impl<S: Stage> SingleAttributeParser<S> for InstructionSetParser {
    const PATH: &[Symbol] = &[sym::instruction_set];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Closure),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
    ]);
    const TEMPLATE: AttributeTemplate = template!(List: &["set"], "https://doc.rust-lang.org/reference/attributes/codegen.html#the-instruction_set-attribute");
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        const POSSIBLE_SYMBOLS: &[Symbol] = &[sym::arm_a32, sym::arm_t32];
        const POSSIBLE_ARM_SYMBOLS: &[Symbol] = &[sym::a32, sym::t32];
        let Some(maybe_meta_item) = args.list().and_then(MetaItemListParser::single) else {
            cx.expected_specific_argument(cx.attr_span, POSSIBLE_SYMBOLS);
            return None;
        };

        let Some(meta_item) = maybe_meta_item.meta_item() else {
            cx.expected_specific_argument(maybe_meta_item.span(), POSSIBLE_SYMBOLS);
            return None;
        };

        let mut segments = meta_item.path().segments();

        let Some(architecture) = segments.next() else {
            cx.expected_specific_argument(meta_item.span(), POSSIBLE_SYMBOLS);
            return None;
        };

        let Some(instruction_set) = segments.next() else {
            cx.expected_specific_argument(architecture.span, POSSIBLE_SYMBOLS);
            return None;
        };

        let instruction_set = match architecture.name {
            sym::arm => {
                if !cx.sess.target.has_thumb_interworking {
                    cx.dcx().emit_err(session_diagnostics::UnsupportedInstructionSet {
                        span: cx.attr_span,
                        instruction_set: sym::arm,
                        current_target: &cx.sess.opts.target_triple,
                    });
                    return None;
                }
                match instruction_set.name {
                    sym::a32 => InstructionSetAttr::ArmA32,
                    sym::t32 => InstructionSetAttr::ArmT32,
                    _ => {
                        cx.expected_specific_argument(instruction_set.span, POSSIBLE_ARM_SYMBOLS);
                        return None;
                    }
                }
            }
            _ => {
                cx.expected_specific_argument(architecture.span, POSSIBLE_SYMBOLS);
                return None;
            }
        };

        Some(AttributeKind::InstructionSet(instruction_set))
    }
}
