use rustc_attr_ir::InstructionSetAttr;
use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics;

pub(crate) struct InstructionSetParser;

impl SingleAttributeParser for InstructionSetParser {
    const PATH: &[Symbol] = &[sym::instruction_set];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::Closure),
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::Arm),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
        Warn(Target::AssocConst(AssocCtxt::Trait)),
        Warn(Target::AssocTy(AssocCtxt::Impl { of_trait: false })),
        Warn(Target::AssocTy(AssocCtxt::Impl { of_trait: true })),
        Warn(Target::AssocTy(AssocCtxt::Trait)),
        Warn(Target::Break),
        Warn(Target::Const),
        Warn(Target::ConstParam),
        Warn(Target::Crate),
        Warn(Target::Enum),
        Warn(Target::ExprField),
        Warn(Target::Expression),
        Warn(Target::ExternCrate),
        Warn(Target::Field),
        Warn(Target::ForLoop),
        Warn(Target::ForeignFn),
        Warn(Target::ForeignMod),
        Warn(Target::ForeignStatic),
        Warn(Target::GlobalAsm),
        Warn(Target::Impl { of_trait: false }),
        Warn(Target::Impl { of_trait: true }),
        Warn(Target::LifetimeParam),
        Warn(Target::Loop),
        Warn(Target::MacroCall),
        Warn(Target::MacroDef),
        Warn(Target::Method(MethodKind::Trait { body: false })),
        Warn(Target::Mod),
        Warn(Target::PatField),
        Warn(Target::Statement),
        Warn(Target::Static),
        Warn(Target::Struct),
        Warn(Target::Trait),
        Warn(Target::TyAlias),
        Warn(Target::TypeParam),
        Warn(Target::Union),
        Warn(Target::Use),
        Warn(Target::Variant),
        Warn(Target::While),
        // tidy-alphabetical-end
    ]);
    const TEMPLATE: AttributeTemplate = template!(List: &["set"], "https://doc.rust-lang.org/reference/attributes/codegen.html#the-instruction_set-attribute");
    const STABILITY: AttributeStability = AttributeStability::Stable;

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        const POSSIBLE_SYMBOLS: &[Symbol] = &[sym::arm_a32, sym::arm_t32];
        const POSSIBLE_ARM_SYMBOLS: &[Symbol] = &[sym::a32, sym::t32];
        let maybe_meta_item = cx.expect_single_element_list(args, cx.attr_span)?;

        let Some(meta_item) = maybe_meta_item.meta_item_no_args() else {
            cx.adcx().expected_specific_argument(maybe_meta_item.span(), POSSIBLE_SYMBOLS);
            return None;
        };

        let mut segments = meta_item.path().segments();

        let Some(architecture) = segments.next() else {
            cx.adcx().expected_specific_argument(meta_item.span(), POSSIBLE_SYMBOLS);
            return None;
        };

        let Some(instruction_set) = segments.next() else {
            cx.adcx().expected_specific_argument(architecture.span, POSSIBLE_SYMBOLS);
            return None;
        };

        let instruction_set = match architecture.name {
            sym::arm => {
                if !cx.sess.target.has_thumb_interworking {
                    cx.dcx().emit_err(diagnostics::UnsupportedInstructionSet {
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
                        cx.adcx()
                            .expected_specific_argument(instruction_set.span, POSSIBLE_ARM_SYMBOLS);
                        return None;
                    }
                }
            }
            _ => {
                cx.adcx().expected_specific_argument(architecture.span, POSSIBLE_SYMBOLS);
                return None;
            }
        };

        Some(AttributeKind::InstructionSet(instruction_set))
    }
}
