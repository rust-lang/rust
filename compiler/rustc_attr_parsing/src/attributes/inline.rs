use rustc_attr_ir::{AttributeKind, InlineAttr, find_attr};
use rustc_feature::AttributeStability;
use rustc_lint_defs::builtin::ILL_FORMED_ATTRIBUTE_INPUT;

use super::prelude::*;
use crate::diagnostics::InlineForceInlineConflict;

pub(crate) struct InlineParser;

impl SingleAttributeParser for InlineParser {
    const PATH: &[Symbol] = &[sym::inline];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Closure),
        Allow(Target::Delegation { mac: false }),
        Warn(Target::Method(MethodKind::Trait { body: false })),
        Warn(Target::ForeignFn),
        Warn(Target::Field),
        Warn(Target::MacroDef),
        Warn(Target::Arm),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
        Warn(Target::AssocConst(AssocCtxt::Trait)),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
        Warn(Target::MacroCall),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word,
        List: &["always", "never"],
        "https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute"
    );
    const STABILITY: AttributeStability = AttributeStability::Stable;

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        match args {
            ArgParser::NoArgs => Some(AttributeKind::Inline(InlineAttr::Hint, cx.attr_span)),
            ArgParser::List(list) => {
                let l = cx.expect_single(list)?;

                match l.meta_item_no_args().and_then(|i| i.path().word_sym()) {
                    Some(sym::always) => {
                        Some(AttributeKind::Inline(InlineAttr::Always, cx.attr_span))
                    }
                    Some(sym::never) => {
                        Some(AttributeKind::Inline(InlineAttr::Never, cx.attr_span))
                    }
                    _ => {
                        cx.adcx().expected_specific_argument(l.span(), &[sym::always, sym::never]);
                        None
                    }
                }
            }
            ArgParser::NameValue(_) => {
                cx.adcx().warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                None
            }
        }
    }
}

pub(crate) struct RustcForceInlineParser;

impl SingleAttributeParser for RustcForceInlineParser {
    const PATH: &[Symbol] = &[sym::rustc_force_inline];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
    ]);
    const STABILITY: AttributeStability = unstable!(
        rustc_attrs,
        "the `rustc_force_inline` attribute forces a free function to be inlined"
    );
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["reason"], NameValueStr: "reason");

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                let l = cx.expect_single(list)?;

                let reason = cx.expect_string_literal(l)?;

                Some(reason)
            }
            ArgParser::NameValue(v) => cx.expect_string_literal(v),
        };

        Some(AttributeKind::Inline(
            InlineAttr::Force { attr_span: cx.attr_span, reason },
            cx.attr_span,
        ))
    }

    fn finalize_check(cx: &FinalizeCheckContext<'_, '_>, attr_span: Span) {
        let Some(inline_span) = find_attr!(cx.parsed_attrs, Inline(attr, span) if !matches!(attr, InlineAttr::Force { .. }) => span)
        else {
            return;
        };

        cx.emit_err(InlineForceInlineConflict {
            inline_span: *inline_span,
            force_inline_span: attr_span,
        });
    }
}
