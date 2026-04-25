// FIXME(jdonszelmann): merge these two parsers and error when both attributes are present here.
//                      note: need to model better how duplicate attr errors work when not using
//                      SingleAttributeParser which is what we have two of here.

use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;

use super::prelude::*;

pub(crate) struct InlineParser;

impl SingleAttributeParser for InlineParser {
    const PATH: &[Symbol] = &[sym::inline];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
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
        Warn(Target::AssocConst),
        Warn(Target::MacroCall),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word,
        List: &["always", "never"],
        "https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        match args {
            ArgParser::NoArgs => Some(AttributeKind::Inline(InlineAttr::Hint, cx.attr_span)),
            ArgParser::List(list) => {
                let l = cx.expect_single(list)?;

                match l.meta_item().and_then(|i| i.path().word_sym()) {
                    Some(sym::always) => {
                        Some(AttributeKind::Inline(InlineAttr::Always, cx.attr_span))
                    }
                    Some(sym::never) => {
                        Some(AttributeKind::Inline(InlineAttr::Never, cx.attr_span))
                    }
                    _ => {
                        cx.adcx().expected_specific_argument(l.span(), &[sym::always, sym::never]);
                        return None;
                    }
                }
            }
            ArgParser::NameValue(_) => {
                cx.adcx().warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                return None;
            }
        }
    }
}

pub(crate) struct RustcForceInlineParser;

impl SingleAttributeParser for RustcForceInlineParser {
    const PATH: &[Symbol] = &[sym::rustc_force_inline];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
    ]);

    const TEMPLATE: AttributeTemplate = template!(Word, List: &["reason"], NameValueStr: "reason");

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                let l = cx.expect_single(list)?;

                let Some(reason) = l.lit().and_then(|i| i.kind.str()) else {
                    cx.adcx().expected_string_literal(l.span(), l.lit());
                    return None;
                };

                Some(reason)
            }
            ArgParser::NameValue(v) => {
                let Some(reason) = v.value_as_str() else {
                    cx.adcx().expected_string_literal(v.value_span, Some(v.value_as_lit()));
                    return None;
                };

                Some(reason)
            }
        };

        Some(AttributeKind::Inline(
            InlineAttr::Force { attr_span: cx.attr_span, reason },
            cx.attr_span,
        ))
    }
}
