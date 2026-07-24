use rustc_feature::AttributeStability;
use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;

use super::prelude::*;
use crate::session_diagnostics::InlineForceInlineConflict;

#[derive(Default)]
pub(crate) struct InlineParser {
    pub(crate) rustc_force_inline: Option<ForceInlineState>,
    pub(crate) inline: Option<InlineState>,
}

pub(crate) struct InlineState {
    attr: InlineAttr,
    span: Span,
}

pub(crate) struct ForceInlineState {
    reason: Option<Symbol>,
    span: Span,
}

impl AttributeParser for InlineParser {
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::ManuallyChecked;
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (
            &[sym::inline],
            template!(
                Word,
                List: &["always", "never"],
                "https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute"
            ),
            AttributeStability::Stable,
            |group: &mut Self, cx, args| {
                let span = cx.attr_span;

                let attr_args = match args {
                    ArgParser::NoArgs => "".to_string(),
                    ArgParser::NameValue(_) => "".to_string(),

                    ArgParser::List(list) => list
                        .as_single()
                        .and_then(|item| {
                            item.meta_item().and_then(|item| match item.path().word_sym() {
                                Some(sym::always) => Some("(always)".to_string()),
                                Some(sym::never) => Some("(never)".to_string()),
                                _ => None,
                            })
                        })
                        .unwrap_or_else(|| "".to_string()),
                };

                let parse_attr = |cx: &mut AcceptContext<'_, '_>| match args {
                    ArgParser::NoArgs => Some(InlineAttr::Hint),

                    ArgParser::List(list) => {
                        let Some(item) = list.as_single().and_then(|e| e.meta_item()) else {
                            cx.adcx().expected_single_argument(list.span, list.len());
                            return None;
                        };

                        if item.args().as_no_args().is_err() {
                            cx.adcx().expected_no_args(item.span());
                            return None;
                        }

                        let Some(word) = item.path().word_sym() else {
                            cx.adcx().warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                            return None;
                        };

                        match word {
                            sym::always => Some(InlineAttr::Always),
                            sym::never => Some(InlineAttr::Never),
                            _ => {
                                cx.adcx().expected_specific_argument(
                                    item.path().span(),
                                    &[sym::always, sym::never],
                                );
                                return None;
                            }
                        }
                    }

                    ArgParser::NameValue(_) => {
                        cx.adcx().warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                        return None;
                    }
                };

                cx.check_target(
                    &attr_args,
                    &AllowedTargets::AllowList(&[
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
                    ]),
                );

                if let Some(prev) = &group.inline {
                    cx.warn_unused_duplicate_future_error(prev.span, span);
                    return;
                }

                if let Some(attr) = parse_attr(cx) {
                    group.inline = Some(InlineState { attr, span });
                }
            },
        ),
        (
            &[sym::rustc_force_inline],
            template!(Word, List: &["reason"], NameValueStr: "reason"),
            unstable!(
                rustc_attrs,
                "the `rustc_force_inline` attribute forces a free function to be inlined"
            ),
            |group: &mut Self, cx, args| {
                let span = cx.attr_span;

                let (reason, attr_args) = match args {
                    ArgParser::NoArgs => (None, "".to_string()),
                    ArgParser::List(list) => {
                        let single = cx.expect_single(list);
                        let reason = single.and_then(|item| cx.expect_string_literal(item));
                        let attr_args =
                            reason.map_or_else(|| "(...)".to_string(), |sym| format!("({})", sym));
                        (reason, attr_args)
                    }
                    ArgParser::NameValue(nv) => {
                        let reason = cx.expect_string_literal(nv);
                        let attr_args = if let Some(val) = reason {
                            format!(" = \"{}\"", val)
                        } else {
                            " = \"...\"".to_string()
                        };
                        (reason, attr_args)
                    }
                };

                cx.check_target(
                    &attr_args,
                    &AllowedTargets::AllowList(&[
                        Allow(Target::Fn),
                        Allow(Target::Method(MethodKind::Inherent)),
                    ]),
                );

                if let Some(prev) = &group.rustc_force_inline {
                    cx.warn_unused_duplicate(prev.span, span);
                    return;
                }

                group.rustc_force_inline = Some(ForceInlineState { reason, span });
            },
        ),
    ];

    fn finalize(self, cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        match (self.inline, self.rustc_force_inline) {
            (Some(inline), None) => Some(AttributeKind::Inline(inline.attr, inline.span)),
            (None, Some(force_inline)) => Some(AttributeKind::Inline(
                InlineAttr::Force { attr_span: force_inline.span, reason: force_inline.reason },
                force_inline.span,
            )),
            (Some(inline), Some(force_inline)) => {
                cx.emit_err(InlineForceInlineConflict {
                    inline_span: inline.span,
                    force_inline_span: force_inline.span,
                });
                None
            }
            (None, None) => None,
        }
    }
}
