use rustc_attr_ir::{Attribute, AttributeKind, InlineAttr, find_attr};
use rustc_feature::AttributeStability;
use rustc_lint_defs::builtin::{ILL_FORMED_ATTRIBUTE_INPUT, UNUSED_ATTRIBUTES};

use super::prelude::*;
use crate::diagnostics::{InlineForceInlineConflict, InlineIgnoredForExported, UnusedMultiple};

#[derive(Default)]
pub(crate) struct InlineParser(Option<(InlineAttr, Span)>);

impl AttributeParser for InlineParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (
            &[sym::inline],
            template!(
                Word,
                List: &["always", "never"],
                "https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute"
            ),
            AttributeStability::Stable,
            |this, cx, args| {
                cx.check_target(
                    "",
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
                        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
                        Warn(Target::AssocConst(AssocCtxt::Trait)),
                        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
                        Warn(Target::MacroCall),
                    ]),
                );

                let inline_span = cx.attr_span;
                let attr = match args {
                    ArgParser::NoArgs => InlineAttr::Hint,
                    ArgParser::List(list) => {
                        let Some(l) = cx.expect_single(list) else { return };

                        match l.meta_item_no_args().and_then(|i| i.path().word_sym()) {
                            Some(sym::always) => InlineAttr::Always,
                            Some(sym::never) => InlineAttr::Never,
                            _ => {
                                cx.adcx().expected_specific_argument(
                                    l.span(),
                                    &[sym::always, sym::never],
                                );
                                return;
                            }
                        }
                    }
                    ArgParser::NameValue(_) => {
                        cx.adcx().warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                        return;
                    }
                };

                match this.0 {
                    None => this.0 = Some((attr, inline_span)),
                    Some((InlineAttr::Force { .. }, force_inline_span)) => {
                        cx.emit_err(InlineForceInlineConflict { inline_span, force_inline_span });
                    }
                    Some((_, first_inline_span)) => {
                        cx.warn_unused_duplicate_future_error(first_inline_span, inline_span);
                    }
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
            |this, cx, args| {
                cx.check_target(
                    "",
                    &AllowedTargets::AllowList(&[
                        Allow(Target::Fn),
                        Allow(Target::Method(MethodKind::Inherent)),
                    ]),
                );
                let force_inline_span = cx.attr_span;
                let reason = match args {
                    ArgParser::NoArgs => None,
                    ArgParser::List(list) => {
                        let Some(l) = cx.expect_single(list) else { return };

                        let Some(reason) = cx.expect_string_literal(l) else { return };

                        Some(reason)
                    }
                    ArgParser::NameValue(v) => cx.expect_string_literal(v),
                };

                let attr = InlineAttr::Force { attr_span: cx.attr_span, reason };

                match this.0 {
                    None => this.0 = Some((attr, force_inline_span)),
                    Some((InlineAttr::Force { .. }, first_force_inline_span)) => {
                        cx.emit_err(UnusedMultiple {
                            this: force_inline_span,
                            other: first_force_inline_span,
                            name: sym::rustc_force_inline,
                        });
                    }
                    Some((_, inline_span)) => {
                        cx.emit_err(InlineForceInlineConflict { inline_span, force_inline_span });
                    }
                }
            },
        ),
    ];

    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::ManuallyChecked;

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        self.0.map(|(a, s)| AttributeKind::Inline(a, s))
    }

    fn deferred_finalize_check(
        &self,
    ) -> Option<(fn(&mut FinalizeCheckContext<'_, '_>, Span), Span)> {
        Some((
            |cx: &mut FinalizeCheckContext<'_, '_>, span| {
                let exported = cx.parsed_attrs.iter().any(|attr| {
                    let Attribute::Parsed(kind) = attr else { return false };
                    kind.is_extern_indicator()
                });
                if matches!(
                    cx.target,
                    Target::Fn
                        | Target::Closure
                        | Target::Method(
                            MethodKind::Trait { body: true }
                                | MethodKind::TraitImpl
                                | MethodKind::Inherent,
                        )
                ) && !find_attr!(cx.parsed_attrs, Inline(InlineAttr::Never, _))
                    && exported
                {
                    cx.emit_lint(UNUSED_ATTRIBUTES, InlineIgnoredForExported, span);
                }
            },
            self.0?.1,
        ))
    }
}
