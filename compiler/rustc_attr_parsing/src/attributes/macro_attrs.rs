use rustc_ast::AttrStyle;
use rustc_errors::DiagArgValue;
use rustc_hir::attrs::MacroUseArgs;

use super::prelude::*;
use crate::session_diagnostics::IllFormedAttributeInputLint;

pub(crate) struct MacroEscapeParser;
impl<S: Stage> NoArgsAttributeParser<S> for MacroEscapeParser {
    const PATH: &[Symbol] = &[sym::macro_escape];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = MACRO_USE_ALLOWED_TARGETS;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::MacroEscape;
}

/// `#[macro_use]` attributes can either:
/// - Use all macros from a crate, if provided without arguments
/// - Use specific macros from a crate, if provided with arguments `#[macro_use(macro1, macro2)]`
/// A warning should be provided if an use all is combined with specific uses, or if multiple use-alls are used.
#[derive(Default)]
pub(crate) struct MacroUseParser {
    state: MacroUseArgs,

    /// Spans of all `#[macro_use]` arguments with arguments, used for linting
    uses_attr_spans: ThinVec<Span>,
    /// If `state` is `UseSpecific`, stores the span of the first `#[macro_use]` argument, used as the span for this attribute
    /// If `state` is `UseAll`, stores the span of the first `#[macro_use]` arguments without arguments
    first_span: Option<Span>,
}

const MACRO_USE_TEMPLATE: AttributeTemplate = template!(
    Word, List: &["name1, name2, ..."],
    "https://doc.rust-lang.org/reference/macros-by-example.html#the-macro_use-attribute"
);
const MACRO_USE_ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
    Allow(Target::Mod),
    Allow(Target::ExternCrate),
    Allow(Target::Crate),
    Error(Target::WherePredicate),
]);

impl<S: Stage> AttributeParser<S> for MacroUseParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::macro_use],
        MACRO_USE_TEMPLATE,
        |group: &mut Self, cx: &mut AcceptContext<'_, '_, S>, args| {
            let span = cx.attr_span;
            group.first_span.get_or_insert(span);
            match args {
                ArgParser::NoArgs => {
                    match group.state {
                        MacroUseArgs::UseAll => {
                            let first_span = group.first_span.expect(
                                "State is UseAll is some so this is not the first attribute",
                            );
                            // Since there is a `#[macro_use]` import already, give a warning
                            cx.warn_unused_duplicate(first_span, span);
                        }
                        MacroUseArgs::UseSpecific(_) => {
                            group.state = MacroUseArgs::UseAll;
                            group.first_span = Some(span);
                            // If there is a `#[macro_use]` attribute, warn on all `#[macro_use(...)]` attributes since everything is already imported
                            for specific_use in group.uses_attr_spans.drain(..) {
                                cx.warn_unused_duplicate(span, specific_use);
                            }
                        }
                    }
                }
                ArgParser::List(list) => {
                    if list.is_empty() {
                        cx.warn_empty_attribute(list.span);
                        return;
                    }

                    match &mut group.state {
                        MacroUseArgs::UseAll => {
                            let first_span = group.first_span.expect(
                                "State is UseAll is some so this is not the first attribute",
                            );
                            cx.warn_unused_duplicate(first_span, span);
                        }
                        MacroUseArgs::UseSpecific(arguments) => {
                            // Store here so if we encounter a `UseAll` later we can still lint this attribute
                            group.uses_attr_spans.push(cx.attr_span);

                            for item in list.mixed() {
                                let Some(item) = item.meta_item() else {
                                    cx.expected_identifier(item.span());
                                    continue;
                                };
                                if let Err(err_span) = item.args().no_args() {
                                    cx.expected_no_args(err_span);
                                    continue;
                                }
                                let Some(item) = item.path().word() else {
                                    cx.expected_identifier(item.span());
                                    continue;
                                };
                                arguments.push(item);
                            }
                        }
                    }
                }
                ArgParser::NameValue(_) => {
                    let suggestions = MACRO_USE_TEMPLATE.suggestions(cx.attr_style, sym::macro_use);
                    cx.emit_err(IllFormedAttributeInputLint {
                        num_suggestions: suggestions.len(),
                        suggestions: DiagArgValue::StrListSepByAnd(
                            suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                        ),
                        span,
                    });
                }
            }
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets = MACRO_USE_ALLOWED_TARGETS;

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        Some(AttributeKind::MacroUse { span: self.first_span?, arguments: self.state })
    }
}

pub(crate) struct AllowInternalUnsafeParser;

impl<S: Stage> NoArgsAttributeParser<S> for AllowInternalUnsafeParser {
    const PATH: &[Symbol] = &[sym::allow_internal_unsafe];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::MacroDef),
        Warn(Target::Field),
        Warn(Target::Arm),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |span| AttributeKind::AllowInternalUnsafe(span);
}

pub(crate) struct MacroExportParser;

impl<S: Stage> SingleAttributeParser<S> for MacroExportParser {
    const PATH: &[Symbol] = &[sym::macro_export];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["local_inner_macros"]);
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::MacroDef),
        Error(Target::WherePredicate),
        Error(Target::Crate),
    ]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let suggestions = || {
            <Self as SingleAttributeParser<S>>::TEMPLATE
                .suggestions(AttrStyle::Inner, "macro_export")
        };
        let local_inner_macros = match args {
            ArgParser::NoArgs => false,
            ArgParser::List(list) => {
                let Some(l) = list.single() else {
                    let span = cx.attr_span;
                    cx.emit_lint(
                        AttributeLintKind::InvalidMacroExportArguments {
                            suggestions: suggestions(),
                        },
                        span,
                    );
                    return None;
                };
                match l.meta_item().and_then(|i| i.path().word_sym()) {
                    Some(sym::local_inner_macros) => true,
                    _ => {
                        let span = cx.attr_span;
                        cx.emit_lint(
                            AttributeLintKind::InvalidMacroExportArguments {
                                suggestions: suggestions(),
                            },
                            span,
                        );
                        return None;
                    }
                }
            }
            ArgParser::NameValue(_) => {
                let span = cx.attr_span;
                let suggestions = suggestions();
                cx.emit_err(IllFormedAttributeInputLint {
                    num_suggestions: suggestions.len(),
                    suggestions: DiagArgValue::StrListSepByAnd(
                        suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                    ),
                    span,
                });
                return None;
            }
        };
        Some(AttributeKind::MacroExport { span: cx.attr_span, local_inner_macros })
    }
}
