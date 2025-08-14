use rustc_errors::DiagArgValue;
use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::{AttributeKind, MacroUseArgs};
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::attributes::{AcceptMapping, AttributeParser, NoArgsAttributeParser, OnDuplicate};
use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics;

pub(crate) struct MacroEscapeParser;
impl<S: Stage> NoArgsAttributeParser<S> for MacroEscapeParser {
    const PATH: &[Symbol] = &[sym::macro_escape];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
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
                    let suggestions = MACRO_USE_TEMPLATE.suggestions(false, sym::macro_use);
                    cx.emit_err(session_diagnostics::IllFormedAttributeInputLint {
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

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        Some(AttributeKind::MacroUse { span: self.first_span?, arguments: self.state })
    }
}

pub(crate) struct AllowInternalUnsafeParser;

impl<S: Stage> NoArgsAttributeParser<S> for AllowInternalUnsafeParser {
    const PATH: &[Symbol] = &[sym::allow_internal_unsafe];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const CREATE: fn(Span) -> AttributeKind = |span| AttributeKind::AllowInternalUnsafe(span);
}
