use rustc_attr_data_structures::{AttributeKind, MacroUseArgs};
use rustc_errors::DiagArgValue;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Ident, Span, Symbol, sym};
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
    /// All specific imports found so far
    uses: ThinVec<Ident>,
    /// Span of the first `#[macro_use]` arguments without arguments, used for linting
    use_all: Option<Span>,
    /// Spans of all `#[macro_use]` arguments with arguments, used for linting
    uses_attr_spans: ThinVec<Span>,
    /// Span of the first `#[macro_use]` argument, used as the span for this attribute
    first_span: Option<Span>,
}

const MACRO_USE_TEMPLATE: AttributeTemplate = template!(Word, List: "name1, name2, ...");

impl<S: Stage> AttributeParser<S> for MacroUseParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::macro_use],
        MACRO_USE_TEMPLATE,
        |group: &mut Self, cx: &mut AcceptContext<'_, '_, S>, args| {
            let span = cx.attr_span;
            group.first_span.get_or_insert(span);
            match args {
                ArgParser::NoArgs => {
                    // If there is a `#[macro_use]` import already, give a warning
                    if let Some(old_attr) = group.use_all.replace(span) {
                        cx.warn_unused_duplicate(old_attr, span);
                    }
                }
                ArgParser::List(list) => {
                    let mut arguments = ThinVec::new();

                    if list.is_empty() {
                        cx.warn_empty_attribute(list.span);
                        return;
                    }
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

                    group.uses.extend(arguments);
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
                    return;
                }
            };
        },
    )];

    fn finalize(self, cx: &mut FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        let arguments = if let Some(use_all) = self.use_all {
            // If there is a `#[macro_use]` attribute, warn on all `#[macro_use(...)]` attributes since everything is already imported
            for specific_use in self.uses_attr_spans {
                cx.warn_unused_duplicate(use_all, specific_use);
            }
            MacroUseArgs::UseAll
        } else {
            MacroUseArgs::UseSpecific(self.uses)
        };
        Some(AttributeKind::MacroUse { span: self.first_span?, arguments })
    }
}
