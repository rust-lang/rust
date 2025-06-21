use rustc_attr_data_structures::lints::AttributeLintKind;
use rustc_attr_data_structures::{AttributeKind, OptimizeAttr, UsedBy};
use rustc_feature::{AttributeTemplate, template};
use rustc_macros::Diagnostic;
use rustc_session::parse::feature_err;
use rustc_span::{Span, sym};

use super::{AcceptMapping, AttributeOrder, AttributeParser, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct OptimizeParser;

impl<S: Stage> SingleAttributeParser<S> for OptimizeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::optimize];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(List: "size|speed|none");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };

        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        let res = match single.meta_item().and_then(|i| i.path().word().map(|i| i.name)) {
            Some(sym::size) => OptimizeAttr::Size,
            Some(sym::speed) => OptimizeAttr::Speed,
            Some(sym::none) => OptimizeAttr::DoNotOptimize,
            _ => {
                cx.expected_specific_argument(single.span(), vec!["size", "speed", "none"]);
                OptimizeAttr::Default
            }
        };

        Some(AttributeKind::Optimize(res, cx.attr_span))
    }
}

pub(crate) struct ColdParser;

impl<S: Stage> SingleAttributeParser<S> for ColdParser {
    const PATH: &[rustc_span::Symbol] = &[sym::cold];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if !args.no_args() {
            cx.expected_no_args(args.span().unwrap_or(cx.attr_span));
            return None;
        };

        Some(AttributeKind::Cold(cx.attr_span))
    }
}

#[derive(Diagnostic)]
#[diag(attr_parsing_used_compiler_linker)]
pub(crate) struct UsedCompilerLinker {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Default)]
pub(crate) struct UsedParser {
    first_compiler: Option<Span>,
    first_linker: Option<Span>,
}

// A custom `AttributeParser` is used rather than a Simple attribute parser because
// - Specifying two `#[used]` attributes is a warning (but will be an error in the future)
// - But specifying two conflicting attributes: `#[used(compiler)]` and `#[used(linker)]` is already an error today
// We can change this to a Simple parser once the warning becomes an error
impl<S: Stage> AttributeParser<S> for UsedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::used],
        template!(Word, List: "compiler|linker"),
        |group: &mut Self, cx, args| {
            let used_by = match args {
                ArgParser::NoArgs => UsedBy::Linker,
                ArgParser::List(list) => {
                    let Some(l) = list.single() else {
                        cx.expected_single_argument(list.span);
                        return;
                    };

                    match l.meta_item().and_then(|i| i.path().word_sym()) {
                        Some(sym::compiler) => {
                            if !cx.features().used_with_arg() {
                                feature_err(
                                    &cx.sess(),
                                    sym::used_with_arg,
                                    cx.attr_span,
                                    "`#[used(compiler)]` is currently unstable",
                                )
                                .emit();
                            }
                            UsedBy::Compiler
                        }
                        Some(sym::linker) => {
                            if !cx.features().used_with_arg() {
                                feature_err(
                                    &cx.sess(),
                                    sym::used_with_arg,
                                    cx.attr_span,
                                    "`#[used(linker)]` is currently unstable",
                                )
                                .emit();
                            }
                            UsedBy::Linker
                        }
                        _ => {
                            cx.expected_specific_argument(l.span(), vec!["compiler", "linker"]);
                            return;
                        }
                    }
                }
                ArgParser::NameValue(_) => return,
            };

            if let Some(prev) = group.first_compiler.or(group.first_linker) {
                cx.emit_lint(
                    AttributeLintKind::UnusedDuplicate {
                        this: cx.attr_span,
                        other: prev,
                        warning: true,
                    },
                    cx.attr_span,
                );
            }

            match used_by {
                UsedBy::Compiler { .. } => {
                    if group.first_compiler.is_none() {
                        group.first_compiler = Some(cx.attr_span);
                    }
                }
                UsedBy::Linker => {
                    if group.first_linker.is_none() {
                        group.first_linker = Some(cx.attr_span);
                    }
                }
            }
        },
    )];

    fn finalize(self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let (Some(linker_span), Some(compiler_span)) = (self.first_linker, self.first_compiler) {
            cx.dcx().emit_err(UsedCompilerLinker { spans: vec![linker_span, compiler_span] });
        }

        match (self.first_linker, self.first_compiler) {
            (_, Some(span)) => Some(AttributeKind::Used { used_by: UsedBy::Compiler, span }),
            (Some(span), _) => Some(AttributeKind::Used { used_by: UsedBy::Linker, span }),
            _ => None,
        }
    }
}
