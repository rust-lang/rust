use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::AttributeKind::MacroExport;
use rustc_attr_data_structures::lints::AttributeLintKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct MacroExportParser;

impl<S: Stage> SingleAttributeParser<S> for crate::attributes::macro_attrs::MacroExportParser {
    const PATH: &[Symbol] = &[sym::macro_export];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "local_inner_macros");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let local_inner_macros = match args {
            ArgParser::NoArgs => false,
            ArgParser::List(list) => {
                let Some(l) = list.single() else {
                    cx.expected_single_argument(list.span);
                    return None;
                };
                match l.meta_item().and_then(|i| i.path().word_sym()) {
                    Some(sym::local_inner_macros) => true,
                    _ => {
                        cx.expected_specific_argument(l.span(), vec!["local_inner_macros"]);
                        return None;
                    }
                }
            }
            ArgParser::NameValue(_) => {
                let suggestions =
                    <Self as SingleAttributeParser<S>>::TEMPLATE.suggestions(false, "macro_export");
                let span = cx.attr_span;
                cx.emit_lint(AttributeLintKind::IllFormedAttributeInput { suggestions }, span);
                return None;
            }
        };
        Some(MacroExport { span: cx.attr_span, local_inner_macros })
    }
}
