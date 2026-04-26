use rustc_hir::attrs::{DebugVisualizer, DebuggerVisualizerType};

use super::prelude::*;

pub(crate) struct DebuggerViualizerParser;

impl<S: Stage> CombineAttributeParser<S> for DebuggerViualizerParser {
    const PATH: &[Symbol] = &[sym::debugger_visualizer];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Mod), Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(
        List: &[r#"natvis_file = "...", gdb_script_file = "...""#],
        "https://doc.rust-lang.org/reference/attributes/debugger.html#the-debugger_visualizer-attribute"
    );

    type Item = DebugVisualizer;
    const CONVERT: ConvertFn<Self::Item> = |v, _| AttributeKind::DebuggerVisualizer(v);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let single = cx.expect_single_element_list(args, cx.attr_span)?;
        let (ident, args) = cx.expect_name_value(single, single.span(), None)?;
        let visualizer_type = match ident.name {
            sym::natvis_file => DebuggerVisualizerType::Natvis,
            sym::gdb_script_file => DebuggerVisualizerType::GdbPrettyPrinter,
            _ => {
                cx.adcx().expected_specific_argument(
                    ident.span,
                    &[sym::natvis_file, sym::gdb_script_file],
                );
                return None;
            }
        };

        let Some(path) = args.value_as_str() else {
            cx.adcx().expected_string_literal(args.value_span, Some(args.value_as_lit()));
            return None;
        };

        Some(DebugVisualizer { span: ident.span.to(args.value_span), visualizer_type, path })
    }
}
