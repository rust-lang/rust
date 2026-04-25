use rustc_hir::attrs::{DebugVisualizer, DebuggerVisualizerType};

use super::prelude::*;

pub(crate) struct DebuggerViualizerParser;

impl CombineAttributeParser for DebuggerViualizerParser {
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
        cx: &mut AcceptContext<'_, '_>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let single = cx.expect_single_element_list(args, cx.attr_span)?;
        let Some(mi) = single.meta_item() else {
            cx.adcx().expected_name_value(single.span(), None);
            return None;
        };
        let path = mi.path().word_sym();
        let visualizer_type = match path {
            Some(sym::natvis_file) => DebuggerVisualizerType::Natvis,
            Some(sym::gdb_script_file) => DebuggerVisualizerType::GdbPrettyPrinter,
            _ => {
                cx.adcx().expected_specific_argument(
                    mi.path().span(),
                    &[sym::natvis_file, sym::gdb_script_file],
                );
                return None;
            }
        };

        let Some(path) = mi.args().name_value() else {
            cx.adcx().expected_name_value(single.span(), path);
            return None;
        };

        let Some(path) = path.value_as_str() else {
            cx.adcx().expected_string_literal(path.value_span, Some(path.value_as_lit()));
            return None;
        };

        Some(DebugVisualizer { span: mi.span(), visualizer_type, path })
    }
}
