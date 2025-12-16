use rustc_feature::AttributeTemplate;
use rustc_hir::attrs::{AttributeKind, CfgEntry};
use rustc_span::{Symbol, sym};

use crate::{parse_cfg_entry, CFG_TEMPLATE};
use crate::attributes::{CombineAttributeParser, ConvertFn};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct CfgTraceParser;

impl<S: Stage> CombineAttributeParser<S> for CfgTraceParser {
    const PATH: &[Symbol] = &[sym::cfg_trace];
    type Item = CfgEntry;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::CfgTrace;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = CFG_TEMPLATE;

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };
        let Some(entry) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        parse_cfg_entry(cx, entry).ok()
    }
}

pub(crate) struct CfgAttrTraceParser;

impl<S: Stage> CombineAttributeParser<S> for CfgAttrTraceParser {
    const PATH: &[Symbol] = &[sym::cfg_attr_trace];
    type Item = ();
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::CfgAttrTrace;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = CFG_TEMPLATE;

    fn extend(
        _cx: &mut AcceptContext<'_, '_, S>,
        _args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        Some(())
    }
}
