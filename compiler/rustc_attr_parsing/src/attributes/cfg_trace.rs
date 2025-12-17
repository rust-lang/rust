use rustc_feature::AttributeTemplate;
use rustc_hir::attrs::{AttributeKind, CfgEntry};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{CombineAttributeParser, ConvertFn};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};
use crate::{CFG_TEMPLATE, parse_cfg_entry};

pub(crate) struct CfgTraceParser;

impl<S: Stage> CombineAttributeParser<S> for CfgTraceParser {
    const PATH: &[Symbol] = &[sym::cfg_trace];
    type Item = (CfgEntry, Span);
    const CONVERT: ConvertFn<Self::Item> = |c, _| AttributeKind::CfgTrace(c);
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = CFG_TEMPLATE;

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let Some(list) = args.list() else {
            return None;
        };
        let Some(entry) = list.single() else {
            return None;
        };

        Some((parse_cfg_entry(cx, entry).ok()?, cx.attr_span))
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
