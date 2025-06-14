use std::iter;

use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Span, Symbol, sym};

use super::{CombineAttributeParser, ConvertFn};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics;

pub(crate) struct AllowInternalUnstableParser;
impl<S: Stage> CombineAttributeParser<S> for AllowInternalUnstableParser {
    const PATH: &[Symbol] = &[sym::allow_internal_unstable];
    type Item = (Symbol, Span);
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::AllowInternalUnstable;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "feat1, feat2, ...");

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> {
        parse_unstable(cx, args, <Self as CombineAttributeParser<S>>::PATH[0])
            .into_iter()
            .zip(iter::repeat(cx.attr_span))
    }
}

pub(crate) struct AllowConstFnUnstableParser;
impl<S: Stage> CombineAttributeParser<S> for AllowConstFnUnstableParser {
    const PATH: &[Symbol] = &[sym::rustc_allow_const_fn_unstable];
    type Item = Symbol;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::AllowConstFnUnstable;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "feat1, feat2, ...");

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        parse_unstable(cx, args, <Self as CombineAttributeParser<S>>::PATH[0])
    }
}

fn parse_unstable<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
    symbol: Symbol,
) -> impl IntoIterator<Item = Symbol> {
    let mut res = Vec::new();

    let Some(list) = args.list() else {
        cx.emit_err(session_diagnostics::ExpectsFeatureList {
            span: cx.attr_span,
            name: symbol.to_ident_string(),
        });
        return res;
    };

    for param in list.mixed() {
        let param_span = param.span();
        if let Some(ident) = param.meta_item().and_then(|i| i.path().word()) {
            res.push(ident.name);
        } else {
            cx.emit_err(session_diagnostics::ExpectsFeatures {
                span: param_span,
                name: symbol.to_ident_string(),
            });
        }
    }

    res
}
