use std::iter;

use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use super::{CombineAttributeParser, ConvertFn};
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::session_diagnostics;

pub(crate) struct AllowInternalUnstableParser;
impl CombineAttributeParser for AllowInternalUnstableParser {
    const PATH: &'static [Symbol] = &[sym::allow_internal_unstable];
    type Item = (Symbol, Span);
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::AllowInternalUnstable;

    fn extend<'a>(
        cx: &'a AcceptContext<'a>,
        args: &'a ArgParser<'a>,
    ) -> impl IntoIterator<Item = Self::Item> + 'a {
        parse_unstable(cx, args, Self::PATH[0]).into_iter().zip(iter::repeat(cx.attr_span))
    }
}

pub(crate) struct AllowConstFnUnstableParser;
impl CombineAttributeParser for AllowConstFnUnstableParser {
    const PATH: &'static [Symbol] = &[sym::rustc_allow_const_fn_unstable];
    type Item = Symbol;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::AllowConstFnUnstable;

    fn extend<'a>(
        cx: &'a AcceptContext<'a>,
        args: &'a ArgParser<'a>,
    ) -> impl IntoIterator<Item = Self::Item> + 'a {
        parse_unstable(cx, args, Self::PATH[0])
    }
}

fn parse_unstable<'a>(
    cx: &AcceptContext<'_>,
    args: &'a ArgParser<'a>,
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
        if let Some(ident) = param.meta_item().and_then(|i| i.path_without_args().word()) {
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
