//! The compiler code necessary to support the cfg! extension, which expands to
//! a literal `true` or `false` based on whether the given cfg matches the
//! current compilation environment.

use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::PResult;
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_parse::exp;
use rustc_span::Span;
use {rustc_ast as ast, rustc_attr_parsing as attr};

use crate::errors;

pub(crate) fn expand_cfg(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);

    ExpandResult::Ready(match parse_cfg(cx, sp, tts) {
        Ok(cfg) => {
            let matches_cfg = attr::cfg_matches(
                &cfg,
                &cx.sess,
                cx.current_expansion.lint_node_id,
                Some(cx.ecfg.features),
            );
            MacEager::expr(cx.expr_bool(sp, matches_cfg))
        }
        Err(err) => {
            let guar = err.emit();
            DummyResult::any(sp, guar)
        }
    })
}

fn parse_cfg<'a>(
    cx: &ExtCtxt<'a>,
    span: Span,
    tts: TokenStream,
) -> PResult<'a, ast::MetaItemInner> {
    let mut p = cx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(cx.dcx().create_err(errors::RequiresCfgPattern { span }));
    }

    let cfg = p.parse_meta_item_inner()?;

    let _ = p.eat(exp!(Comma));

    if !p.eat(exp!(Eof)) {
        return Err(cx.dcx().create_err(errors::OneCfgPattern { span }));
    }

    Ok(cfg)
}
