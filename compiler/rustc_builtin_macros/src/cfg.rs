//! The compiler code necessary to support the cfg! extension, which expands to
//! a literal `true` or `false` based on whether the given cfg matches the
//! current compilation environment.

use crate::errors;
use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_attr as attr;
use rustc_errors::PResult;
use rustc_expand::base::{DummyResult, ExtCtxt, MacEager, MacResult};
use rustc_span::Span;

pub fn expand_cfg(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);

    match parse_cfg(cx, sp, tts) {
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
    }
}

fn parse_cfg<'a>(cx: &mut ExtCtxt<'a>, span: Span, tts: TokenStream) -> PResult<'a, ast::MetaItem> {
    let mut p = cx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(cx.dcx().create_err(errors::RequiresCfgPattern { span }));
    }

    let cfg = p.parse_meta_item()?;

    let _ = p.eat(&token::Comma);

    if !p.eat(&token::Eof) {
        return Err(cx.dcx().create_err(errors::OneCfgPattern { span }));
    }

    Ok(cfg)
}
