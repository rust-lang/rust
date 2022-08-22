//! The compiler code necessary to support the cfg! extension, which expands to
//! a literal `true` or `false` based on whether the given cfg matches the
//! current compilation environment.

use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_attr as attr;
use rustc_errors::PResult;
use rustc_expand::base::{self, *};
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

pub fn expand_cfg(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);

    match parse_cfg(cx, sp, tts) {
        Ok(cfg) => {
            let matches_cfg = attr::cfg_matches(
                &cfg,
                &cx.sess.parse_sess,
                cx.current_expansion.lint_node_id,
                cx.ecfg.features,
            );
            MacEager::expr(cx.expr_bool(sp, matches_cfg))
        }
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::requires_cfg_pattern)]
struct RequiresCfgPattern {
    #[primary_span]
    #[label]
    span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::expected_one_cfg_pattern)]
struct OneCfgPattern {
    #[primary_span]
    span: Span,
}

fn parse_cfg<'a>(cx: &mut ExtCtxt<'a>, span: Span, tts: TokenStream) -> PResult<'a, ast::MetaItem> {
    let mut p = cx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(cx.create_err(RequiresCfgPattern { span }));
    }

    let cfg = p.parse_meta_item()?;

    let _ = p.eat(&token::Comma);

    if !p.eat(&token::Eof) {
        return Err(cx.create_err(OneCfgPattern { span }));
    }

    Ok(cfg)
}
