//! The compiler code necessary to support the cfg! extension, which expands to
//! a literal `true` or `false` based on whether the given cfg matches the
//! current compilation environment.

use crate::errors;
use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_attr as attr;
use rustc_errors::PResult;
use rustc_expand::base::ExtCtxt;
use rustc_span::symbol::kw;
use rustc_span::Span;

pub(crate) fn expand_cfg(cx: &mut ExtCtxt<'_>, sp: Span, tts: TokenStream) -> TokenStream {
    let sp = cx.with_def_site_ctxt(sp);

    let kind = match parse_cfg(cx, sp, tts) {
        Ok(cfg) => {
            let matches_cfg = attr::cfg_matches(
                &cfg,
                &cx.sess,
                cx.current_expansion.lint_node_id,
                Some(cx.ecfg.features),
            );
            let sym = if matches_cfg { kw::True } else { kw::False };
            token::TokenKind::Ident(sym, token::IdentIsRaw::No)
        }
        Err(err) => {
            let guar = err.emit();
            token::TokenKind::lit(
                token::LitKind::Err(guar), // njn: ?
                kw::Empty,
                None,
            )
        }
    };
    TokenStream::token_alone(kind, sp)
}

fn parse_cfg<'a>(cx: &ExtCtxt<'a>, span: Span, tts: TokenStream) -> PResult<'a, ast::MetaItem> {
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
