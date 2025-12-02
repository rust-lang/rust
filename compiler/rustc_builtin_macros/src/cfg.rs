//! The compiler code necessary to support the cfg! extension, which expands to
//! a literal `true` or `false` based on whether the given cfg matches the
//! current compilation environment.

use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrStyle, token};
use rustc_attr_parsing as attr;
use rustc_attr_parsing::parser::MetaItemOrLitParser;
use rustc_attr_parsing::{
    AttributeParser, CFG_TEMPLATE, ParsedDescription, ShouldEmit, parse_cfg_entry,
};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_hir::AttrPath;
use rustc_hir::attrs::CfgEntry;
use rustc_parse::exp;
use rustc_span::{ErrorGuaranteed, Ident, Span};

use crate::errors;

pub(crate) fn expand_cfg(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);

    ExpandResult::Ready(match parse_cfg(cx, sp, tts) {
        Ok(cfg) => {
            let matches_cfg = attr::eval_config_entry(
                cx.sess,
                &cfg,
                cx.current_expansion.lint_node_id,
                ShouldEmit::ErrorsAndLints,
            )
            .as_bool();

            MacEager::expr(cx.expr_bool(sp, matches_cfg))
        }
        Err(guar) => DummyResult::any(sp, guar),
    })
}

fn parse_cfg(cx: &ExtCtxt<'_>, span: Span, tts: TokenStream) -> Result<CfgEntry, ErrorGuaranteed> {
    let mut parser = cx.new_parser_from_tts(tts);
    if parser.token == token::Eof {
        return Err(cx.dcx().emit_err(errors::RequiresCfgPattern { span }));
    }

    let meta = MetaItemOrLitParser::parse_single(&mut parser, ShouldEmit::ErrorsAndLints)
        .map_err(|diag| diag.emit())?;
    let cfg = AttributeParser::parse_single_args(
        cx.sess,
        span,
        span,
        AttrStyle::Inner,
        AttrPath { segments: vec![Ident::from_str("cfg")].into_boxed_slice(), span },
        ParsedDescription::Macro,
        span,
        cx.current_expansion.lint_node_id,
        Some(cx.ecfg.features),
        ShouldEmit::ErrorsAndLints,
        &meta,
        parse_cfg_entry,
        &CFG_TEMPLATE,
    )?;

    let _ = parser.eat(exp!(Comma));

    if !parser.eat(exp!(Eof)) {
        return Err(cx.dcx().emit_err(errors::OneCfgPattern { span }));
    }

    Ok(cfg)
}
