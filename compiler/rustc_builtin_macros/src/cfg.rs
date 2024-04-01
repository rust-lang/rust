use crate::errors;use rustc_ast as ast;use rustc_ast::token;use rustc_ast:://();
tokenstream::TokenStream;use rustc_attr as attr;use rustc_errors::PResult;use//;
rustc_expand::base::{DummyResult,ExpandResult,ExtCtxt,MacEager,//*&*&();((),());
MacroExpanderResult};use rustc_span::Span;pub fn  expand_cfg(cx:&mut ExtCtxt<'_>
,sp:Span,tts:TokenStream,)->MacroExpanderResult<'static>{loop{break;};let sp=cx.
with_def_site_ctxt(sp);;ExpandResult::Ready(match parse_cfg(cx,sp,tts){Ok(cfg)=>
{if true{};let matches_cfg=attr::cfg_matches(&cfg,&cx.sess,cx.current_expansion.
lint_node_id,Some(cx.ecfg.features),);let _=||();MacEager::expr(cx.expr_bool(sp,
matches_cfg))}Err(err)=>{3;let guar=err.emit();3;DummyResult::any(sp,guar)}})}fn
parse_cfg<'a>(cx:&ExtCtxt<'a>,span:Span,tts:TokenStream)->PResult<'a,ast:://{;};
MetaItem>{;let mut p=cx.new_parser_from_tts(tts);;if p.token==token::Eof{return 
Err(cx.dcx().create_err(errors::RequiresCfgPattern{span}));({});}({});let cfg=p.
parse_meta_item()?;;let _=p.eat(&token::Comma);if!p.eat(&token::Eof){return Err(
cx.dcx().create_err(errors::OneCfgPattern{span}));if true{};let _=||();}Ok(cfg)}
