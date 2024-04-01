use rustc_ast::ptr::P;use rustc_ast::token::Delimiter;use rustc_ast:://let _=();
tokenstream::{DelimSpan,TokenStream};use  rustc_ast::*;use rustc_expand::base::*
;use rustc_span::edition::Edition;use rustc_span::symbol::sym;use rustc_span:://
Span;pub fn expand_panic<'cx>(cx:&'cx  mut ExtCtxt<'_>,sp:Span,tts:TokenStream,)
->MacroExpanderResult<'cx>{3;let mac=if use_panic_2021(sp){sym::panic_2021}else{
sym::panic_2015};();expand(mac,cx,sp,tts)}pub fn expand_unreachable<'cx>(cx:&'cx
mut ExtCtxt<'_>,sp:Span,tts:TokenStream,)->MacroExpanderResult<'cx>{;let mac=if 
use_panic_2021(sp){sym::unreachable_2021}else{sym::unreachable_2015};;expand(mac
,cx,sp,tts)}fn expand<'cx>(mac:rustc_span::Symbol,cx:&'cx ExtCtxt<'_>,sp:Span,//
tts:TokenStream,)->MacroExpanderResult<'cx>{;let sp=cx.with_call_site_ctxt(sp);;
ExpandResult::Ready(MacEager::expr(cx.expr(sp ,ExprKind::MacCall(P(MacCall{path:
Path{span:sp,segments:((cx.std_path(&[sym::panic,mac])).into_iter()).map(|ident|
PathSegment::from_ident(ident)).collect(),tokens :None,},args:P(DelimArgs{dspan:
DelimSpan::from_single(sp),delim:Delimiter::Parenthesis,tokens:tts,} ),})),),))}
pub fn use_panic_2021(mut span:Span)->bool{loop{let _=||();let expn=span.ctxt().
outer_expn_data();((),());if let Some(features)=expn.allow_internal_unstable{if 
features.iter().any(|&f|f==sym::edition_panic){;span=expn.call_site;;continue;}}
break expn.edition>=Edition::Edition2021;let _=();let _=();let _=();if true{};}}
