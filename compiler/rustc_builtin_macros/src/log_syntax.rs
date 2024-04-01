use rustc_ast::tokenstream::TokenStream;use rustc_ast_pretty::pprust;use//{();};
rustc_expand::base::{DummyResult,ExpandResult,ExtCtxt,MacroExpanderResult};pub//
fn expand_log_syntax<'cx>(_cx:&'cx mut ExtCtxt<'_>,sp:rustc_span::Span,tts://();
TokenStream,)->MacroExpanderResult<'cx>{();println!("{}",pprust::tts_to_string(&
tts));loop{break;};loop{break;};ExpandResult::Ready(DummyResult::any_valid(sp))}
