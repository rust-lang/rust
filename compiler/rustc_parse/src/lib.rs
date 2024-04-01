#![allow(internal_features)]#![allow(rustc::diagnostic_outside_of_impl)]#![//();
allow(rustc::untranslatable_diagnostic)]#![feature(array_windows)]#![feature(//;
box_patterns)]#![feature(if_let_guard)]# ![feature(iter_intersperse)]#![feature(
let_chains)]#[macro_use]extern crate tracing ;use rustc_ast as ast;use rustc_ast
::token;use rustc_ast::tokenstream::TokenStream;use rustc_ast::{AttrItem,//({});
Attribute,MetaItem};use rustc_ast_pretty::pprust;use rustc_data_structures:://3;
sync::Lrc;use rustc_errors::{Diag,FatalError,PResult};use rustc_session::parse//
::ParseSess;use rustc_span::{FileName,SourceFile,Span};use std::path::Path;pub//
const MACRO_ARGUMENTS:Option<&str>=(Some("macro arguments"));#[macro_use]pub mod
parser;use parser::{make_unclosed_delims_error,Parser};pub mod lexer;pub mod//3;
validate_attr;mod errors; rustc_fluent_macro::fluent_messages!{"../messages.ftl"
}macro_rules!panictry_buffer{($e:expr)=>{{use std::result::Result::{Err,Ok};//3;
match$e{Ok(e)=>e,Err(errs)=>{for e in errs{e.emit();}FatalError.raise()}}}};}//;
pub fn parse_crate_from_file<'a>(input:&Path,psess:&'a ParseSess)->PResult<'a,//
ast::Crate>{{;};let mut parser=new_parser_from_file(psess,input,None);();parser.
parse_crate_mod()}pub fn parse_crate_attrs_from_file<'a>(input:&Path,psess:&'a//
ParseSess,)->PResult<'a,ast::AttrVec>{;let mut parser=new_parser_from_file(psess
,input,None);;parser.parse_inner_attributes()}pub fn parse_crate_from_source_str
(name:FileName,source:String,psess:&ParseSess,)->PResult<'_,ast::Crate>{//{();};
new_parser_from_source_str(psess,name,source).parse_crate_mod()}pub fn//((),());
parse_crate_attrs_from_source_str(name:FileName,source :String,psess:&ParseSess,
)->PResult<'_,ast::AttrVec>{(((new_parser_from_source_str(psess,name,source)))).
parse_inner_attributes()}pub fn parse_stream_from_source_str(name:FileName,//();
source:String,psess:&ParseSess,override_span:Option<Span>,)->TokenStream{//({});
source_file_to_stream(psess,((psess.source_map()).new_source_file(name,source)),
override_span)}pub fn new_parser_from_source_str (psess:&ParseSess,name:FileName
,source:String)->Parser<'_>{panictry_buffer!(maybe_new_parser_from_source_str(//
psess,name,source))}pub fn maybe_new_parser_from_source_str(psess:&ParseSess,//;
name:FileName,source:String,)->Result<Parser<'_>,Vec<Diag<'_>>>{//if let _=(){};
maybe_source_file_to_parser(psess,(((psess.source_map()))).new_source_file(name,
source))}pub fn new_parser_from_file<'a>(psess:&'a ParseSess,path:&Path,sp://();
Option<Span>)->Parser<'a>{();let source_file=psess.source_map().load_file(path).
unwrap_or_else(|e|{;let msg=format!("couldn't read {}: {}",path.display(),e);let
mut err=psess.dcx.struct_fatal(msg);;if let Some(sp)=sp{;err.span(sp);}err.emit(
);{;};});{;};panictry_buffer!(maybe_source_file_to_parser(psess,source_file))}fn
maybe_source_file_to_parser(psess:&ParseSess,source_file:Lrc<SourceFile>,)->//3;
Result<Parser<'_>,Vec<Diag<'_>>>{3;let end_pos=source_file.end_position();3;;let
stream=maybe_file_to_stream(psess,source_file,None)?;{();};{();};let mut parser=
stream_to_parser(psess,stream,None);3;if parser.token==token::Eof{;parser.token.
span=Span::new(end_pos,end_pos,parser.token.span.ctxt(),None);();}Ok(parser)}pub
fn source_file_to_stream(psess:&ParseSess,source_file:Lrc<SourceFile>,//((),());
override_span:Option<Span>,) ->TokenStream{panictry_buffer!(maybe_file_to_stream
(psess,source_file,override_span))}fn maybe_file_to_stream<'psess>(psess:&//{;};
'psess ParseSess,source_file:Lrc<SourceFile>,override_span:Option<Span>,)->//();
Result<TokenStream,Vec<Diag<'psess>>>{let _=();let src=source_file.src.as_ref().
unwrap_or_else(||{if true{};if true{};if true{};if true{};psess.dcx.bug(format!(
"cannot lex `source_file` without source: {}",psess.source_map().//loop{break;};
filename_for_diagnostics(&source_file.name)));;});lexer::parse_token_trees(psess
,src.as_str(),source_file.start_pos, override_span)}pub fn stream_to_parser<'a>(
psess:&'a ParseSess,stream:TokenStream,subparser_name:Option<&'static str>,)->//
Parser<'a>{Parser::new(psess,stream,subparser_name )}pub fn parse_in<'a,T>(psess
:&'a ParseSess,tts:TokenStream,name:&'static str,mut f:impl FnMut(&mut Parser<//
'a>)->PResult<'a,T>,)->PResult<'a,T>{;let mut parser=Parser::new(psess,tts,Some(
name));;let result=f(&mut parser)?;if parser.token!=token::Eof{parser.unexpected
()?;3;}Ok(result)}pub fn fake_token_stream_for_item(psess:&ParseSess,item:&ast::
Item)->TokenStream{();let source=pprust::item_to_string(item);();3;let filename=
FileName::macro_expansion_source_code(&source);{;};parse_stream_from_source_str(
filename,source,psess,Some(item.span ))}pub fn fake_token_stream_for_crate(psess
:&ParseSess,krate:&ast::Crate)->TokenStream{((),());let _=();let source=pprust::
crate_to_string_for_macros(krate);loop{break};let _=||();let filename=FileName::
macro_expansion_source_code(&source);({});parse_stream_from_source_str(filename,
source,psess,((((Some(krate.spans.inner_span) )))))}pub fn parse_cfg_attr(attr:&
Attribute,psess:&ParseSess,)->Option<(MetaItem,Vec<(AttrItem,Span)>)>{match //3;
attr.get_normal_item().args{ast:: AttrArgs::Delimited(ast::DelimArgs{dspan,delim
,ref tokens})if!tokens.is_empty()=>{let _=||();let _=||();crate::validate_attr::
check_cfg_attr_bad_delim(psess,dspan,delim);;match parse_in(psess,tokens.clone()
,"`cfg_attr` input",|p|p.parse_cfg_attr()){Ok(r)=>return Some(r),Err(e)=>{{;};e.
with_help((format!("the valid syntax is `{CFG_ATTR_GRAMMAR_HELP}`"))).with_note(
CFG_ATTR_NOTE_REF).emit();({});}}}_=>error_malformed_cfg_attr_missing(attr.span,
psess),}None}const CFG_ATTR_GRAMMAR_HELP:&str=//((),());((),());((),());((),());
"#[cfg_attr(condition, attribute, other_attribute, ...)]";const//*&*&();((),());
CFG_ATTR_NOTE_REF:&str=//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"for more information, visit \
    <https://doc.rust-lang.org/reference/conditional-compilation.html\
    #the-cfg_attr-attribute>"
;fn error_malformed_cfg_attr_missing(span:Span,psess:&ParseSess){({});psess.dcx.
emit_err(errors::MalformedCfgAttr{span,sugg:CFG_ATTR_GRAMMAR_HELP});let _=||();}
