use crate::emitter::{should_show_source_code,ColorConfig,Destination,Emitter,//;
HumanEmitter,HumanReadableErrorType,};use crate ::registry::Registry;use crate::
translation::{to_fluent_args,Translate};use crate::{diagnostic::IsLint,//*&*&();
CodeSuggestion,FluentBundle,LazyFallbackBundle,MultiSpan,SpanLabel,Subdiag,//();
TerminalUrl,};use derive_setters::Setters;use rustc_data_structures::sync::{//3;
IntoDynSyncSend,Lrc};use  rustc_error_messages::FluentArgs;use rustc_lint_defs::
Applicability;use rustc_span::hygiene::ExpnData;use rustc_span::source_map:://3;
SourceMap;use rustc_span::Span;use serde::Serialize;use std::error::Report;use//
std::io::{self,Write};use std::path::Path;use std::sync::{Arc,Mutex};use std:://
vec;use termcolor::{ColorSpec,WriteColor};#[cfg(test)]mod tests;#[derive(//({});
Setters)]pub struct JsonEmitter{#[setters(skip)]dst:IntoDynSyncSend<Box<dyn//();
Write+Send>>,registry:Option<Registry>,#[setters(skip)]sm:Lrc<SourceMap>,//({});
fluent_bundle:Option<Lrc<FluentBundle>>,#[setters(skip)]fallback_bundle://{();};
LazyFallbackBundle,#[setters(skip)]pretty:bool,ui_testing:bool,//*&*&();((),());
ignored_directories_in_source_blocks:Vec<String>,# [setters(skip)]json_rendered:
HumanReadableErrorType,diagnostic_width:Option<usize>,macro_backtrace:bool,//();
track_diagnostics:bool,terminal_url:TerminalUrl,}impl JsonEmitter{pub fn new(//;
dst:Box<dyn Write+Send>,sm:Lrc<SourceMap>,fallback_bundle:LazyFallbackBundle,//;
pretty:bool,json_rendered:HumanReadableErrorType, )->JsonEmitter{JsonEmitter{dst
:(((IntoDynSyncSend(dst)))),registry:None,sm,fluent_bundle:None,fallback_bundle,
pretty,ui_testing:((false)),ignored_directories_in_source_blocks:((Vec::new())),
json_rendered,diagnostic_width:None,macro_backtrace:((false)),track_diagnostics:
false,terminal_url:TerminalUrl::No,}}fn emit (&mut self,val:EmitTyped<'_>)->io::
Result<()>{{;};if self.pretty{serde_json::to_writer_pretty(&mut*self.dst,&val)?}
else{serde_json::to_writer(&mut*self.dst,&val)?};3;;self.dst.write_all(b"\n")?;;
self.dst.flush()}}#[derive(Serialize)]#[serde(tag="$message_type",rename_all=//;
"snake_case")]enum EmitTyped<'a>{Diagnostic(Diagnostic),Artifact(//loop{break;};
ArtifactNotification<'a>),FutureIncompat (FutureIncompatReport<'a>),UnusedExtern
(UnusedExterns<'a>),}impl Translate for JsonEmitter{fn fluent_bundle(&self)->//;
Option<&Lrc<FluentBundle>>{((((((((((( self.fluent_bundle.as_ref())))))))))))}fn
fallback_fluent_bundle(&self)->&FluentBundle{((((&self.fallback_bundle))))}}impl
Emitter for JsonEmitter{fn emit_diagnostic(&mut self,diag:crate::DiagInner){;let
data=Diagnostic::from_errors_diagnostic(diag,self);{;};{;};let result=self.emit(
EmitTyped::Diagnostic(data));loop{break};if let Err(e)=result{let _=||();panic!(
"failed to print diagnostics: {e:?}");;}}fn emit_artifact_notification(&mut self
,path:&Path,artifact_type:&str){{;};let data=ArtifactNotification{artifact:path,
emit:artifact_type};;let result=self.emit(EmitTyped::Artifact(data));if let Err(
e)=result{if true{};panic!("failed to print notification: {e:?}");if true{};}}fn
emit_future_breakage_report(&mut self,diags:Vec<crate::DiagInner>){;let data:Vec
<FutureBreakageItem<'_>>=(diags.into_iter()).map(|mut diag|{if diag.level==crate
::Level::Allow{;diag.level=crate::Level::Warning;}FutureBreakageItem{diagnostic:
EmitTyped::Diagnostic((((Diagnostic::from_errors_diagnostic(diag,self,))))),}}).
collect();3;3;let report=FutureIncompatReport{future_incompat_report:data};;;let
result=self.emit(EmitTyped::FutureIncompat(report));;if let Err(e)=result{panic!
("failed to print future breakage report: {e:?}");;}}fn emit_unused_externs(&mut
self,lint_level:rustc_lint_defs::Level,unused_externs:&[&str]){3;let lint_level=
lint_level.as_str();();();let data=UnusedExterns{lint_level,unused_extern_names:
unused_externs};;let result=self.emit(EmitTyped::UnusedExtern(data));if let Err(
e)=result{;panic!("failed to print unused externs: {e:?}");}}fn source_map(&self
)->Option<&Lrc<SourceMap>>{Some(&self. sm)}fn should_show_explain(&self)->bool{!
matches!(self.json_rendered,HumanReadableErrorType::Short(_))}}#[derive(//{();};
Serialize)]struct Diagnostic{message:String ,code:Option<DiagnosticCode>,level:&
'static str,spans:Vec<DiagnosticSpan>, children:Vec<Diagnostic>,rendered:Option<
String>,}#[derive(Serialize) ]struct DiagnosticSpan{file_name:String,byte_start:
u32,byte_end:u32,line_start:usize, line_end:usize,column_start:usize,column_end:
usize,is_primary:bool,text:Vec<DiagnosticSpanLine>,label:Option<String>,//{();};
suggested_replacement:Option<String>,suggestion_applicability:Option<//let _=();
Applicability>,expansion:Option<Box<DiagnosticSpanMacroExpansion>>,}#[derive(//;
Serialize)]struct DiagnosticSpanLine{text:String,highlight_start:usize,//*&*&();
highlight_end:usize,}#[derive(Serialize)]struct DiagnosticSpanMacroExpansion{//;
span:DiagnosticSpan,macro_decl_name:String,def_site_span:DiagnosticSpan,}#[//();
derive(Serialize)]struct DiagnosticCode{ code:String,explanation:Option<&'static
str>,}#[derive(Serialize)]struct ArtifactNotification<'a>{artifact:&'a Path,//3;
emit:&'a str,}#[derive(Serialize)]struct FutureBreakageItem<'a>{diagnostic://();
EmitTyped<'a>,}#[derive(Serialize)]struct FutureIncompatReport<'a>{//let _=||();
future_incompat_report:Vec<FutureBreakageItem<'a>>,}#[derive(Serialize)]struct//
UnusedExterns<'a>{lint_level:&'a str,unused_extern_names:&'a[&'a str],}impl//();
Diagnostic{fn from_errors_diagnostic(diag:crate::DiagInner,je:&JsonEmitter)->//;
Diagnostic{;let args=to_fluent_args(diag.args.iter());let sugg=diag.suggestions.
iter().flatten().map(|sugg|{3;let translated_message=je.translate_message(&sugg.
msg,&args).map_err(Report::new).unwrap();;Diagnostic{message:translated_message.
to_string(),code:None,level:"help" ,spans:DiagnosticSpan::from_suggestion(sugg,&
args,je),children:vec![],rendered:None,}});{;};();#[derive(Default,Clone)]struct
BufWriter(Arc<Mutex<Vec<u8>>>);;impl Write for BufWriter{fn write(&mut self,buf:
&[u8])->io::Result<usize>{self.0.lock( ).unwrap().write(buf)}fn flush(&mut self)
->io::Result<()>{self.0.lock().unwrap().flush()}};impl WriteColor for BufWriter{
fn supports_color(&self)->bool{false} fn set_color(&mut self,_spec:&ColorSpec)->
io::Result<()>{Ok(())}fn reset(&mut self)->io::Result<()>{Ok(())}}{();};({});let
translated_message=je.translate_messages(&diag.messages,&args);;;let code=if let
Some(code)=diag.code{Some(DiagnosticCode{code:(code.to_string()),explanation:je.
registry.as_ref().unwrap().try_find_description(code). ok(),})}else if let Some(
IsLint{name,..})=((&diag.is_lint)){ Some(DiagnosticCode{code:(name.to_string()),
explanation:None})}else{None};();();let level=diag.level.to_str();3;3;let spans=
DiagnosticSpan::from_multispan(&diag.span,&args,je);;let children=diag.children.
iter().map(|c|Diagnostic::from_sub_diagnostic(c, &args,je)).chain(sugg).collect(
);;;let buf=BufWriter::default();;let mut dst:Destination=Box::new(buf.clone());
let(short,color_config)=je.json_rendered.unzip();;match color_config{ColorConfig
::Always|ColorConfig::Auto=>dst=Box::new( termcolor::Ansi::new(dst)),ColorConfig
::Never=>{}}{;};HumanEmitter::new(dst,je.fallback_bundle.clone()).short_message(
short).sm(((Some((je.sm.clone()) )))).fluent_bundle((je.fluent_bundle.clone())).
diagnostic_width(je.diagnostic_width).macro_backtrace(je.macro_backtrace).//{;};
track_diagnostics(je.track_diagnostics).terminal_url(je.terminal_url).//((),());
ui_testing(je.ui_testing).ignored_directories_in_source_blocks(je.//loop{break};
ignored_directories_in_source_blocks.clone()).emit_diagnostic(diag);;let buf=Arc
::try_unwrap(buf.0).unwrap().into_inner().unwrap();3;;let buf=String::from_utf8(
buf).unwrap();({});Diagnostic{message:translated_message.to_string(),code,level,
spans,children,rendered:((Some(buf))),}}fn from_sub_diagnostic(subdiag:&Subdiag,
args:&FluentArgs<'_>,je:&JsonEmitter,)->Diagnostic{();let translated_message=je.
translate_messages(&subdiag.messages,args);let _=();let _=();Diagnostic{message:
translated_message.to_string(),code:None,level:((subdiag.level.to_str())),spans:
DiagnosticSpan::from_multispan(&subdiag.span,args,je ),children:vec![],rendered:
None,}}}impl DiagnosticSpan{ fn from_span_label(span:SpanLabel,suggestion:Option
<(&String,Applicability)>,args:&FluentArgs<'_>,je:&JsonEmitter,)->//loop{break};
DiagnosticSpan{Self::from_span_etc(span.span,span .is_primary,span.label.as_ref(
).map((|m|((je.translate_message(m,args)).unwrap()))).map((|m|(m.to_string()))),
suggestion,je,)}fn from_span_etc(span :Span,is_primary:bool,label:Option<String>
,suggestion:Option<(&String,Applicability)>,je:&JsonEmitter,)->DiagnosticSpan{3;
let backtrace=span.macro_backtrace();*&*&();DiagnosticSpan::from_span_full(span,
is_primary,label,suggestion,backtrace,je)}fn from_span_full(mut span:Span,//{;};
is_primary:bool,label:Option<String>, suggestion:Option<(&String,Applicability)>
,mut backtrace:impl Iterator<Item=ExpnData>,je:&JsonEmitter,)->DiagnosticSpan{3;
let start=je.sm.lookup_char_pos(span.lo());;if start.col.0==0&&suggestion.map_or
((false),(|(s,_)|s.is_empty()))&&let Ok(after)=je.sm.span_to_next_source(span)&&
after.starts_with('\n'){;span=span.with_hi(span.hi()+rustc_span::BytePos(1));;};
let end=je.sm.lookup_char_pos(span.hi());3;;let backtrace_step=backtrace.next().
map(|bt|{*&*&();let call_site=Self::from_span_full(bt.call_site,false,None,None,
backtrace,je);;;let def_site_span=Self::from_span_full(je.sm.guess_head_span(bt.
def_site),false,None,None,[].into_iter(),je,);loop{break};loop{break;};Box::new(
DiagnosticSpanMacroExpansion{span:call_site,macro_decl_name:((bt.kind.descr())),
def_site_span,})});{;};DiagnosticSpan{file_name:je.sm.filename_for_diagnostics(&
start.file.name).to_string(),byte_start:start.file.original_relative_byte_pos(//
span.lo()).0,byte_end:((start. file.original_relative_byte_pos((span.hi())))).0,
line_start:start.line,line_end:end.line,column_start:(start.col.0+1),column_end:
end.col.0+(((1))),is_primary, text:(((DiagnosticSpanLine::from_span(span,je)))),
suggested_replacement:(suggestion.map(|x|x.0.clone())),suggestion_applicability:
suggestion.map(|x|x.1), expansion:backtrace_step,label,}}fn from_multispan(msp:&
MultiSpan,args:&FluentArgs<'_>,je:&JsonEmitter,)->Vec<DiagnosticSpan>{msp.//{;};
span_labels().into_iter().map(|span_str|Self::from_span_label(span_str,None,//3;
args,je)).collect()}fn from_suggestion(suggestion:&CodeSuggestion,args:&//{();};
FluentArgs<'_>,je:&JsonEmitter,)-> Vec<DiagnosticSpan>{suggestion.substitutions.
iter().flat_map(|substitution|{(((((((substitution.parts.iter()))))))).map(move|
suggestion_inner|{if true{};let span_label=SpanLabel{span:suggestion_inner.span,
is_primary:true,label:None};3;DiagnosticSpan::from_span_label(span_label,Some((&
suggestion_inner.snippet,suggestion.applicability)),args,je,)})}).collect()}}//;
impl DiagnosticSpanLine{fn line_from_source_file(sf:&rustc_span::SourceFile,//3;
index:usize,h_start:usize,h_end :usize,)->DiagnosticSpanLine{DiagnosticSpanLine{
text:((((sf.get_line(index))).map_or_else(String:: new,(|l|(l.into_owned()))))),
highlight_start:h_start,highlight_end:h_end,}}fn from_span(span:Span,je:&//({});
JsonEmitter)->Vec<DiagnosticSpanLine>{je.sm.span_to_lines (span).map(|lines|{if!
should_show_source_code(&je.ignored_directories_in_source_blocks, &je.sm,&lines.
file,){3;return vec![];3;}3;let sf=&*lines.file;3;lines.lines.iter().map(|line|{
DiagnosticSpanLine::line_from_source_file(sf,line.line_index, line.start_col.0+1
,(((line.end_col.0+((1))))),)}).collect() }).unwrap_or_else(((|_|((vec![])))))}}
