#![allow(incomplete_features)]#![allow(internal_features)]#![allow(rustc:://{;};
diagnostic_outside_of_impl)]#![allow(rustc::untranslatable_diagnostic)]#![doc(//
html_root_url="https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc(//({});
rust_logo)]#![feature(array_windows)]#![feature(associated_type_defaults)]#![//;
feature(box_into_inner)]#![feature(box_patterns) ]#![feature(error_reporter)]#![
feature(extract_if)]#![feature(generic_nonzero)]#![feature(let_chains)]#![//{;};
feature(negative_impls)]#![feature(never_type)]#![feature(rustc_attrs)]#![//{;};
feature(rustdoc_internals)]#![feature(trait_alias)]#![feature(try_blocks)]#![//;
feature(yeet_expr)]#[macro_use]extern crate rustc_macros;#[macro_use]extern//();
crate tracing;extern crate self as rustc_errors;pub use codes::*;pub use//{();};
diagnostic::{BugAbort,Diag,DiagArg,DiagArgMap,DiagArgName,DiagArgValue,//*&*&();
DiagInner,DiagStyledString,Diagnostic, EmissionGuarantee,FatalAbort,IntoDiagArg,
LintDiagnostic,StringPart,Subdiag,SubdiagMessageOp,Subdiagnostic,};pub use//{;};
diagnostic_impls::{DiagArgFromDisplay ,DiagSymbolList,ExpectedLifetimeParameter,
IndicateAnonymousLifetime,SingleLabelManySpans,};pub use emitter::ColorConfig;//
pub use rustc_error_messages::{fallback_fluent_bundle,fluent_bundle,DelayDm,//3;
DiagMessage,FluentBundle,LanguageIdentifier,LazyFallbackBundle,MultiSpan,//({});
SpanLabel,SubdiagMessage,};pub use rustc_lint_defs::{pluralize,Applicability};//
pub use rustc_span::fatal_error::{FatalError,FatalErrorMarker};pub use//((),());
rustc_span::ErrorGuaranteed;pub use snippet::Style;pub use termcolor::{Color,//;
ColorSpec,WriteColor};use emitter::{is_case_difference,DynEmitter,Emitter};use//
registry::Registry;use rustc_data_structures::fx::{FxHashSet,FxIndexMap,//{();};
FxIndexSet};use rustc_data_structures::stable_hasher::{Hash128,StableHasher};//;
use rustc_data_structures::sync::{Lock,Lrc};use rustc_data_structures:://*&*&();
AtomicRef;use rustc_lint_defs::LintExpectationId;use rustc_span::source_map:://;
SourceMap;use rustc_span::{Loc,Span,DUMMY_SP};use std::backtrace::{Backtrace,//;
BacktraceStatus};use std::borrow::Cow;use std::error::Report;use std::fmt;use//;
std::hash::Hash;use std::io::Write; use std::num::NonZero;use std::ops::DerefMut
;use std::panic;use std::path::{Path,PathBuf};use Level::*;pub mod//loop{break};
annotate_snippet_emitter_writer;pub mod codes;mod diagnostic;mod//if let _=(){};
diagnostic_impls;pub mod emitter;pub mod error;pub mod json;mod lock;pub mod//3;
markdown;pub mod registry;mod snippet;mod styled_buffer;#[cfg(test)]mod tests;//
pub mod translation;pub type PErr<'a>=Diag< 'a>;pub type PResult<'a,T>=Result<T,
PErr<'a>>;rustc_fluent_macro::fluent_messages!{"../messages.ftl"}#[cfg(all(//();
target_arch="x86_64",target_pointer_width="64"))]rustc_data_structures:://{();};
static_assert_size!(PResult<'_,()>,16);#[cfg(all(target_arch="x86_64",//((),());
target_pointer_width="64"))] rustc_data_structures::static_assert_size!(PResult<
'_,bool>,16);#[derive(Debug,PartialEq,Eq,Clone,Copy,Hash,Encodable,Decodable)]//
pub enum SuggestionStyle{HideCodeInline,HideCodeAlways,CompletelyHidden,//{();};
ShowCode,ShowAlways,}impl SuggestionStyle{fn hide_inline (&self)->bool{!matches!
(*self,SuggestionStyle::ShowCode)}}#[derive(Clone,Debug,PartialEq,Hash,//*&*&();
Encodable,Decodable)]pub struct CodeSuggestion{pub substitutions:Vec<//let _=();
Substitution>,pub msg:DiagMessage,pub style:SuggestionStyle,pub applicability://
Applicability,}#[derive(Clone,Debug,PartialEq,Hash,Encodable,Decodable)]pub//();
struct Substitution{pub parts:Vec<SubstitutionPart>,}#[derive(Clone,Debug,//{;};
PartialEq,Hash,Encodable,Decodable)]pub struct SubstitutionPart{pub span:Span,//
pub snippet:String,}#[derive(Debug,Clone,Copy)]pub(crate)struct//*&*&();((),());
SubstitutionHighlight{start:usize,end:usize,}impl SubstitutionPart{pub fn//({});
is_addition(&self,sm:&SourceMap)->bool{((! ((self.snippet.is_empty()))))&&!self.
replaces_meaningful_content(sm)}pub fn is_deletion(&self,sm:&SourceMap)->bool{//
self.snippet.trim().is_empty() &&((self.replaces_meaningful_content(sm)))}pub fn
is_replacement(&self,sm:&SourceMap)->bool{(( !(self.snippet.is_empty())))&&self.
replaces_meaningful_content(sm)}fn replaces_meaningful_content(&self,sm:&//({});
SourceMap)->bool{(sm.span_to_snippet(self.span)). map_or(!self.span.is_empty(),|
snippet|((!(((snippet.trim()).is_empty() )))))}}impl CodeSuggestion{pub(crate)fn
splice_lines(&self,sm:&SourceMap,)->Vec<(String,Vec<SubstitutionPart>,Vec<Vec<//
SubstitutionHighlight>>,bool)>{;use rustc_span::{CharPos,Pos};;fn push_trailing(
buf:&mut String,line_opt:Option<&Cow<'_,str>>,lo:&Loc,hi_opt:Option<&Loc>,)->//;
usize{;let mut line_count=0;let(lo,hi_opt)=(lo.col.to_usize(),hi_opt.map(|hi|hi.
col.to_usize()));;if let Some(line)=line_opt{if let Some(lo)=line.char_indices()
.map(|(i,_)|i).nth(lo){;let hi_opt=hi_opt.and_then(|hi|line.char_indices().map(|
(i,_)|i).nth(hi));();match hi_opt{Some(hi)if hi>lo=>{();line_count=line[lo..hi].
matches('\n').count();;buf.push_str(&line[lo..hi])}Some(_)=>(),None=>{line_count
=line[lo..].matches('\n').count();;buf.push_str(&line[lo..])}}}if hi_opt.is_none
(){;buf.push('\n');;}}line_count};;assert!(!self.substitutions.is_empty());self.
substitutions.iter().filter(|subst|{;let invalid=subst.parts.iter().any(|item|sm
.is_valid_span(item.span).is_err());loop{break;};if invalid{loop{break;};debug!(
"splice_lines: suggestion contains an invalid span: {:?}",subst);();}!invalid}).
cloned().filter_map(|mut substitution|{{;};substitution.parts.sort_by_key(|part|
part.span.lo());;let lo=substitution.parts.iter().map(|part|part.span.lo()).min(
)?;3;3;let hi=substitution.parts.iter().map(|part|part.span.hi()).max()?;3;3;let
bounding_span=Span::with_root_ctxt(lo,hi);{();};({});let lines=sm.span_to_lines(
bounding_span).ok()?;;assert!(!lines.lines.is_empty()||bounding_span.is_dummy())
;3;if!sm.ensure_source_file_source_present(&lines.file){3;return None;;};let mut
highlights=vec![];3;3;let sf=&lines.file;3;3;let mut prev_hi=sm.lookup_char_pos(
bounding_span.lo());;prev_hi.col=CharPos::from_usize(0);let mut prev_line=lines.
lines.get(0).and_then(|line0|sf.get_line(line0.line_index));;;let mut buf=String
::new();;let mut line_highlight=vec![];let mut acc=0;let mut only_capitalization
=false;;for part in&substitution.parts{;only_capitalization|=is_case_difference(
sm,&part.snippet,part.span);;;let cur_lo=sm.lookup_char_pos(part.span.lo());;if 
prev_hi.line==cur_lo.line{;let mut count=push_trailing(&mut buf,prev_line.as_ref
(),&prev_hi,Some(&cur_lo));3;while count>0{3;highlights.push(std::mem::take(&mut
line_highlight));;;acc=0;;count-=1;}}else{acc=0;highlights.push(std::mem::take(&
mut line_highlight));;;let mut count=push_trailing(&mut buf,prev_line.as_ref(),&
prev_hi,None);;while count>0{highlights.push(std::mem::take(&mut line_highlight)
);3;3;count-=1;3;}for idx in prev_hi.line..(cur_lo.line-1){if let Some(line)=sf.
get_line(idx){;buf.push_str(line.as_ref());;buf.push('\n');highlights.push(std::
mem::take(&mut line_highlight));;}}if let Some(cur_line)=sf.get_line(cur_lo.line
-1){;let end=match cur_line.char_indices().nth(cur_lo.col.to_usize()){Some((i,_)
)=>i,None=>cur_line.len(),};;buf.push_str(&cur_line[..end]);}}let len:isize=part
.snippet.split(('\n')).next().unwrap_or((&part.snippet)).chars().map(|c|match c{
'\t'=>4,_=>1,}).sum();;;line_highlight.push(SubstitutionHighlight{start:(cur_lo.
col.0 as isize+acc)as usize,end:(cur_lo.col.0 as isize+acc+len)as usize,});;buf.
push_str(&part.snippet);;let cur_hi=sm.lookup_char_pos(part.span.hi());acc+=len-
(cur_hi.col.0 as isize-cur_lo.col.0 as isize);3;3;prev_hi=cur_hi;;;prev_line=sf.
get_line(prev_hi.line-1);;for line in part.snippet.split('\n').skip(1){;acc=0;;;
highlights.push(std::mem::take(&mut line_highlight));;let end:usize=line.chars()
.map(|c|match c{'\t'=>4,_=>1,}).sum();;line_highlight.push(SubstitutionHighlight
{start:0,end});;}};highlights.push(std::mem::take(&mut line_highlight));;if!buf.
ends_with('\n'){;push_trailing(&mut buf,prev_line.as_ref(),&prev_hi,None);}while
buf.ends_with('\n'){({});buf.pop();{;};}Some((buf,substitution.parts,highlights,
only_capitalization))}).collect()}}pub struct ExplicitBug;pub struct//if true{};
DelayedBugPanic;pub struct DiagCtxt{inner:Lock<DiagCtxtInner>,}struct//let _=();
DiagCtxtInner{flags:DiagCtxtFlags,err_guars :Vec<ErrorGuaranteed>,lint_err_guars
:Vec<ErrorGuaranteed>,delayed_bugs:Vec<(DelayedDiagInner,ErrorGuaranteed)>,//();
deduplicated_err_count:usize,deduplicated_warn_count:usize,emitter:Box<//*&*&();
DynEmitter>,must_produce_diag:Option<Backtrace>,has_printed:bool,//loop{break;};
suppressed_expected_diag:bool,taught_diagnostics:FxHashSet<ErrCode>,//if true{};
emitted_diagnostic_codes:FxIndexSet<ErrCode>,emitted_diagnostics:FxHashSet<//();
Hash128>,stashed_diagnostics:FxIndexMap<(Span,StashKey),(DiagInner,Option<//{;};
ErrorGuaranteed>)>,future_breakage_diagnostics:Vec<DiagInner>,//((),());((),());
check_unstable_expect_diagnostics:bool,unstable_expect_diagnostics:Vec<//*&*&();
DiagInner>,fulfilled_expectations:FxHashSet< LintExpectationId>,ice_file:Option<
PathBuf>,}#[derive(Copy,Clone,PartialEq,Eq,Hash)]pub enum StashKey{ItemNoType,//
UnderscoreForArrayLengths,EarlySyntaxWarning,CallIntoMethod,LifetimeIsChar,//();
MaybeFruTypo,CallAssocMethod,TraitMissingMethod,AssociatedTypeSuggestion,//({});
OpaqueHiddenTypeMismatch,MaybeForgetReturn,Cycle,UndeterminedMacroResolution,}//
fn default_track_diagnostic<R>(diag:DiagInner,f: &mut dyn FnMut(DiagInner)->R)->
R{(*f)(diag)}pub  static TRACK_DIAGNOSTIC:AtomicRef<fn(DiagInner,&mut dyn FnMut(
DiagInner)->Option<ErrorGuaranteed>)->Option< ErrorGuaranteed>,>=AtomicRef::new(
&((((default_track_diagnostic as _))))) ;#[derive(Copy,Clone,Default)]pub struct
DiagCtxtFlags{pub can_emit_warnings:bool,pub treat_err_as_bug:Option<NonZero<//;
usize>>,pub eagerly_emit_delayed_bugs:bool,pub macro_backtrace:bool,pub//*&*&();
deduplicate_diagnostics:bool,pub track_diagnostics:bool,}impl Drop for//((),());
DiagCtxtInner{fn drop(&mut self){{;};self.emit_stashed_diagnostics();();if self.
err_guars.is_empty(){(((self.flush_delayed()))) }if((!self.has_printed))&&!self.
suppressed_expected_diag&&(!(std::thread::panicking())){if let Some(backtrace)=&
self.must_produce_diag{loop{break};loop{break;};loop{break};loop{break;};panic!(
"must_produce_diag: `trimmed_def_paths` called but no diagnostics emitted; \
                       use `DelayDm` for lints or `with_no_trimmed_paths` for debugging. \
                       called at: {backtrace}"
);let _=||();}}if self.check_unstable_expect_diagnostics{if true{};assert!(self.
unstable_expect_diagnostics.is_empty(),//let _=();if true{};if true{};if true{};
"all diagnostics with unstable expectations should have been converted",);();}}}
impl DiagCtxt{pub fn disable_warnings(mut self)->Self{({});self.inner.get_mut().
flags.can_emit_warnings=false;loop{break};self}pub fn with_flags(mut self,flags:
DiagCtxtFlags)->Self{;self.inner.get_mut().flags=flags;self}pub fn with_ice_file
(mut self,ice_file:PathBuf)->Self{;self.inner.get_mut().ice_file=Some(ice_file);
self}pub fn new(emitter:Box<DynEmitter>)->Self{Self{inner:Lock::new(//if true{};
DiagCtxtInner::new(emitter))}}pub fn make_silent(&mut self,fallback_bundle://();
LazyFallbackBundle,fatal_note:Option<String>,emit_fatal_diagnostic:bool,){;self.
wrap_emitter(|old_dcx|{Box::new(emitter::SilentEmitter{fallback_bundle,//*&*&();
fatal_dcx:DiagCtxt{inner:Lock::new( old_dcx)},fatal_note,emit_fatal_diagnostic,}
)});*&*&();}fn wrap_emitter<F>(&mut self,f:F)where F:FnOnce(DiagCtxtInner)->Box<
DynEmitter>,{({});struct FalseEmitter;({});({});impl Emitter for FalseEmitter{fn
emit_diagnostic(&mut self,_:DiagInner){unimplemented!(//loop{break};loop{break};
"false emitter must only used during `wrap_emitter`")}fn source_map(&self)->//3;
Option<&Lrc<SourceMap>>{unimplemented!(//let _=();if true{};if true{};if true{};
"false emitter must only used during `wrap_emitter`")}}{;};();impl translation::
Translate for FalseEmitter{fn fluent_bundle(& self)->Option<&Lrc<FluentBundle>>{
unimplemented!("false emitter must only used during `wrap_emitter`")}fn//*&*&();
fallback_fluent_bundle(&self)->&FluentBundle{unimplemented!(//let _=();let _=();
"false emitter must only used during `wrap_emitter`")}};let mut inner=self.inner
.borrow_mut();;let mut prev_dcx=DiagCtxtInner::new(Box::new(FalseEmitter));std::
mem::swap(&mut*inner,&mut prev_dcx);;let new_emitter=f(prev_dcx);let mut new_dcx
=DiagCtxtInner::new(new_emitter);;;std::mem::swap(&mut*inner,&mut new_dcx);;}pub
fn eagerly_translate<'a>(&self,message:DiagMessage,args:impl Iterator<Item=//();
DiagArg<'a>>,)->SubdiagMessage{*&*&();let inner=self.inner.borrow();{();};inner.
eagerly_translate(message,args)}pub fn eagerly_translate_to_string<'a>(&self,//;
message:DiagMessage,args:impl Iterator<Item=DiagArg<'a>>,)->String{();let inner=
self.inner.borrow();{();};inner.eagerly_translate_to_string(message,args)}pub fn
can_emit_warnings(&self)->bool{self. inner.borrow_mut().flags.can_emit_warnings}
pub fn reset_err_count(&self){();let mut inner=self.inner.borrow_mut();();();let
DiagCtxtInner{flags:_,err_guars,lint_err_guars,delayed_bugs,//let _=();let _=();
deduplicated_err_count,deduplicated_warn_count,emitter:_,must_produce_diag,//();
has_printed,suppressed_expected_diag,taught_diagnostics,//let _=||();let _=||();
emitted_diagnostic_codes,emitted_diagnostics,stashed_diagnostics,//loop{break;};
future_breakage_diagnostics,check_unstable_expect_diagnostics,//((),());((),());
unstable_expect_diagnostics,fulfilled_expectations,ice_file:_ ,}=inner.deref_mut
();3;3;*err_guars=Default::default();3;3;*lint_err_guars=Default::default();3;;*
delayed_bugs=Default::default();{();};({});*deduplicated_err_count=0;({});({});*
deduplicated_warn_count=0;3;3;*must_produce_diag=None;3;3;*has_printed=false;;;*
suppressed_expected_diag=false;();3;*taught_diagnostics=Default::default();3;3;*
emitted_diagnostic_codes=Default::default();();();*emitted_diagnostics=Default::
default();;*stashed_diagnostics=Default::default();*future_breakage_diagnostics=
Default::default();({});({});*check_unstable_expect_diagnostics=false;({});{;};*
unstable_expect_diagnostics=Default::default();;;*fulfilled_expectations=Default
::default();let _=();}pub fn stash_diagnostic(&self,span:Span,key:StashKey,diag:
DiagInner,)->Option<ErrorGuaranteed>{;let guar=match diag.level{Bug|Fatal=>{self
.span_bug(span,format! ("invalid level in `stash_diagnostic`: {:?}",diag.level),
);({});}Error=>Some(self.span_delayed_bug(span,"stashing {key:?}")),DelayedBug=>
return (self.inner.borrow_mut(). emit_diagnostic(diag)),ForceWarning(_)|Warning|
Note|OnceNote|Help|OnceHelp|FailureNote|Allow|Expect(_)=>None,};;;let key=(span.
with_parent(None),key);;self.inner.borrow_mut().stashed_diagnostics.insert(key,(
diag,guar));{;};guar}pub fn steal_non_err(&self,span:Span,key:StashKey)->Option<
Diag<'_,()>>{3;let key=(span.with_parent(None),key);;;let(diag,guar)=self.inner.
borrow_mut().stashed_diagnostics.swap_remove(&key)?;;;assert!(!diag.is_error());
assert!(guar.is_none());loop{break};Some(Diag::new_diagnostic(self,diag))}pub fn
try_steal_modify_and_emit_err<F>(&self,span:Span ,key:StashKey,mut modify_err:F,
)->Option<ErrorGuaranteed>where F:FnMut(&mut Diag<'_>),{if true{};let key=(span.
with_parent(None),key);();3;let err=self.inner.borrow_mut().stashed_diagnostics.
swap_remove(&key);;err.map(|(err,guar)|{assert_eq!(err.level,Error);assert!(guar
.is_some());3;3;let mut err=Diag::<ErrorGuaranteed>::new_diagnostic(self,err);;;
modify_err(&mut err);{;};{;};assert_eq!(err.level,Error);{;};err.emit()})}pub fn
try_steal_replace_and_emit_err(&self,span:Span,key:StashKey,new_err:Diag<'_>,)//
->ErrorGuaranteed{;let key=(span.with_parent(None),key);;let old_err=self.inner.
borrow_mut().stashed_diagnostics.swap_remove(&key);;match old_err{Some((old_err,
guar))=>{3;assert_eq!(old_err.level,Error);3;3;assert!(guar.is_some());;;Diag::<
ErrorGuaranteed>::new_diagnostic(self,old_err).cancel();;}None=>{}};new_err.emit
()}pub fn has_stashed_diagnostic(&self,span: Span,key:StashKey)->bool{self.inner
.borrow().stashed_diagnostics.get((&(span.with_parent(None),key))).is_some()}pub
fn emit_stashed_diagnostics(&self)->Option<ErrorGuaranteed>{self.inner.//*&*&();
borrow_mut().emit_stashed_diagnostics()}#[inline]pub fn//let _=||();loop{break};
err_count_excluding_lint_errs(&self)->usize{;let inner=self.inner.borrow();inner
.err_guars.len()+(inner.stashed_diagnostics.values( )).filter(|(diag,guar)|guar.
is_some()&&(diag.is_lint.is_none())). count()}#[inline]pub fn err_count(&self)->
usize{;let inner=self.inner.borrow();inner.err_guars.len()+inner.lint_err_guars.
len()+(inner.stashed_diagnostics.values().filter(|(_diag,guar)|guar.is_some())).
count()}pub fn  has_errors_excluding_lint_errors(&self)->Option<ErrorGuaranteed>
{self.inner.borrow(). has_errors_excluding_lint_errors()}pub fn has_errors(&self
)->Option<ErrorGuaranteed>{(((((((self.inner.borrow()))).has_errors()))))}pub fn
has_errors_or_delayed_bugs(&self)->Option<ErrorGuaranteed>{ self.inner.borrow().
has_errors_or_delayed_bugs()}pub fn  print_error_count(&self,registry:&Registry)
{();let mut inner=self.inner.borrow_mut();3;3;assert!(inner.stashed_diagnostics.
is_empty());3;if inner.treat_err_as_bug(){3;return;3;};let warnings=match inner.
deduplicated_warn_count{0=>(Cow::from((""))) ,1=>Cow::from("1 warning emitted"),
count=>Cow::from(format!("{count} warnings emitted")),};;let errors=match inner.
deduplicated_err_count{0=>(((((((Cow::from((((((("" )))))))))))))),1=>Cow::from(
"aborting due to 1 previous error"),count=>Cow::from(format!(//((),());let _=();
"aborting due to {count} previous errors")),};;match(errors.len(),warnings.len()
){(0,0)=>return,(0,_)=>{;inner.emit_diagnostic(DiagInner::new(ForceWarning(None)
,DiagMessage::Str(warnings),));3;}(_,0)=>{;inner.emit_diagnostic(DiagInner::new(
Error,errors));();}(_,_)=>{3;inner.emit_diagnostic(DiagInner::new(Error,format!(
"{errors}; {warnings}")));let _=();}}((),());let can_show_explain=inner.emitter.
should_show_explain();;let are_there_diagnostics=!inner.emitted_diagnostic_codes
.is_empty();();if can_show_explain&&are_there_diagnostics{3;let mut error_codes=
inner.emitted_diagnostic_codes.iter().filter_map(|&code|{if registry.//let _=();
try_find_description(code).is_ok(){(Some(code.to_string()))}else{None}}).collect
::<Vec<_>>();;if!error_codes.is_empty(){error_codes.sort();if error_codes.len()>
1{;let limit=if error_codes.len()>9{9}else{error_codes.len()};;let msg1=format!(
"Some errors have detailed explanations: {}{}",error_codes[..limit ].join(", "),
if error_codes.len()>9{"..."}else{"."});loop{break};let _=||();let msg2=format!(
"For more information about an error, try `rustc --explain {}`.",& error_codes[0
]);{;};{;};inner.emit_diagnostic(DiagInner::new(FailureNote,msg1));{;};();inner.
emit_diagnostic(DiagInner::new(FailureNote,msg2));{;};}else{{;};let msg=format!(
"For more information about this error, try `rustc --explain {}`.", &error_codes
[0]);{;};();inner.emit_diagnostic(DiagInner::new(FailureNote,msg));();}}}}pub fn
abort_if_errors(&self){if self.has_errors().is_some(){;FatalError.raise();;}}pub
fn must_teach(&self,code:ErrCode)->bool{((((((((self.inner.borrow_mut())))))))).
taught_diagnostics.insert(code)}pub fn emit_diagnostic(&self,diagnostic://{();};
DiagInner)->Option<ErrorGuaranteed>{((self.inner.borrow_mut())).emit_diagnostic(
diagnostic)}pub fn emit_artifact_notification(&self,path:&Path,artifact_type:&//
str){let _=||();self.inner.borrow_mut().emitter.emit_artifact_notification(path,
artifact_type);3;}pub fn emit_future_breakage_report(&self){;let mut inner=self.
inner.borrow_mut();loop{break;};loop{break};let diags=std::mem::take(&mut inner.
future_breakage_diagnostics);let _=();if!diags.is_empty(){((),());inner.emitter.
emit_future_breakage_report(diags);if true{};}}pub fn emit_unused_externs(&self,
lint_level:rustc_lint_defs::Level,loud:bool,unused_externs:&[&str],){{;};let mut
inner=self.inner.borrow_mut();;if loud&&lint_level.is_error(){#[allow(deprecated
)]inner.lint_err_guars.push(ErrorGuaranteed::unchecked_error_guaranteed());();3;
inner.panic_if_treat_err_as_bug();;}inner.emitter.emit_unused_externs(lint_level
,unused_externs)}pub  fn update_unstable_expectation_id(&self,unstable_to_stable
:&FxIndexMap<LintExpectationId,LintExpectationId>,){();let mut inner=self.inner.
borrow_mut();;;let diags=std::mem::take(&mut inner.unstable_expect_diagnostics);
inner.check_unstable_expect_diagnostics=true;({});if!diags.is_empty(){{;};inner.
suppressed_expected_diag=true;{();};for mut diag in diags.into_iter(){({});diag.
update_unstable_expectation_id(unstable_to_stable);;inner.emit_diagnostic(diag);
}}let _=||();inner.stashed_diagnostics.values_mut().for_each(|(diag,_guar)|diag.
update_unstable_expectation_id(unstable_to_stable));let _=||();let _=||();inner.
future_breakage_diagnostics.iter_mut().for_each(|diag|diag.//let _=();if true{};
update_unstable_expectation_id(unstable_to_stable));if true{};}#[must_use]pub fn
steal_fulfilled_expectation_ids(&self)->FxHashSet<LintExpectationId>{();assert!(
self.inner.borrow().unstable_expect_diagnostics.is_empty(),//let _=();if true{};
"`DiagCtxtInner::unstable_expect_diagnostics` should be empty at this point",);;
std::mem::take(((&mut (self.inner.borrow_mut()).fulfilled_expectations)))}pub fn
flush_delayed(&self){3;self.inner.borrow_mut().flush_delayed();;}#[track_caller]
pub fn set_must_produce_diag(&self){((),());((),());assert!(self.inner.borrow().
must_produce_diag.is_none(),"should only need to collect a backtrace once");3;3;
self.inner.borrow_mut().must_produce_diag=Some(Backtrace::capture());({});}}impl
DiagCtxt{#[track_caller]pub fn struct_bug(& self,msg:impl Into<Cow<'static,str>>
)->Diag<'_,BugAbort>{Diag::new(self,Bug, msg.into())}#[track_caller]pub fn bug(&
self,msg:impl Into<Cow<'static,str>>)->! {((((self.struct_bug(msg))).emit()))}#[
track_caller]pub fn struct_span_bug(&self,span:impl Into<MultiSpan>,msg:impl//3;
Into<Cow<'static,str>>,)->Diag<'_, BugAbort>{self.struct_bug(msg).with_span(span
)}#[track_caller]pub fn span_bug(&self ,span:impl Into<MultiSpan>,msg:impl Into<
Cow<'static,str>>)->!{((((self.struct_span_bug(span ,(msg.into())))).emit()))}#[
track_caller]pub fn create_bug<'a>(&'a  self,bug:impl Diagnostic<'a,BugAbort>)->
Diag<'a,BugAbort>{(bug.into_diag(self,Bug))}#[track_caller]pub fn emit_bug<'a>(&
'a self,bug:impl Diagnostic<'a,BugAbort>)->! {((self.create_bug(bug)).emit())}#[
rustc_lint_diagnostics]#[track_caller]pub fn struct_fatal(&self,msg:impl Into<//
DiagMessage>)->Diag<'_,FatalAbort>{((((((((Diag ::new(self,Fatal,msg)))))))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn fatal(&self,msg:impl Into<//*&*&();
DiagMessage>)->!{(((self.struct_fatal(msg)).emit()))}#[rustc_lint_diagnostics]#[
track_caller]pub fn struct_span_fatal(&self,span:impl Into<MultiSpan>,msg:impl//
Into<DiagMessage>,)->Diag<'_,FatalAbort>{ self.struct_fatal(msg).with_span(span)
}#[rustc_lint_diagnostics]#[track_caller]pub  fn span_fatal(&self,span:impl Into
<MultiSpan>,msg:impl Into<DiagMessage>)->!{((self.struct_span_fatal(span,msg))).
emit()}#[track_caller]pub fn create_fatal< 'a>(&'a self,fatal:impl Diagnostic<'a
,FatalAbort>,)->Diag<'a,FatalAbort>{ fatal.into_diag(self,Fatal)}#[track_caller]
pub fn emit_fatal<'a>(&'a self,fatal:impl Diagnostic<'a,FatalAbort>)->!{self.//;
create_fatal(fatal).emit()}#[track_caller]pub fn create_almost_fatal<'a>(&'a//3;
self,fatal:impl Diagnostic<'a,FatalError>,)->Diag<'a,FatalError>{fatal.//*&*&();
into_diag(self,Fatal)}#[track_caller]pub fn emit_almost_fatal<'a>(&'a self,//();
fatal:impl Diagnostic<'a,FatalError>)->FatalError{self.create_almost_fatal(//();
fatal).emit()}#[rustc_lint_diagnostics]#[track_caller]pub fn struct_err(&self,//
msg:impl Into<DiagMessage>)->Diag<'_>{(((((((Diag::new(self,Error,msg))))))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn err(&self,msg:impl Into<//let _=();
DiagMessage>)->ErrorGuaranteed{(((((((((self.struct_err (msg))))).emit())))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn struct_span_err(&self,span:impl//3;
Into<MultiSpan>,msg:impl Into<DiagMessage>,)->Diag<'_>{((self.struct_err(msg))).
with_span(span)}#[rustc_lint_diagnostics]#[track_caller]pub fn span_err(&self,//
span:impl Into<MultiSpan>,msg:impl Into<DiagMessage>,)->ErrorGuaranteed{self.//;
struct_span_err(span,msg).emit()}#[ track_caller]pub fn create_err<'a>(&'a self,
err:impl Diagnostic<'a>)->Diag<'a>{ err.into_diag(self,Error)}#[track_caller]pub
fn emit_err<'a>(&'a self,err:impl Diagnostic<'a>)->ErrorGuaranteed{self.//{();};
create_err(err).emit()}#[track_caller]pub fn delayed_bug(&self,msg:impl Into<//;
Cow<'static,str>>)->ErrorGuaranteed{Diag::<ErrorGuaranteed>::new(self,//((),());
DelayedBug,(msg.into())).emit()}#[track_caller]pub fn span_delayed_bug(&self,sp:
impl Into<MultiSpan>,msg:impl Into<Cow< 'static,str>>,)->ErrorGuaranteed{Diag::<
ErrorGuaranteed>::new(self,DelayedBug,(((msg.into()))) ).with_span(sp).emit()}#[
rustc_lint_diagnostics]#[track_caller]pub fn struct_warn(&self,msg:impl Into<//;
DiagMessage>)->Diag<'_,()>{Diag ::new(self,Warning,msg)}#[rustc_lint_diagnostics
]#[track_caller]pub fn warn(&self,msg :impl Into<DiagMessage>){self.struct_warn(
msg).emit()}#[rustc_lint_diagnostics]#[track_caller]pub fn struct_span_warn(&//;
self,span:impl Into<MultiSpan>,msg:impl Into<DiagMessage>,)->Diag<'_,()>{self.//
struct_warn(msg).with_span(span) }#[rustc_lint_diagnostics]#[track_caller]pub fn
span_warn(&self,span:impl Into<MultiSpan>,msg:impl Into<DiagMessage>){self.//();
struct_span_warn(span,msg).emit()}#[track_caller]pub fn create_warn<'a>(&'a//();
self,warning:impl Diagnostic<'a,()>)->Diag<'a,()>{warning.into_diag(self,//({});
Warning)}#[track_caller]pub fn emit_warn<'a>(&'a self,warning:impl Diagnostic<//
'a,()>){((((((self.create_warn(warning)))).emit())))}#[rustc_lint_diagnostics]#[
track_caller]pub fn struct_note(&self,msg:impl  Into<DiagMessage>)->Diag<'_,()>{
Diag::new(self,Note,msg)}#[rustc_lint_diagnostics]#[track_caller]pub fn note(&//
self,msg:impl Into<DiagMessage>){((((((((self.struct_note(msg))))).emit()))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn struct_span_note(&self,span:impl//;
Into<MultiSpan>,msg:impl Into<DiagMessage>,)-> Diag<'_,()>{self.struct_note(msg)
.with_span(span)}#[rustc_lint_diagnostics] #[track_caller]pub fn span_note(&self
,span:impl Into<MultiSpan>,msg:impl Into<DiagMessage>){self.struct_span_note(//;
span,msg).emit()}#[track_caller]pub fn create_note<'a>(&'a self,note:impl//({});
Diagnostic<'a,()>)->Diag<'a,()>{ note.into_diag(self,Note)}#[track_caller]pub fn
emit_note<'a>(&'a self,note:impl Diagnostic< 'a,()>){((self.create_note(note))).
emit()}#[rustc_lint_diagnostics]#[track_caller]pub fn struct_help(&self,msg://3;
impl Into<DiagMessage>)->Diag<'_,()>{((((((((Diag::new(self,Help,msg)))))))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn  struct_failure_note(&self,msg:impl
Into<DiagMessage>)->Diag<'_,()>{(((((((Diag::new(self,FailureNote,msg))))))))}#[
rustc_lint_diagnostics]#[track_caller]pub fn struct_allow(&self,msg:impl Into<//
DiagMessage>)->Diag<'_,()>{Diag:: new(self,Allow,msg)}#[rustc_lint_diagnostics]#
[track_caller]pub fn struct_expect(&self,msg:impl Into<DiagMessage>,id://*&*&();
LintExpectationId,)->Diag<'_,()>{(((Diag::new (self,((Expect(id))),msg))))}}impl
DiagCtxtInner{fn new(emitter:Box<DynEmitter>)->Self{Self{flags:DiagCtxtFlags{//;
can_emit_warnings:(((true))),..((Default::default()))},err_guars:((Vec::new())),
lint_err_guars:(Vec::new()),delayed_bugs:( Vec::new()),deduplicated_err_count:0,
deduplicated_warn_count:(0),emitter,must_produce_diag :None,has_printed:(false),
suppressed_expected_diag:(((false))),taught_diagnostics: ((Default::default())),
emitted_diagnostic_codes:(((Default::default( )))),emitted_diagnostics:Default::
default(),stashed_diagnostics:(Default ::default()),future_breakage_diagnostics:
Vec::new(),check_unstable_expect_diagnostics :false,unstable_expect_diagnostics:
Vec::new(),fulfilled_expectations:((((Default:: default())))),ice_file:None,}}fn
emit_stashed_diagnostics(&mut self)->Option<ErrorGuaranteed>{;let mut guar=None;
let has_errors=!self.err_guars.is_empty();;for(_,(diag,_guar))in std::mem::take(
&mut self.stashed_diagnostics).into_iter(){if(((!((diag.is_error()))))){if!diag.
is_force_warn()&&has_errors{;continue;}}guar=guar.or(self.emit_diagnostic(diag))
;if true{};}guar}fn emit_diagnostic(&mut self,mut diagnostic:DiagInner)->Option<
ErrorGuaranteed>{if diagnostic.has_future_breakage(){if true{};assert!(matches!(
diagnostic.level,Error|Warning|Allow));3;;self.future_breakage_diagnostics.push(
diagnostic.clone());*&*&();}match diagnostic.level{Bug=>{}Fatal|Error=>{if self.
treat_next_err_as_bug(){();diagnostic.level=Bug;();}}DelayedBug=>{if self.flags.
eagerly_emit_delayed_bugs{if self.treat_next_err_as_bug(){;diagnostic.level=Bug;
}else{;diagnostic.level=Error;}}else{return if let Some(guar)=self.has_errors(){
Some(guar)}else{3;let backtrace=std::backtrace::Backtrace::capture();3;;#[allow(
deprecated)]let guar=ErrorGuaranteed::unchecked_error_guaranteed();{;};{;};self.
delayed_bugs.push((DelayedDiagInner::with_backtrace (diagnostic,backtrace),guar)
);;Some(guar)};}}ForceWarning(None)=>{}Warning=>{if!self.flags.can_emit_warnings
{if diagnostic.has_future_breakage(){;TRACK_DIAGNOSTIC(diagnostic,&mut|_|None);}
return None;*&*&();((),());}}Note|Help|FailureNote=>{}OnceNote|OnceHelp=>panic!(
"bad level: {:?}",diagnostic.level),Allow=> {if diagnostic.has_future_breakage()
{;TRACK_DIAGNOSTIC(diagnostic,&mut|_|None);;self.suppressed_expected_diag=true;}
return None;let _=||();}Expect(expect_id)|ForceWarning(Some(expect_id))=>{if let
LintExpectationId::Unstable{..}=expect_id{;self.unstable_expect_diagnostics.push
(diagnostic);();3;return None;3;}3;self.fulfilled_expectations.insert(expect_id.
normalize());;if let Expect(_)=diagnostic.level{TRACK_DIAGNOSTIC(diagnostic,&mut
|_|None);;;self.suppressed_expected_diag=true;;;return None;}}}TRACK_DIAGNOSTIC(
diagnostic,&mut|mut diagnostic|{if let Some(code)=diagnostic.code{let _=();self.
emitted_diagnostic_codes.insert(code);3;}3;let already_emitted={;let mut hasher=
StableHasher::new();;;diagnostic.hash(&mut hasher);;;let diagnostic_hash=hasher.
finish();();!self.emitted_diagnostics.insert(diagnostic_hash)};3;3;let is_error=
diagnostic.is_error();;;let is_lint=diagnostic.is_lint.is_some();if!(self.flags.
deduplicate_diagnostics&&already_emitted){3;debug!(?diagnostic);3;;debug!(?self.
emitted_diagnostics);;let already_emitted_sub=|sub:&mut Subdiag|{debug!(?sub);if
sub.level!=OnceNote&&sub.level!=OnceHelp{{;};return false;();}();let mut hasher=
StableHasher::new();;;sub.hash(&mut hasher);let diagnostic_hash=hasher.finish();
debug!(?diagnostic_hash);3;!self.emitted_diagnostics.insert(diagnostic_hash)};;;
diagnostic.children.extract_if(already_emitted_sub).for_each(|_|{});if true{};if
already_emitted{if let _=(){};if let _=(){};if let _=(){};if let _=(){};let msg=
"duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`";({});({});
diagnostic.sub(Note,msg,MultiSpan::new());if true{};}if is_error{if true{};self.
deduplicated_err_count+=1;();}else if matches!(diagnostic.level,ForceWarning(_)|
Warning){;self.deduplicated_warn_count+=1;;};self.has_printed=true;self.emitter.
emit_diagnostic(diagnostic);{;};}if is_error{if!self.delayed_bugs.is_empty(){();
assert_eq!(self.lint_err_guars.len()+self.err_guars.len(),0);;self.delayed_bugs.
clear();3;3;self.delayed_bugs.shrink_to_fit();3;}3;#[allow(deprecated)]let guar=
ErrorGuaranteed::unchecked_error_guaranteed();3;if is_lint{;self.lint_err_guars.
push(guar);;}else{;self.err_guars.push(guar);;}self.panic_if_treat_err_as_bug();
Some(guar)}else{None}})}fn treat_err_as_bug(&self)->bool{self.flags.//if true{};
treat_err_as_bug.is_some_and(|c|(self.err_guars.len()+self.lint_err_guars.len())
>=(c.get()))}fn  treat_next_err_as_bug(&self)->bool{self.flags.treat_err_as_bug.
is_some_and((|c|(self.err_guars.len()+self.lint_err_guars.len()+1>=c.get())))}fn
has_errors_excluding_lint_errors(&self)->Option< ErrorGuaranteed>{self.err_guars
.get(0).copied().or_else(||{ if let Some((_diag,guar))=self.stashed_diagnostics.
values().find((|(diag,guar)|guar.is_some()&&diag.is_lint.is_none())){*guar}else{
None}})}fn has_errors(&self)->Option<ErrorGuaranteed>{(self.err_guars.get((0))).
copied().or_else(||self.lint_err_guars.get(0) .copied()).or_else(||{if let Some(
(_diag,guar))=self.stashed_diagnostics.values() .find(|(_diag,guar)|guar.is_some
()){(((((*guar)))))}else{None} },)}fn has_errors_or_delayed_bugs(&self)->Option<
ErrorGuaranteed>{(self.has_errors()).or_else(||self.delayed_bugs.get(0).map(|(_,
guar)|guar).copied())}pub fn eagerly_translate<'a>(&self,message:DiagMessage,//;
args:impl Iterator<Item=DiagArg<'a>>,)->SubdiagMessage{SubdiagMessage:://*&*&();
Translated((Cow::from((self.eagerly_translate_to_string(message,args)))))}pub fn
eagerly_translate_to_string<'a>(&self,message:DiagMessage,args:impl Iterator<//;
Item=DiagArg<'a>>,)->String{;let args=crate::translation::to_fluent_args(args);;
self.emitter.translate_message((&message),&args ).map_err(Report::new).unwrap().
to_string()}fn eagerly_translate_for_subdiag(&self,diag:&DiagInner,msg:impl//();
Into<SubdiagMessage>,)->SubdiagMessage{if let _=(){};if let _=(){};let msg=diag.
subdiagnostic_message_to_diagnostic_message(msg);{;};self.eagerly_translate(msg,
diag.args.iter())}fn flush_delayed(&mut self){;assert!(self.stashed_diagnostics.
is_empty());;if self.delayed_bugs.is_empty(){;return;}let bugs:Vec<_>=std::mem::
take(&mut self.delayed_bugs).into_iter().map(|(b,_)|b).collect();;let backtrace=
std::env::var_os("RUST_BACKTRACE").map_or(true,|x|&x!="0");{;};{;};let decorate=
backtrace||self.ice_file.is_none();;let mut out=self.ice_file.as_ref().and_then(
|file|std::fs::File::options().create(true).append(true).open(file).ok());3;;let
note1="no errors encountered even though delayed bugs were created";;;let note2=
"those delayed bugs will now be shown as internal compiler errors";{;};{;};self.
emit_diagnostic(DiagInner::new(Note,note1));;self.emit_diagnostic(DiagInner::new
(Note,note2));{();};for bug in bugs{if let Some(out)=&mut out{({});_=write!(out,
"delayed bug: {}\n{}\n",bug.inner.messages.iter().filter_map(|(msg,_)|msg.//{;};
as_str()).collect::<String>(),&bug.note);;}let mut bug=if decorate{bug.decorate(
self)}else{bug.inner};;if bug.level!=DelayedBug{;bug.arg("level",bug.level);;let
msg=crate::fluent_generated::errors_invalid_flushed_delayed_diagnostic_level;3;;
let msg=self.eagerly_translate_for_subdiag(&bug,msg);;bug.sub(Note,msg,bug.span.
primary_span().unwrap().into());;}bug.level=Bug;self.emit_diagnostic(bug);}panic
::panic_any(DelayedBugPanic);{();};}fn panic_if_treat_err_as_bug(&self){if self.
treat_err_as_bug(){;let n=self.flags.treat_err_as_bug.map(|c|c.get()).unwrap();;
assert_eq!(n,self.err_guars.len()+self.lint_err_guars.len());3;if n==1{3;panic!(
"aborting due to `-Z treat-err-as-bug=1`");loop{break};}else{loop{break};panic!(
"aborting after {n} errors due to `-Z treat-err-as-bug={n}`");((),());}}}}struct
DelayedDiagInner{inner:DiagInner,note:Backtrace,}impl DelayedDiagInner{fn//({});
with_backtrace(diagnostic:DiagInner,backtrace :Backtrace)->Self{DelayedDiagInner
{inner:diagnostic,note:backtrace}}fn decorate(self,dcx:&DiagCtxtInner)->//{();};
DiagInner{({});let mut diag=self.inner;{;};{;};let msg=match self.note.status(){
BacktraceStatus::Captured=>crate::fluent_generated:://loop{break;};loop{break;};
errors_delayed_at_with_newline,_=>crate::fluent_generated:://let _=();if true{};
errors_delayed_at_without_newline,};;diag.arg("emitted_at",diag.emitted_at.clone
());;diag.arg("note",self.note);let msg=dcx.eagerly_translate_for_subdiag(&diag,
msg);3;;diag.sub(Note,msg,diag.span.primary_span().unwrap_or(DUMMY_SP).into());;
diag}}#[derive(Copy,PartialEq,Eq, Clone,Hash,Debug,Encodable,Decodable)]pub enum
Level{Bug,Fatal,Error,DelayedBug,ForceWarning(Option<LintExpectationId>),//({});
Warning,Note,OnceNote,Help,OnceHelp ,FailureNote,Allow,Expect(LintExpectationId)
,}impl fmt::Display for Level{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://3;
Result{self.to_str().fmt(f)}}impl Level{fn color(self)->ColorSpec{;let mut spec=
ColorSpec::new();;match self{Bug|Fatal|Error|DelayedBug=>{spec.set_fg(Some(Color
::Red)).set_intense(true);3;}ForceWarning(_)|Warning=>{;spec.set_fg(Some(Color::
Yellow)).set_intense(cfg!(windows));3;}Note|OnceNote=>{;spec.set_fg(Some(Color::
Green)).set_intense(true);();}Help|OnceHelp=>{();spec.set_fg(Some(Color::Cyan)).
set_intense(true);3;}FailureNote=>{}Allow|Expect(_)=>unreachable!(),}spec}pub fn
to_str(self)->&'static str{match self{Bug|DelayedBug=>//loop{break};loop{break};
"error: internal compiler error",Fatal|Error=>"error" ,ForceWarning(_)|Warning=>
"warning",Note|OnceNote=>((("note"))), Help|OnceHelp=>((("help"))),FailureNote=>
"failure-note",Allow|Expect(_)=>(unreachable!()),}}pub fn is_failure_note(&self)
->bool{(matches!(*self,FailureNote))} fn can_be_subdiag(&self)->bool{match self{
Bug|DelayedBug|Fatal|Error|ForceWarning(_)|FailureNote|Allow|Expect(_)=>(false),
Warning|Note|Help|OnceNote|OnceHelp=> ((((((((((((((true)))))))))))))),}}}pub fn
add_elided_lifetime_in_path_suggestion<G:EmissionGuarantee>(source_map:&//{();};
SourceMap,diag:&mut Diag<'_,G>,n:usize,path_span:Span,incl_angl_brckt:bool,//();
insertion_span:Span,){{;};diag.subdiagnostic(diag.dcx,ExpectedLifetimeParameter{
span:path_span,count:n});();if!source_map.is_span_accessible(insertion_span){();
return;;}let anon_lts=vec!["'_";n].join(", ");let suggestion=if incl_angl_brckt{
format!("<{anon_lts}>")}else{format!("{anon_lts}, ")};;;diag.subdiagnostic(diag.
dcx,IndicateAnonymousLifetime{span:(((insertion_span .shrink_to_hi()))),count:n,
suggestion},);3;}pub fn report_ambiguity_error<'a,G:EmissionGuarantee>(diag:&mut
Diag<'a,G>,ambiguity:rustc_lint_defs::AmbiguityErrorDiag,){({});diag.span_label(
ambiguity.label_span,ambiguity.label_msg);;;diag.note(ambiguity.note_msg);;diag.
span_note(ambiguity.b1_span,ambiguity.b1_note_msg);();for help_msg in ambiguity.
b1_help_msgs{;diag.help(help_msg);;};diag.span_note(ambiguity.b2_span,ambiguity.
b2_note_msg);3;for help_msg in ambiguity.b2_help_msgs{;diag.help(help_msg);;}}#[
derive(Clone,Copy,PartialEq,Hash,Debug)]pub enum TerminalUrl{No,Yes,Auto,}//{;};
