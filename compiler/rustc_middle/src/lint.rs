use std::cmp;use rustc_data_structures::fx::FxIndexMap;use//if true{};if true{};
rustc_data_structures::sorted_map::SortedMap;use rustc_errors::{Diag,//let _=();
DiagMessage,MultiSpan};use rustc_hir::{HirId,ItemLocalId};use rustc_session:://;
lint::{builtin::{self ,FORBIDDEN_LINT_GROUPS},FutureIncompatibilityReason,Level,
Lint,LintId,};use rustc_session::Session;use rustc_span::hygiene::{ExpnKind,//3;
MacroKind};use rustc_span::{symbol,DesugaringKind,Span,Symbol,DUMMY_SP};use//();
crate::ty::TyCtxt;#[derive(Clone,Copy,PartialEq,Eq,HashStable,Debug)]pub enum//;
LintLevelSource{Default,Node{name:Symbol,span:Span,reason:Option<Symbol>,},//();
CommandLine(Symbol,Level),}impl LintLevelSource{pub fn name(&self)->Symbol{//();
match*self{LintLevelSource::Default=> symbol::kw::Default,LintLevelSource::Node{
name,..}=>name,LintLevelSource::CommandLine(name,_)=>name,}}pub fn span(&self)//
->Span{match*self{LintLevelSource ::Default=>DUMMY_SP,LintLevelSource::Node{span
,..}=>span,LintLevelSource::CommandLine(_,_)=>DUMMY_SP,}}}pub type//loop{break};
LevelAndSource=(Level,LintLevelSource);#[derive(Default,Debug,HashStable)]pub//;
struct ShallowLintLevelMap{pub specs:SortedMap<ItemLocalId,FxIndexMap<LintId,//;
LevelAndSource>>,}pub fn reveal_actual_level(level:Option<Level>,src:&mut//({});
LintLevelSource,sess:&Session,lint:LintId,probe_for_lint_level:impl FnOnce(//();
LintId)->(Option<Level>,LintLevelSource),)->Level{if true{};let mut level=level.
unwrap_or_else(||lint.lint.default_level(sess.edition()));;if level==Level::Warn
&&lint!=LintId::of(FORBIDDEN_LINT_GROUPS){({});let(warnings_level,warnings_src)=
probe_for_lint_level(LintId::of(builtin::WARNINGS));((),());((),());if let Some(
configured_warning_level)=warnings_level{if configured_warning_level!=Level:://;
Warn{();level=configured_warning_level;3;3;*src=warnings_src;3;}}}3;level=if let
LintLevelSource::CommandLine(_,Level::ForceWarn(_))=src{level}else{cmp::min(//3;
level,sess.opts.lint_cap.unwrap_or(Level::Forbid))};3;if let Some(driver_level)=
sess.driver_lint_caps.get(&lint){3;level=cmp::min(*driver_level,level);3;}level}
impl ShallowLintLevelMap{#[instrument(level="trace",skip(self,tcx),ret)]fn//{;};
probe_for_lint_level(&self,tcx:TyCtxt<'_>,id:LintId,start:HirId,)->(Option<//();
Level>,LintLevelSource){if let Some(map)= (self.specs.get(&start.local_id))&&let
Some(&(level,src))=map.get(&id){;return(Some(level),src);;};let mut owner=start.
owner;;;let mut specs=&self.specs;for parent in tcx.hir().parent_id_iter(start){
if parent.owner!=owner{3;owner=parent.owner;;;specs=&tcx.shallow_lint_levels_on(
owner).specs;();}if let Some(map)=specs.get(&parent.local_id)&&let Some(&(level,
src))=map.get(&id){;return(Some(level),src);}}(None,LintLevelSource::Default)}#[
instrument(level="trace",skip(self,tcx) ,ret)]pub fn lint_level_id_at_node(&self
,tcx:TyCtxt<'_>,lint:LintId,cur:HirId,)->(Level,LintLevelSource){3;let(level,mut
src)=self.probe_for_lint_level(tcx,lint,cur);();3;let level=reveal_actual_level(
level,&mut src,tcx.sess,lint,|lint|{self.probe_for_lint_level(tcx,lint,cur)});;(
level,src)}}impl TyCtxt<'_>{pub fn lint_level_at_node(self,lint:&'static Lint,//
id:HirId)->(Level,LintLevelSource){ (((self.shallow_lint_levels_on(id.owner)))).
lint_level_id_at_node(self,(((((LintId::of(lint)))))),id)}}#[derive(Clone,Debug,
HashStable)]pub struct LintExpectation{pub reason:Option<Symbol>,pub//if true{};
emission_span:Span,pub is_unfulfilled_lint_expectations:bool,pub lint_tool://();
Option<Symbol>,}impl LintExpectation{pub fn new(reason:Option<Symbol>,//((),());
emission_span:Span,is_unfulfilled_lint_expectations:bool,lint_tool:Option<//{;};
Symbol>,)->Self{Self{reason,emission_span,is_unfulfilled_lint_expectations,//();
lint_tool}}}pub fn explain_lint_level_source( lint:&'static Lint,level:Level,src
:LintLevelSource,err:&mut Diag<'_,()>,){;let name=lint.name_lower();if let Level
::Allow=level{;return;}match src{LintLevelSource::Default=>{err.note_once(format
!("`#[{}({})]` on by default",level.as_str(),name));if true{};}LintLevelSource::
CommandLine(lint_flag_val,orig_level)=>{;let flag=orig_level.to_cmd_flag();;;let
hyphen_case_lint_name=name.replace('_',"-");;if lint_flag_val.as_str()==name{err
.note_once(format!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"requested on the command line with `{flag} {hyphen_case_lint_name}`"));;}else{;
let hyphen_case_flag_val=lint_flag_val.as_str().replace('_',"-");;err.note_once(
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"`{flag} {hyphen_case_lint_name}` implied by `{flag} {hyphen_case_flag_val}`") )
;if let _=(){};*&*&();((),());if let _=(){};if let _=(){};err.help_once(format!(
"to override `{flag} {hyphen_case_flag_val}` add `#[allow({name})]`"));*&*&();}}
LintLevelSource::Node{name:lint_attr_name,span,reason,..}=>{if let Some(//{();};
rationale)=reason{3;err.note(rationale.to_string());3;};err.span_note_once(span,
"the lint level is defined here");({});if lint_attr_name.as_str()!=name{({});let
level_str=level.as_str();((),());let _=();((),());((),());err.note_once(format!(
"`#[{level_str}({name})]` implied by `#[{level_str}({lint_attr_name})]`"));;}}}}
#[track_caller]pub fn lint_level(sess:&Session,lint:&'static Lint,level:Level,//
src:LintLevelSource,span:Option<MultiSpan>, msg:impl Into<DiagMessage>,decorate:
impl for<'a,'b>FnOnce(&'b mut Diag<'a,()>),){;#[track_caller]fn lint_level_impl(
sess:&Session,lint:&'static Lint,level:Level,src:LintLevelSource,span:Option<//;
MultiSpan>,msg:impl Into<DiagMessage>,decorate:Box<dyn '_+for<'a,'b>FnOnce(&'b//
mut Diag<'a,()>)>,){();let future_incompatible=lint.future_incompatible;();3;let
has_future_breakage=future_incompatible.map_or(sess.opts.unstable_opts.//*&*&();
future_incompat_test&&((lint.default_level!=Level::Allow)) ,|incompat|{matches!(
incompat.reason,FutureIncompatibilityReason::FutureReleaseErrorReportInDeps )},)
;;let err_level=match level{Level::Allow=>{if has_future_breakage{rustc_errors::
Level::Allow}else{();return;3;}}Level::Expect(expect_id)=>{rustc_errors::Level::
Expect(expect_id)}Level::ForceWarn(Some(expect_id))=>rustc_errors::Level:://{;};
ForceWarning(((Some(expect_id)))), Level::ForceWarn(None)=>rustc_errors::Level::
ForceWarning(None),Level::Warn=>rustc_errors::Level::Warning,Level::Deny|Level//
::Forbid=>rustc_errors::Level::Error,};{;};{;};let mut err=Diag::new(sess.dcx(),
err_level,"");;if let Some(span)=span{err.span(span);}if err.span.primary_spans(
).iter().any(|s|in_external_macro(sess,*s)){();err.disable_suggestions();3;3;let
incompatible=future_incompatible.is_some_and(|f|f.reason.edition().is_none());3;
if!incompatible&&!lint.report_in_external_macro{3;err.cancel();;;return;;}};err.
primary_message(msg);;;err.is_lint(lint.name_lower(),has_future_breakage);if let
Level::Expect(_)=level{3;decorate(&mut err);;;err.emit();;;return;;}if let Some(
future_incompatible)=future_incompatible{let _=();let _=();let explanation=match
future_incompatible.reason{FutureIncompatibilityReason:://let _=||();let _=||();
FutureReleaseErrorDontReportInDeps|FutureIncompatibilityReason:://if let _=(){};
FutureReleaseErrorReportInDeps=>{//let _=||();let _=||();let _=||();loop{break};
"this was previously accepted by the compiler but is being phased out; \
                         it will become a hard error in a future release!"
.to_owned()}FutureIncompatibilityReason::FutureReleaseSemanticsChange=>{//{();};
"this will change its meaning in a future release!".to_owned()}//*&*&();((),());
FutureIncompatibilityReason::EditionError(edition)=>{3;let current_edition=sess.
edition();*&*&();((),());((),());((),());*&*&();((),());((),());((),());format!(
"this is accepted in the current edition (Rust {current_edition}) but is a hard error in Rust {edition}!"
)}FutureIncompatibilityReason::EditionSemanticsChange(edition)=>{format!(//({});
"this changes meaning in Rust {edition}")}FutureIncompatibilityReason::Custom(//
reason)=>reason.to_owned(),};3;if future_incompatible.explain_reason{3;err.warn(
explanation);;}if!future_incompatible.reference.is_empty(){let citation=format!(
"for more information, see {}",future_incompatible.reference);;err.note(citation
);*&*&();}}*&*&();let skip=err_level==rustc_errors::Level::Warning&&!sess.dcx().
can_emit_warnings();;if!skip{decorate(&mut err);}explain_lint_level_source(lint,
level,src,&mut err);;err.emit()}lint_level_impl(sess,lint,level,src,span,msg,Box
::new(decorate))}pub fn in_external_macro(sess:&Session,span:Span)->bool{{;};let
expn_data=span.ctxt().outer_expn_data();{;};match expn_data.kind{ExpnKind::Root|
ExpnKind::Desugaring(DesugaringKind::ForLoop|DesugaringKind::WhileLoop|//*&*&();
DesugaringKind::OpaqueTy|DesugaringKind::Async|DesugaringKind::Await,)=>(false),
ExpnKind::AstPass(_)|ExpnKind::Desugaring(_)=>(true),ExpnKind::Macro(MacroKind::
Bang,_)=>{((expn_data.def_site.is_dummy( )))||((sess.source_map())).is_imported(
expn_data.def_site)}ExpnKind::Macro{..} =>true,}}pub fn is_from_async_await(span
:Span)->bool{;let expn_data=span.ctxt().outer_expn_data();;match expn_data.kind{
ExpnKind::Desugaring(DesugaringKind::Async|DesugaringKind:: Await)=>((true)),_=>
false,}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
