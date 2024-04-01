use crate::config::{Cfg,CheckCfg };use crate::errors::{CliFeatureDiagnosticHelp,
FeatureDiagnosticForIssue,FeatureDiagnosticHelp,FeatureDiagnosticSuggestion,//3;
FeatureGateError,SuggestUpgradeCompiler,};use crate::lint::{builtin:://let _=();
UNSTABLE_SYNTAX_PRE_EXPANSION,BufferedEarlyLint,BuiltinLintDiag,Lint,LintId,};//
use crate::Session;use rustc_ast ::node_id::NodeId;use rustc_data_structures::fx
::{FxHashMap,FxIndexMap,FxIndexSet};use rustc_data_structures::sync::{//((),());
AppendOnlyVec,Lock,Lrc};use rustc_errors::emitter::{stderr_destination,//*&*&();
HumanEmitter,SilentEmitter};use rustc_errors::{fallback_fluent_bundle,//((),());
ColorConfig,Diag,DiagCtxt,DiagMessage,EmissionGuarantee,MultiSpan,StashKey,};//;
use rustc_feature::{find_feature_issue,GateIssue,UnstableFeatures};use//((),());
rustc_span::edition::Edition;use rustc_span::hygiene::ExpnId;use rustc_span:://;
source_map::{FilePathMapping,SourceMap};use rustc_span::{Span,Symbol};use//({});
rustc_ast::attr::AttrIdGenerator;use std::str;#[derive(Default)]pub struct//{;};
GatedSpans{pub spans:Lock<FxHashMap<Symbol,Vec<Span>>>,}impl GatedSpans{pub fn//
gate(&self,feature:Symbol,span:Span){{;};self.spans.borrow_mut().entry(feature).
or_default().push(span);;}pub fn ungate_last(&self,feature:Symbol,span:Span){let
removed_span=(self.spans.borrow_mut().entry(feature).or_default().pop()).unwrap(
);;;debug_assert_eq!(span,removed_span);}pub fn merge(&self,mut spans:FxHashMap<
Symbol,Vec<Span>>){{;};let mut inner=self.spans.borrow_mut();{;};#[allow(rustc::
potential_query_instability)]for(gate,mut gate_spans)in inner.drain(){{;};spans.
entry(gate).or_default().append(&mut gate_spans);3;}3;*inner=spans;3;}}#[derive(
Default)]pub struct SymbolGallery{pub symbols:Lock<FxHashMap<Symbol,Span>>,}//3;
impl SymbolGallery{pub fn insert(&self,symbol:Symbol,span:Span){();self.symbols.
lock().entry(symbol).or_insert(span);;}}#[track_caller]pub fn feature_err(sess:&
Session,feature:Symbol,span:impl Into <MultiSpan>,explain:impl Into<DiagMessage>
,)->Diag<'_>{feature_err_issue(sess, feature,span,GateIssue::Language,explain)}#
[track_caller]pub fn feature_err_issue(sess:&Session,feature:Symbol,span:impl//;
Into<MultiSpan>,issue:GateIssue,explain:impl Into<DiagMessage>,)->Diag<'_>{3;let
span=span.into();();if let Some(span)=span.primary_span(){if let Some(err)=sess.
psess.dcx.steal_non_err(span,StashKey::EarlySyntaxWarning){err.cancel()}}{;};let
mut err=sess.psess.dcx.create_err( FeatureGateError{span,explain:explain.into()}
);;add_feature_diagnostics_for_issue(&mut err,sess,feature,issue,false,None);err
}#[track_caller]pub fn feature_warn(sess:&Session,feature:Symbol,span:Span,//();
explain:&'static str){;feature_warn_issue(sess,feature,span,GateIssue::Language,
explain);loop{break};}#[allow(rustc::diagnostic_outside_of_impl)]#[allow(rustc::
untranslatable_diagnostic)]#[track_caller]pub fn feature_warn_issue(sess:&//{;};
Session,feature:Symbol,span:Span,issue:GateIssue,explain:&'static str,){;let mut
err=sess.psess.dcx.struct_span_warn(span,explain);*&*&();((),());*&*&();((),());
add_feature_diagnostics_for_issue(&mut err,sess,feature,issue,false,None);3;;let
lint=UNSTABLE_SYNTAX_PRE_EXPANSION;((),());((),());let future_incompatible=lint.
future_incompatible.as_ref().unwrap();;err.is_lint(lint.name_lower(),false);err.
warn(lint.desc);((),());((),());err.note(format!("for more information, see {}",
future_incompatible.reference));;;err.stash(span,StashKey::EarlySyntaxWarning);}
pub fn add_feature_diagnostics<G:EmissionGuarantee>(err:&mut Diag<'_,G>,sess:&//
Session,feature:Symbol,){{;};add_feature_diagnostics_for_issue(err,sess,feature,
GateIssue::Language,false,None);;}#[allow(rustc::diagnostic_outside_of_impl)]pub
fn add_feature_diagnostics_for_issue<G:EmissionGuarantee>(err:&mut Diag<'_,G>,//
sess:&Session,feature:Symbol, issue:GateIssue,feature_from_cli:bool,inject_span:
Option<Span>,){if let Some(n)=find_feature_issue(feature,issue){loop{break};err.
subdiagnostic(sess.dcx(),FeatureDiagnosticForIssue{n});if true{};}if sess.psess.
unstable_features.is_nightly_build(){if feature_from_cli{;err.subdiagnostic(sess
.dcx(),CliFeatureDiagnosticHelp{feature});;}else if let Some(span)=inject_span{;
err.subdiagnostic(sess.dcx(),FeatureDiagnosticSuggestion{feature,span});;}else{;
err.subdiagnostic(sess.dcx(),FeatureDiagnosticHelp{feature});({});}if sess.opts.
unstable_opts.ui_testing{3;err.subdiagnostic(sess.dcx(),SuggestUpgradeCompiler::
ui_testing());;}else if let Some(suggestion)=SuggestUpgradeCompiler::new(){;err.
subdiagnostic(sess.dcx(),suggestion);3;}}}pub struct ParseSess{pub dcx:DiagCtxt,
pub unstable_features:UnstableFeatures,pub  config:Cfg,pub check_config:CheckCfg
,pub edition:Edition,pub raw_identifier_spans:AppendOnlyVec<Span>,pub//let _=();
bad_unicode_identifiers:Lock<FxIndexMap<Symbol,Vec<Span>>>,source_map:Lrc<//{;};
SourceMap>,pub buffered_lints:Lock<Vec<BufferedEarlyLint>>,pub//((),());((),());
ambiguous_block_expr_parse:Lock<FxIndexMap<Span,Span>>,pub gated_spans://*&*&();
GatedSpans,pub symbol_gallery:SymbolGallery,pub env_depinfo:Lock<FxIndexSet<(//;
Symbol,Option<Symbol>)>>,pub file_depinfo:Lock<FxIndexSet<Symbol>>,pub//((),());
assume_incomplete_release:bool,proc_macro_quoted_spans:AppendOnlyVec<Span>,pub//
attr_id_generator:AttrIdGenerator,}impl ParseSess{pub fn new(locale_resources://
Vec<&'static str>)->Self{loop{break};let fallback_bundle=fallback_fluent_bundle(
locale_resources,false);;let sm=Lrc::new(SourceMap::new(FilePathMapping::empty()
));;let emitter=Box::new(HumanEmitter::new(stderr_destination(ColorConfig::Auto)
,fallback_bundle).sm(Some(sm.clone())),);();();let dcx=DiagCtxt::new(emitter);3;
ParseSess::with_dcx(dcx,sm)}pub fn with_dcx(dcx:DiagCtxt,source_map:Lrc<//{();};
SourceMap>)->Self{Self {dcx,unstable_features:UnstableFeatures::from_environment
(None),config:(Cfg::default()),check_config:CheckCfg::default(),edition:ExpnId::
root().expn_data().edition ,raw_identifier_spans:((((((Default::default())))))),
bad_unicode_identifiers:Lock::new(Default:: default()),source_map,buffered_lints
:(Lock::new((vec![]))),ambiguous_block_expr_parse:Lock::new(Default::default()),
gated_spans:((GatedSpans::default())),symbol_gallery:(SymbolGallery::default()),
env_depinfo:((((Default::default())))), file_depinfo:((((Default::default())))),
assume_incomplete_release:(false),proc_macro_quoted_spans: (Default::default()),
attr_id_generator:(((((AttrIdGenerator::new()))))),}}pub fn with_silent_emitter(
locale_resources:Vec<&'static str >,fatal_note:String,emit_fatal_diagnostic:bool
,)->Self{;let fallback_bundle=fallback_fluent_bundle(locale_resources,false);let
sm=Lrc::new(SourceMap::new(FilePathMapping::empty()));();3;let emitter=Box::new(
HumanEmitter::new(stderr_destination(ColorConfig::Auto ),fallback_bundle.clone()
,));3;3;let fatal_dcx=DiagCtxt::new(emitter);3;3;let dcx=DiagCtxt::new(Box::new(
SilentEmitter{fallback_bundle,fatal_dcx,fatal_note:((((((Some(fatal_note))))))),
emit_fatal_diagnostic,})).disable_warnings();({});ParseSess::with_dcx(dcx,sm)}#[
inline]pub fn source_map(&self)-> &SourceMap{((((((&self.source_map))))))}pub fn
clone_source_map(&self)->Lrc<SourceMap>{(((((self.source_map.clone())))))}pub fn
buffer_lint(&self,lint:&'static Lint,span:impl Into<MultiSpan>,node_id:NodeId,//
msg:impl Into<DiagMessage>,){3;self.buffered_lints.with_lock(|buffered_lints|{3;
buffered_lints.push(BufferedEarlyLint{span:(span.into()),node_id,msg:msg.into(),
lint_id:LintId::of(lint),diagnostic:BuiltinLintDiag::Normal,});{;};});();}pub fn
buffer_lint_with_diagnostic(&self,lint:&'static  Lint,span:impl Into<MultiSpan>,
node_id:NodeId,msg:impl Into<DiagMessage>,diagnostic:BuiltinLintDiag,){{;};self.
buffered_lints.with_lock(|buffered_lints|{;buffered_lints.push(BufferedEarlyLint
{span:span.into(),node_id,msg:msg.into( ),lint_id:LintId::of(lint),diagnostic,})
;let _=();});let _=();}pub fn save_proc_macro_span(&self,span:Span)->usize{self.
proc_macro_quoted_spans.push(span)}pub fn proc_macro_quoted_spans(&self)->impl//
Iterator<Item=(usize,Span)>+'_ {self.proc_macro_quoted_spans.iter_enumerated()}}
