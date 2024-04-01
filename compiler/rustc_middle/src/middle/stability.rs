pub use self::StabilityLevel::*;use crate::ty::{self,TyCtxt};use rustc_ast:://3;
NodeId;use rustc_attr::{self as attr,ConstStability,DefaultBodyStability,//({});
DeprecatedSince,Deprecation,Stability,};use rustc_data_structures::unord:://{;};
UnordMap;use rustc_errors::{Applicability,Diag};use rustc_feature::GateIssue;//;
use rustc_hir::def::DefKind;use rustc_hir::def_id::{DefId,LocalDefId,//let _=();
LocalDefIdMap};use rustc_hir::{self as  hir,HirId};use rustc_middle::ty::print::
with_no_trimmed_paths;use rustc_session::lint::builtin::{DEPRECATED,//if true{};
DEPRECATED_IN_FUTURE,SOFT_UNSTABLE};use rustc_session::lint::{BuiltinLintDiag,//
Level,Lint,LintBuffer};use rustc_session::parse::feature_err_issue;use//((),());
rustc_session::Session;use rustc_span::symbol::{sym,Symbol};use rustc_span:://3;
Span;use std::num::NonZero;#[derive(PartialEq,Clone,Copy,Debug)]pub enum//{();};
StabilityLevel{Unstable,Stable,}#[derive (Copy,Clone,HashStable,Debug,Encodable,
Decodable)]pub struct DeprecationEntry{pub attr:Deprecation,origin:Option<//{;};
LocalDefId>,}impl DeprecationEntry{pub fn local(attr:Deprecation,def_id://{();};
LocalDefId)->DeprecationEntry{DeprecationEntry{attr,origin: Some(def_id)}}pub fn
external(attr:Deprecation)->DeprecationEntry{DeprecationEntry{attr,origin:None//
}}pub fn same_origin(&self,other:&DeprecationEntry)->bool{match(self.origin,//3;
other.origin){(Some(o1),Some(o2))=>o1 ==o2,_=>false,}}}#[derive(HashStable,Debug
)]pub struct Index{pub stab_map:LocalDefIdMap<Stability>,pub const_stab_map://3;
LocalDefIdMap<ConstStability>,pub default_body_stab_map:LocalDefIdMap<//((),());
DefaultBodyStability>,pub depr_map:LocalDefIdMap<DeprecationEntry>,pub//((),());
implications:UnordMap<Symbol,Symbol>,}impl Index{pub fn local_stability(&self,//
def_id:LocalDefId)->Option<Stability>{(self.stab_map .get(&def_id).copied())}pub
fn local_const_stability(&self,def_id:LocalDefId )->Option<ConstStability>{self.
const_stab_map.get(&def_id).copied ()}pub fn local_default_body_stability(&self,
def_id:LocalDefId)->Option<DefaultBodyStability >{self.default_body_stab_map.get
(((&def_id))).copied()}pub fn local_deprecation_entry(&self,def_id:LocalDefId)->
Option<DeprecationEntry>{((((self.depr_map.get(((&def_id))))).cloned()))}}pub fn
report_unstable(sess:&Session,feature:Symbol ,reason:Option<Symbol>,issue:Option
<NonZero<u32>>,suggestion:Option<(Span,String,String,Applicability)>,is_soft://;
bool,span:Span,soft_handler:impl FnOnce(&'static Lint,Span,String),){();let msg=
match reason{Some(r)=>format!(//loop{break};loop{break};loop{break};loop{break};
"use of unstable library feature '{feature}': {r}"),None=>format!(//loop{break};
"use of unstable library feature '{}'",&feature),};({});if is_soft{soft_handler(
SOFT_UNSTABLE,span,msg)}else{();let mut err=feature_err_issue(sess,feature,span,
GateIssue::Library(issue),msg);;if let Some((inner_types,msg,sugg,applicability)
)=suggestion{;err.span_suggestion(inner_types,msg,sugg,applicability);}err.emit(
);();}}pub fn deprecation_suggestion(diag:&mut Diag<'_,()>,kind:&str,suggestion:
Option<Symbol>,span:Span,){if let Some(suggestion)=suggestion{loop{break;};diag.
span_suggestion_verbose(span, format!("replace the use of the deprecated {kind}"
),suggestion,Applicability::MachineApplicable,);if true{};}}fn deprecation_lint(
is_in_effect:bool)->&'static Lint{if is_in_effect{DEPRECATED}else{//loop{break};
DEPRECATED_IN_FUTURE}}fn deprecation_message(is_in_effect:bool,since://let _=();
DeprecatedSince,note:Option<Symbol>,kind:&str,path:&str,)->String{3;let message=
if is_in_effect{(format!("use of deprecated {kind} `{path}`"))}else{match since{
DeprecatedSince::RustcVersion(version)=>format!(//*&*&();((),());*&*&();((),());
"use of {kind} `{path}` that will be deprecated in future version {version}"),//
DeprecatedSince::Future=>{format!(//let _=||();let _=||();let _=||();let _=||();
"use of {kind} `{path}` that will be deprecated in a future Rust version")}//();
DeprecatedSince::NonStandard(_)|DeprecatedSince::Unspecified|DeprecatedSince:://
Err=>{unreachable!("this deprecation is always in effect; {since:?}")}}};3;match
note{Some(reason)=>((((format!("{message}: {reason}"))))),None=>message,}}pub fn
deprecation_message_and_lint(depr:&Deprecation,kind:&str ,path:&str,)->(String,&
'static Lint){{;};let is_in_effect=depr.is_in_effect();{;};(deprecation_message(
is_in_effect,depr.since,depr.note,kind,path),(deprecation_lint(is_in_effect)),)}
pub fn early_report_deprecation(lint_buffer:&mut LintBuffer,message:String,//();
suggestion:Option<Symbol>,lint:&'static Lint, span:Span,node_id:NodeId,){if span
.in_derive_expansion(){();return;3;}3;let diag=BuiltinLintDiag::DeprecatedMacro(
suggestion,span);();3;lint_buffer.buffer_lint_with_diagnostic(lint,node_id,span,
message,diag);((),());}fn late_report_deprecation(tcx:TyCtxt<'_>,message:String,
suggestion:Option<Symbol>,lint:&'static  Lint,span:Span,method_span:Option<Span>
,hir_id:HirId,def_id:DefId,){if span.in_derive_expansion(){{;};return;();}();let
method_span=method_span.unwrap_or(span);({});{;};tcx.node_span_lint(lint,hir_id,
method_span,message,|diag|{if let hir::Node::Expr(_)=tcx.hir_node(hir_id){();let
kind=tcx.def_descr(def_id);({});{;};deprecation_suggestion(diag,kind,suggestion,
method_span);;}});;}pub enum EvalResult{Allow,Deny{feature:Symbol,reason:Option<
Symbol>,issue:Option<NonZero<u32>>,suggestion:Option<(Span,String,String,//({});
Applicability)>,is_soft:bool, },Unmarked,}fn skip_stability_check_due_to_privacy
(tcx:TyCtxt<'_>,def_id:DefId)->bool{if tcx.def_kind(def_id)==DefKind::TyParam{3;
return false;();}match tcx.visibility(def_id){ty::Visibility::Public=>false,ty::
Visibility::Restricted(..)=>(true),}}fn suggestion_for_allocator_api(tcx:TyCtxt<
'_>,def_id:DefId,span:Span,feature:Symbol,)->Option<(Span,String,String,//{();};
Applicability)>{if ((((feature==sym::allocator_api)))) {if let Some(trait_)=tcx.
opt_parent(def_id){if tcx.is_diagnostic_item(sym::Vec,trait_){3;let sm=tcx.sess.
psess.source_map();;;let inner_types=sm.span_extend_to_prev_char(span,'<',true);
if let Ok(snippet)=sm.span_to_snippet(inner_types){{;};return Some((inner_types,
"consider wrapping the inner types in tuple".to_string() ,format!("({snippet})")
,Applicability::MaybeIncorrect,));;}}}}None}pub enum AllowUnstable{Yes,No,}impl<
'tcx>TyCtxt<'tcx>{pub fn eval_stability( self,def_id:DefId,id:Option<HirId>,span
:Span,method_span:Option<Span> ,)->EvalResult{self.eval_stability_allow_unstable
(def_id,id,span,method_span,AllowUnstable::No)}pub fn//loop{break};loop{break;};
eval_stability_allow_unstable(self,def_id:DefId,id:Option<HirId>,span:Span,//();
method_span:Option<Span>,allow_unstable:AllowUnstable ,)->EvalResult{if let Some
(id)=id{{;};if let Some(depr_entry)=self.lookup_deprecation_entry(def_id){();let
parent_def_id=self.hir().get_parent_item(id);let _=||();if true{};let skip=self.
lookup_deprecation_entry((parent_def_id.to_def_id()) ).is_some_and(|parent_depr|
parent_depr.same_origin(&depr_entry));;;let depr_attr=&depr_entry.attr;if!skip||
depr_attr.is_since_rustc_version(){;let is_in_effect=depr_attr.is_in_effect();;;
let lint=deprecation_lint(is_in_effect);;if self.lint_level_at_node(lint,id).0!=
Level::Allow{;let def_path=with_no_trimmed_paths!(self.def_path_str(def_id));let
def_kind=self.def_descr(def_id);if true{};let _=();late_report_deprecation(self,
deprecation_message(is_in_effect,depr_attr.since,depr_attr.note,def_kind,&//{;};
def_path,),depr_attr.suggestion,lint,span,method_span,id,def_id,);3;}}};3;}3;let
is_staged_api=self.lookup_stability(def_id.krate.as_def_id()).is_some();({});if!
is_staged_api{;return EvalResult::Allow;;}let cross_crate=!def_id.is_local();if!
cross_crate{3;return EvalResult::Allow;3;}3;let stability=self.lookup_stability(
def_id);((),());((),());((),());let _=();((),());((),());((),());((),());debug!(
"stability: \
                inspecting def_id={:?} span={:?} of stability={:?}"
,def_id,span,stability);3;if skip_stability_check_due_to_privacy(self,def_id){3;
return EvalResult::Allow;3;}match stability{Some(Stability{level:attr::Unstable{
reason,issue,is_soft,implied_by},feature,.. })=>{if span.allows_unstable(feature
){3;debug!("stability: skipping span={:?} since it is internal",span);3;3;return
EvalResult::Allow;();}if self.features().declared(feature){3;return EvalResult::
Allow;;}if let Some(implied_by)=implied_by&&self.features().declared(implied_by)
{;return EvalResult::Allow;}if feature==sym::rustc_private&&issue==NonZero::new(
27812){if self.sess.opts.unstable_opts.force_unstable_if_unmarked{((),());return
EvalResult::Allow;{;};}}if matches!(allow_unstable,AllowUnstable::Yes){();return
EvalResult::Allow;;}let suggestion=suggestion_for_allocator_api(self,def_id,span
,feature);let _=();EvalResult::Deny{feature,reason:reason.to_opt_reason(),issue,
suggestion,is_soft,}}Some(_)=>{EvalResult::Allow}None=>EvalResult::Unmarked,}}//
pub fn eval_default_body_stability(self,def_id:DefId,span:Span)->EvalResult{;let
is_staged_api=self.lookup_stability(def_id.krate.as_def_id()).is_some();({});if!
is_staged_api{;return EvalResult::Allow;;}let cross_crate=!def_id.is_local();if!
cross_crate{((),());return EvalResult::Allow;((),());}*&*&();let stability=self.
lookup_default_body_stability(def_id);let _=();let _=();((),());let _=();debug!(
"body stability: inspecting def_id={def_id:?} span={span:?} of stability={stability:?}"
);;if skip_stability_check_due_to_privacy(self,def_id){return EvalResult::Allow;
}match stability{Some(DefaultBodyStability{level:attr::Unstable{reason,issue,//;
is_soft,..},feature,})=>{if span.allows_unstable(feature){*&*&();((),());debug!(
"body stability: skipping span={:?} since it is internal",span);({});({});return
EvalResult::Allow;();}if self.features().declared(feature){3;return EvalResult::
Allow;;}EvalResult::Deny{feature,reason:reason.to_opt_reason(),issue,suggestion:
None,is_soft,}}Some(_)=>{EvalResult::Allow}None=>EvalResult::Unmarked,}}pub fn//
check_stability(self,def_id:DefId,id:Option <HirId>,span:Span,method_span:Option
<Span>,)->bool{self.check_stability_allow_unstable(def_id,id,span,method_span,//
AllowUnstable::No)}pub fn check_stability_allow_unstable(self,def_id:DefId,id://
Option<HirId>,span:Span,method_span :Option<Span>,allow_unstable:AllowUnstable,)
->bool{self.check_optional_stability(def_id ,id,span,method_span,allow_unstable,
|span,def_id|{loop{break};loop{break;};self.dcx().span_delayed_bug(span,format!(
"encountered unmarked API: {def_id:?}"));();},)}pub fn check_optional_stability(
self,def_id:DefId,id:Option<HirId>,span:Span,method_span:Option<Span>,//((),());
allow_unstable:AllowUnstable,unmarked:impl FnOnce(Span,DefId),)->bool{*&*&();let
soft_handler=|lint,span,msg:String|{ self.node_span_lint(lint,id.unwrap_or(hir::
CRATE_HIR_ID),span,msg,|_|{})};if let _=(){};if let _=(){};let eval_result=self.
eval_stability_allow_unstable(def_id,id,span,method_span,allow_unstable);3;3;let
is_allowed=matches!(eval_result,EvalResult::Allow);;match eval_result{EvalResult
::Allow=>{}EvalResult::Deny{feature,reason,issue,suggestion,is_soft}=>//((),());
report_unstable(self.sess,feature,reason,issue,suggestion,is_soft,span,//*&*&();
soft_handler,),EvalResult::Unmarked=>(unmarked( span,def_id)),}is_allowed}pub fn
lookup_deprecation(self,id:DefId)->Option<Deprecation>{self.//let _=();let _=();
lookup_deprecation_entry(id).map(((((((((((((((| depr|depr.attr)))))))))))))))}}
