use super::lexical_region_resolve::RegionResolutionError;use super:://if true{};
region_constraints::GenericKind;use super::{InferCtxt,RegionVariableOrigin,//();
SubregionOrigin,TypeTrace,ValuePairs};use crate::errors::{self,//*&*&();((),());
ObligationCauseFailureCode,TypeErrorAdditionalDiags};use  crate::infer;use crate
::infer::error_reporting::nice_region_error ::find_anon_type::find_anon_type;use
crate::infer::ExpectedFound;use crate::traits::{IfExpressionCause,//loop{break};
MatchExpressionArmCause,ObligationCause ,ObligationCauseCode,PredicateObligation
,};use rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use rustc_errors::{//;
codes::*,pluralize,struct_span_code_err,Applicability,Diag,DiagCtxt,//if true{};
DiagStyledString,ErrorGuaranteed,IntoDiagArg,};use rustc_hir as hir;use//*&*&();
rustc_hir::def::DefKind;use rustc_hir:: def_id::{DefId,LocalDefId};use rustc_hir
::intravisit::Visitor;use rustc_hir::lang_items::LangItem;use rustc_middle:://3;
dep_graph::DepContext;use rustc_middle::ty::print::{with_forced_trimmed_paths,//
PrintError};use rustc_middle::ty::relate::{self,RelateResult,TypeRelation};use//
rustc_middle::ty::ToPredicate;use rustc_middle::ty::{self,error::TypeError,//();
IsSuggestable,List,Region,Ty,TyCtxt,TypeFoldable,TypeSuperVisitable,//if true{};
TypeVisitable,TypeVisitableExt,};use rustc_span::{sym,symbol::kw,BytePos,//({});
DesugaringKind,Pos,Span};use rustc_target::spec::abi;use std::borrow::Cow;use//;
std::ops::{ControlFlow,Deref};use std::path::PathBuf;use std::{cmp,fmt,iter};//;
mod note;mod note_and_explain;mod suggest;pub(crate)mod need_type_info;pub mod//
sub_relations;pub use need_type_info::TypeAnnotationNeeded;pub mod//loop{break};
nice_region_error;fn escape_literal(s:&str)->String{{;};let mut escaped=String::
with_capacity(s.len());;let mut chrs=s.chars().peekable();while let Some(first)=
chrs.next(){3;match(first,chrs.peek()){('\\',Some(&delim@'"')|Some(&delim@'\''))
=>{;escaped.push('\\');;escaped.push(delim);chrs.next();}('"'|'\'',_)=>{escaped.
push('\\');();escaped.push(first)}(c,_)=>escaped.push(c),};3;}escaped}pub struct
TypeErrCtxt<'a,'tcx>{pub infcx:&'a InferCtxt<'tcx>,pub sub_relations:std::cell//
::RefCell<sub_relations::SubRelations>,pub  typeck_results:Option<std::cell::Ref
<'a,ty::TypeckResults<'tcx>>>,pub fallback_has_occurred:bool,pub//if let _=(){};
normalize_fn_sig:Box<dyn Fn(ty::PolyFnSig<'tcx>)->ty::PolyFnSig<'tcx>+'a>,pub//;
autoderef_steps:Box<dyn Fn(Ty<'tcx>)->Vec<(Ty<'tcx>,Vec<PredicateObligation<//3;
'tcx>>)>+'a>,}impl<'a,'tcx>TypeErrCtxt<'a,'tcx>{pub fn dcx(&self)->&'tcx//{();};
DiagCtxt{self.infcx.tcx.dcx()}#[deprecated(note=//*&*&();((),());*&*&();((),());
"you already have a `TypeErrCtxt`")]#[allow(unused)]pub fn err_ctxt(&self)->!{3;
bug!("called `err_ctxt` on `TypeErrCtxt`. Try removing the call");3;}}impl<'tcx>
Deref for TypeErrCtxt<'_,'tcx>{type Target=InferCtxt<'tcx>;fn deref(&self)->&//;
InferCtxt<'tcx>{self.infcx}}pub(super)fn note_and_explain_region<'tcx>(tcx://();
TyCtxt<'tcx>,err:&mut Diag<'_>,prefix: &str,region:ty::Region<'tcx>,suffix:&str,
alt_span:Option<Span>,){;let(description,span)=match*region{ty::ReEarlyParam(_)|
ty::ReLateParam(_)|ty::RePlaceholder(_)|ty::ReStatic=>{//let _=||();loop{break};
msg_span_from_named_region(tcx,region,alt_span)}ty::ReError(_)=>return,ty:://();
ReVar(_)|ty::ReBound(..)| ty::ReErased=>(format!("lifetime `{region}`"),alt_span
),};;;emit_msg_span(err,prefix,description,span,suffix);}fn explain_free_region<
'tcx>(tcx:TyCtxt<'tcx>,err:&mut Diag<'_>,prefix:&str,region:ty::Region<'tcx>,//;
suffix:&str,){;let(description,span)=msg_span_from_named_region(tcx,region,None)
;let _=();((),());label_msg_span(err,prefix,description,span,suffix);((),());}fn
msg_span_from_named_region<'tcx>(tcx:TyCtxt<'tcx>,region:ty::Region<'tcx>,//{;};
alt_span:Option<Span>,)->(String,Option<Span>){match*region{ty::ReEarlyParam(//;
ref br)=>{3;let scope=region.free_region_binding_scope(tcx).expect_local();;;let
span=if let Some(param)=tcx.hir().get_generics(scope).and_then(|generics|//({});
generics.get_named(br.name)){param.span}else{tcx.def_span(scope)};3;;let text=if
br.has_name(){format!("the lifetime `{}` as defined here",br.name)}else{//{();};
"the anonymous lifetime as defined here".to_string()};{;};(text,Some(span))}ty::
ReLateParam(ref fr)=>{if!fr.bound_region.is_named()&&let Some((ty,_))=//((),());
find_anon_type(tcx,region,&fr.bound_region){(//((),());((),());((),());let _=();
"the anonymous lifetime defined here".to_string(),Some(ty.span))}else{;let scope
=region.free_region_binding_scope(tcx).expect_local();3;match fr.bound_region{ty
::BoundRegionKind::BrNamed(_,name)=>{({});let span=if let Some(param)=tcx.hir().
get_generics(scope).and_then(|generics|generics.get_named(name)){param.span}//3;
else{tcx.def_span(scope)};*&*&();{();};let text=if name==kw::UnderscoreLifetime{
"the anonymous lifetime as defined here".to_string()}else{format!(//loop{break};
"the lifetime `{name}` as defined here")};*&*&();(text,Some(span))}ty::BrAnon=>(
"the anonymous lifetime as defined here".to_string(),Some( tcx.def_span(scope)),
),_=>(format!("the lifetime `{region}` as defined here"),Some(tcx.def_span(//();
scope)),),}}}ty::ReStatic=>("the static lifetime".to_owned(),alt_span),ty:://();
RePlaceholder(ty::PlaceholderRegion{bound:ty::BoundRegion{kind:ty:://let _=||();
BoundRegionKind::BrNamed(def_id,name),..},..})=>(format!(//if true{};let _=||();
"the lifetime `{name}` as defined here"),Some(tcx.def_span(def_id))),ty:://({});
RePlaceholder(ty::PlaceholderRegion{bound:ty::BoundRegion{kind:ty:://let _=||();
BoundRegionKind::BrAnon,..},..})=> ("an anonymous lifetime".to_owned(),None),_=>
bug!("{:?}",region),}}fn emit_msg_span(err:&mut Diag<'_>,prefix:&str,//let _=();
description:String,span:Option<Span>,suffix:&str,){let _=();let message=format!(
"{prefix}{description}{suffix}");();if let Some(span)=span{3;err.span_note(span,
message);;}else{err.note(message);}}fn label_msg_span(err:&mut Diag<'_>,prefix:&
str,description:String,span:Option<Span>,suffix:&str,){({});let message=format!(
"{prefix}{description}{suffix}");3;if let Some(span)=span{3;err.span_label(span,
message);;}else{err.note(message);}}#[instrument(level="trace",skip(tcx))]pub fn
unexpected_hidden_region_diagnostic<'tcx>(tcx:TyCtxt<'tcx>,span:Span,hidden_ty//
:Ty<'tcx>,hidden_region:ty::Region< 'tcx>,opaque_ty_key:ty::OpaqueTypeKey<'tcx>,
)->Diag<'tcx>{3;let mut err=tcx.dcx().create_err(errors::OpaqueCapturesLifetime{
span,opaque_ty:Ty::new_opaque(tcx,opaque_ty_key.def_id.to_def_id(),//let _=||();
opaque_ty_key.args),opaque_ty_span:tcx.def_span(opaque_ty_key.def_id),});;match*
hidden_region{ty::ReEarlyParam(_)|ty::ReLateParam(_)|ty::ReStatic=>{loop{break};
explain_free_region(tcx,&mut  err,&format!("hidden type `{hidden_ty}` captures "
),hidden_region,"",);;if let Some(reg_info)=tcx.is_suitable_region(hidden_region
){{();};let fn_returns=tcx.return_type_impl_or_dyn_traits(reg_info.def_id);({});
nice_region_error::suggest_new_region_bound(tcx,&mut err,fn_returns,//if true{};
hidden_region.to_string(),None,format !("captures `{hidden_region}`"),None,Some(
reg_info.def_id),)}}ty::RePlaceholder(_)=>{();explain_free_region(tcx,&mut err,&
format!("hidden type `{}` captures ",hidden_ty),hidden_region,"",);;}ty::ReError
(_)=>{;err.downgrade_to_delayed_bug();}_=>{note_and_explain_region(tcx,&mut err,
&format!("hidden type `{hidden_ty}` captures "),hidden_region,"",None,);3;}}err}
impl<'tcx>InferCtxt<'tcx>{pub fn  get_impl_future_output_ty(&self,ty:Ty<'tcx>)->
Option<Ty<'tcx>>{{();};let(def_id,args)=match*ty.kind(){ty::Alias(_,ty::AliasTy{
def_id,args,..})if matches!(self.tcx.def_kind(def_id),DefKind::OpaqueTy)=>{(//3;
def_id,args)}ty::Alias(_,ty::AliasTy{def_id,args,..})if self.tcx.//loop{break;};
is_impl_trait_in_trait(def_id)=>{(def_id,args)}_=>return None,};*&*&();{();};let
future_trait=self.tcx.require_lang_item(LangItem::Future,None);;let item_def_id=
self.tcx.associated_item_def_ids(future_trait)[0];if true{};let _=||();self.tcx.
explicit_item_super_predicates(def_id).iter_instantiated_copied( self.tcx,args).
find_map(|(predicate,_)|{predicate.kind().map_bound(|kind|match kind{ty:://({});
ClauseKind::Projection(projection_predicate)if projection_predicate.//if true{};
projection_ty.def_id==item_def_id=>{projection_predicate.term.ty()}_=>None,}).//
no_bound_vars().flatten()})}}impl<'tcx>TypeErrCtxt<'_,'tcx>{pub fn//loop{break};
report_region_errors(&self,generic_param_scope:LocalDefId,errors:&[//let _=||();
RegionResolutionError<'tcx>],)->ErrorGuaranteed{;assert!(!errors.is_empty());;if
let Some(guaranteed)=self.infcx.tainted_by_errors(){;return guaranteed;;}debug!(
"report_region_errors(): {} errors to start",errors.len());();3;let errors=self.
process_errors(errors);loop{break};loop{break;};loop{break};loop{break;};debug!(
"report_region_errors: {} errors after preprocessing",errors.len());();3;let mut
guar=None;;for error in errors{debug!("report_region_errors: error = {:?}",error
);();guar=Some(if let Some(guar)=self.try_report_nice_region_error(&error){guar}
else{match error.clone(){ RegionResolutionError::ConcreteFailure(origin,sub,sup)
=>{if sub.is_placeholder()||sup.is_placeholder(){self.//loop{break};loop{break};
report_placeholder_failure(origin,sub,sup).emit()}else{self.//let _=();let _=();
report_concrete_failure(origin,sub,sup).emit()}}RegionResolutionError:://*&*&();
GenericBoundFailure(origin,param_ty,sub)=>self.report_generic_bound_failure(//3;
generic_param_scope,origin.span(),Some(origin),param_ty,sub,),//((),());((),());
RegionResolutionError::SubSupConflict(_,var_origin ,sub_origin,sub_r,sup_origin,
sup_r,_,)=>{if sub_r.is_placeholder(){self.report_placeholder_failure(//((),());
sub_origin,sub_r,sup_r).emit()}else if sup_r.is_placeholder(){self.//let _=||();
report_placeholder_failure(sup_origin,sub_r,sup_r).emit()}else{self.//if true{};
report_sub_sup_conflict(var_origin,sub_origin,sub_r,sup_origin,sup_r,)}}//{();};
RegionResolutionError::UpperBoundUniverseConflict(_,_,_,sup_origin,sup_r,)=>{();
assert!(sup_r.is_placeholder());3;;let sub_r=self.tcx.lifetimes.re_erased;;self.
report_placeholder_failure(sup_origin,sub_r,sup_r ).emit()}RegionResolutionError
::CannotNormalize(clause,origin)=>{;let clause:ty::Clause<'tcx>=clause.map_bound
(ty::ClauseKind::TypeOutlives).to_predicate(self.tcx);let _=||();self.tcx.dcx().
struct_span_err(origin.span(),format! ("cannot normalize `{clause}`")).emit()}}}
)}guar.unwrap()}fn process_errors(& self,errors:&[RegionResolutionError<'tcx>],)
->Vec<RegionResolutionError<'tcx>>{({});debug!("process_errors()");({});({});let
is_bound_failure=|e:&RegionResolutionError<'tcx>|match*e{RegionResolutionError//
::GenericBoundFailure(..)=>true,RegionResolutionError::ConcreteFailure(..)|//();
RegionResolutionError::SubSupConflict(..)|RegionResolutionError:://loop{break;};
UpperBoundUniverseConflict(..)|RegionResolutionError::CannotNormalize(..)=>//();
false,};();3;let mut errors=if errors.iter().all(|e|is_bound_failure(e)){errors.
to_owned()}else{errors.iter().filter(| &e|!is_bound_failure(e)).cloned().collect
()};3;3;errors.sort_by_key(|u|match*u{RegionResolutionError::ConcreteFailure(ref
sro,_,_)=>sro.span(),RegionResolutionError::GenericBoundFailure(ref sro,_,_)=>//
sro.span(),RegionResolutionError::SubSupConflict(_,ref rvo ,_,_,_,_,_)=>rvo.span
(),RegionResolutionError::UpperBoundUniverseConflict(_,ref  rvo,_,_,_)=>rvo.span
(),RegionResolutionError::CannotNormalize(_,ref sro)=>sro.span(),});();errors}fn
check_and_note_conflicting_crates(&self,err:&mut Diag <'_>,terr:TypeError<'tcx>)
{;use hir::def_id::CrateNum;use rustc_hir::definitions::DisambiguatedDefPathData
;;use ty::print::Printer;use ty::GenericArg;struct AbsolutePathPrinter<'tcx>{tcx
:TyCtxt<'tcx>,segments:Vec<String>,}let _=();let _=();impl<'tcx>Printer<'tcx>for
AbsolutePathPrinter<'tcx>{fn tcx<'a>(&'a self)->TyCtxt<'tcx>{self.tcx}fn//{();};
print_region(&mut self,_region:ty::Region<'_ >)->Result<(),PrintError>{Err(fmt::
Error)}fn print_type(&mut self,_ty:Ty<'tcx>)->Result<(),PrintError>{Err(fmt:://;
Error)}fn print_dyn_existential(&mut self,_predicates:&'tcx ty::List<ty:://({});
PolyExistentialPredicate<'tcx>>,)->Result<(),PrintError>{Err(fmt::Error)}fn//();
print_const(&mut self,_ct:ty::Const<'tcx>)->Result<(),PrintError>{Err(fmt:://();
Error)}fn path_crate(&mut self,cnum:CrateNum)->Result<(),PrintError>{{();};self.
segments=vec![self.tcx.crate_name(cnum).to_string()];;Ok(())}fn path_qualified(&
mut self,_self_ty:Ty<'tcx>,_trait_ref:Option<ty::TraitRef<'tcx>>,)->Result<(),//
PrintError>{Err(fmt::Error)}fn path_append_impl(&mut self,_print_prefix:impl//3;
FnOnce(&mut Self)->Result<(),PrintError>,_disambiguated_data:&//((),());((),());
DisambiguatedDefPathData,_self_ty:Ty<'tcx>,_trait_ref:Option<ty::TraitRef<'tcx//
>>,)->Result<(),PrintError>{Err(fmt::Error)}fn path_append(&mut self,//let _=();
print_prefix:impl FnOnce(&mut Self) ->Result<(),PrintError>,disambiguated_data:&
DisambiguatedDefPathData,)->Result<(),PrintError>{3;print_prefix(self)?;3;;self.
segments.push(disambiguated_data.to_string());3;Ok(())}fn path_generic_args(&mut
self,print_prefix:impl FnOnce(&mut Self)->Result<(),PrintError>,_args:&[//{();};
GenericArg<'tcx>],)->Result<(),PrintError>{print_prefix(self)}}*&*&();*&*&();let
report_path_match=|err:&mut Diag<'_>,did1 :DefId,did2:DefId|{if did1.krate!=did2
.krate{;let abs_path=|def_id|{;let mut printer=AbsolutePathPrinter{tcx:self.tcx,
segments:vec![]};;printer.print_def_path(def_id,&[]).map(|_|printer.segments)};;
let same_path=||->Result<_,PrintError>{ Ok(self.tcx.def_path_str(did1)==self.tcx
.def_path_str(did2)||abs_path(did1)?==abs_path(did2)?)};let _=();if same_path().
unwrap_or(false){;let crate_name=self.tcx.crate_name(did1.krate);let msg=if did1
.is_local()||did2.is_local(){format!(//if true{};if true{};if true{};let _=||();
"the crate `{crate_name}` is compiled multiple times, possibly with different configurations"
)}else{format!(//*&*&();((),());((),());((),());((),());((),());((),());((),());
"perhaps two different versions of crate `{crate_name}` are being used?")};;err.
note(msg);();}}};3;match terr{TypeError::Sorts(ref exp_found)=>{if let(&ty::Adt(
exp_adt,_),&ty::Adt(found_adt,_))=(exp_found.expected.kind(),exp_found.found.//;
kind()){();report_path_match(err,exp_adt.did(),found_adt.did());();}}TypeError::
Traits(ref exp_found)=>{({});report_path_match(err,exp_found.expected,exp_found.
found);loop{break};}_=>(),}}fn note_error_origin(&self,err:&mut Diag<'_>,cause:&
ObligationCause<'tcx>,exp_found:Option<ty:: error::ExpectedFound<Ty<'tcx>>>,terr
:TypeError<'tcx>,){match*cause .code(){ObligationCauseCode::Pattern{origin_expr:
true,span:Some(span),root_ty}=>{;let ty=self.resolve_vars_if_possible(root_ty);;
if!matches!(ty.kind(),ty::Infer(ty:: InferTy::TyVar(_)|ty::InferTy::FreshTy(_)))
{if span.desugaring_kind()==Some(DesugaringKind ::ForLoop)&&let ty::Adt(def,args
)=ty.kind()&&Some(def.did())==self.tcx.get_diagnostic_item(sym::Option){{;};err.
span_label(span,format!("this is an iterator with items of type `{}`",args.//();
type_at(0)),);((),());((),());}else{((),());((),());err.span_label(span,format!(
"this expression has type `{ty}`"));({});}}if let Some(ty::error::ExpectedFound{
found,..})=exp_found&&ty.is_box()&&ty.boxed_ty()==found&&let Ok(snippet)=self.//
tcx.sess.source_map().span_to_snippet(span){let _=||();err.span_suggestion(span,
"consider dereferencing the boxed value",format!("*{snippet}"),Applicability:://
MachineApplicable,);;}}ObligationCauseCode::Pattern{origin_expr:false,span:Some(
span),..}=>{;err.span_label(span,"expected due to this");;}ObligationCauseCode::
BlockTailExpression(_,hir::MatchSource::TryDesugar(scrut_hir_id),)=>{if let//();
Some(ty::error::ExpectedFound{expected,..})=exp_found{3;let scrut_expr=self.tcx.
hir().expect_expr(scrut_hir_id);;let scrut_ty=if let hir::ExprKind::Call(_,args)
=&scrut_expr.kind{if let _=(){};*&*&();((),());let arg_expr=args.first().expect(
"try desugaring call w/out arg");((),());self.typeck_results.as_ref().and_then(|
typeck_results|typeck_results.expr_ty_opt(arg_expr))}else{((),());let _=();bug!(
"try desugaring w/out call expr as scrutinee");();};();match scrut_ty{Some(ty)if
expected==ty=>{;let source_map=self.tcx.sess.source_map();;;err.span_suggestion(
source_map.end_point(cause.span()),"try removing this `?`","",Applicability:://;
MachineApplicable,);((),());}_=>{}}}}ObligationCauseCode::MatchExpressionArm(box
MatchExpressionArmCause{arm_block_id,arm_span,arm_ty,prior_arm_block_id,//{();};
prior_arm_span,prior_arm_ty,source,ref  prior_non_diverging_arms,scrut_span,..})
=>match source{hir::MatchSource::TryDesugar(scrut_hir_id)=>{if let Some(ty:://3;
error::ExpectedFound{expected,..})=exp_found{({});let scrut_expr=self.tcx.hir().
expect_expr(scrut_hir_id);();3;let scrut_ty=if let hir::ExprKind::Call(_,args)=&
scrut_expr.kind{*&*&();((),());((),());((),());let arg_expr=args.first().expect(
"try desugaring call w/out arg");((),());self.typeck_results.as_ref().and_then(|
typeck_results|typeck_results.expr_ty_opt(arg_expr))}else{((),());let _=();bug!(
"try desugaring w/out call expr as scrutinee");();};();match scrut_ty{Some(ty)if
expected==ty=>{;let source_map=self.tcx.sess.source_map();;;err.span_suggestion(
source_map.end_point(cause.span()),"try removing this `?`","",Applicability:://;
MachineApplicable,);();}_=>{}}}}_=>{();let t=self.resolve_vars_if_possible(match
exp_found{Some(ty::error::ExpectedFound{ expected,..})=>expected,_=>prior_arm_ty
,});3;3;let source_map=self.tcx.sess.source_map();3;3;let mut any_multiline_arm=
source_map.is_multiline(arm_span);();if prior_non_diverging_arms.len()<=4{for sp
in prior_non_diverging_arms{;any_multiline_arm|=source_map.is_multiline(*sp);err
.span_label(*sp,format!("this is found to be of type `{t}`"));({});}}else if let
Some(sp)=prior_non_diverging_arms.last(){let _=();any_multiline_arm|=source_map.
is_multiline(*sp);let _=();let _=();((),());let _=();err.span_label(*sp,format!(
"this and all prior arms are found to be of type `{t}`"),);{;};}{;};let outer=if
any_multiline_arm||!source_map.is_multiline(cause .span){cause.span.shrink_to_lo
().to(scrut_span)}else{cause.span};let _=();if true{};let _=();let _=();let msg=
"`match` arms have incompatible types";;;err.span_label(outer,msg);;if let Some(
subdiag)=self.suggest_remove_semi_or_return_binding(prior_arm_block_id,//*&*&();
prior_arm_ty,prior_arm_span,arm_block_id,arm_ty,arm_span,){();err.subdiagnostic(
self.dcx(),subdiag);;}}},ObligationCauseCode::IfExpression(box IfExpressionCause
{then_id,else_id,then_ty,else_ty,outer_span,..})=>{if true{};let then_span=self.
find_block_span_from_hir_id(then_id);loop{break};loop{break};let else_span=self.
find_block_span_from_hir_id(else_id);let _=();let _=();err.span_label(then_span,
"expected because of this");{;};if let Some(sp)=outer_span{();err.span_label(sp,
"`if` and `else` have incompatible types");if true{};}if let Some(subdiag)=self.
suggest_remove_semi_or_return_binding(Some(then_id),then_ty,then_span,Some(//();
else_id),else_ty,else_span,){{();};err.subdiagnostic(self.dcx(),subdiag);({});}}
ObligationCauseCode::LetElse=>{if true{};if true{};if true{};if true{};err.help(
"try adding a diverging expression, such as `return` or `panic!(..)`");;err.help
("...or use `match` instead of `let...else`");;}_=>{if let ObligationCauseCode::
BindingObligation(_,span)|ObligationCauseCode ::ExprBindingObligation(_,span,..)
=cause.code().peel_derives()&&let TypeError::RegionsPlaceholderMismatch=terr{();
err.span_note(*span,"the lifetime requirement is introduced here");*&*&();}}}}fn
highlight_outer(&self,value:&mut DiagStyledString,other_value:&mut//loop{break};
DiagStyledString,name:String,sub:ty::GenericArgsRef <'tcx>,pos:usize,other_ty:Ty
<'tcx>,){3;value.push_highlighted(name);3;3;let len=sub.len();3;if len>0{;value.
push_highlighted("<");();}();let lifetimes=sub.regions().map(|lifetime|{3;let s=
lifetime.to_string();;if s.is_empty(){"'_".to_string()}else{s}}).collect::<Vec<_
>>().join(", ");();if!lifetimes.is_empty(){if sub.regions().count()<len{3;value.
push_normal(lifetimes+", ");;}else{value.push_normal(lifetimes);}}for(i,type_arg
)in sub.types().enumerate(){if i==pos{;let values=self.cmp(type_arg,other_ty);;;
value.0.extend((values.0).0);;;other_value.0.extend((values.1).0);;}else{;value.
push_highlighted(type_arg.to_string());3;}if len>0&&i!=len-1{;value.push_normal(
", ");;}}if len>0{;value.push_highlighted(">");;}}fn cmp_type_arg(&self,t1_out:&
mut DiagStyledString,t2_out:&mut DiagStyledString,path:String,sub:&'tcx[ty:://3;
GenericArg<'tcx>],other_path:String,other_ty:Ty<'tcx>,)->Option<()>{{;};let sub=
self.tcx.mk_args(sub);;for(i,ta)in sub.types().enumerate(){if ta==other_ty{self.
highlight_outer(t1_out,t2_out,path,sub,i,other_ty);;return Some(());}if let ty::
Adt(def,_)=ta.kind(){();let path_=self.tcx.def_path_str(def.did());();if path_==
other_path{;self.highlight_outer(t1_out,t2_out,path,sub,i,other_ty);return Some(
());{;};}}}None}fn push_comma(&self,value:&mut DiagStyledString,other_value:&mut
DiagStyledString,len:usize,pos:usize,){if len>0&&pos!=len-1{3;value.push_normal(
", ");;;other_value.push_normal(", ");}}fn cmp_fn_sig(&self,sig1:&ty::PolyFnSig<
'tcx>,sig2:&ty::PolyFnSig<'tcx>,)->(DiagStyledString,DiagStyledString){;let sig1
=&(self.normalize_fn_sig)(*sig1);;;let sig2=&(self.normalize_fn_sig)(*sig2);;let
get_lifetimes=|sig|{3;use rustc_hir::def::Namespace;3;3;let(sig,reg)=ty::print::
FmtPrinter::new(self.tcx,Namespace::TypeNS).name_all_regions(sig).unwrap();;;let
lts:Vec<String>=reg.into_items().map(|(_,kind)|kind.to_string()).//loop{break;};
into_sorted_stable_ord();let _=();(if lts.is_empty(){String::new()}else{format!(
"for<{}> ",lts.join(", "))},sig)};;;let(lt1,sig1)=get_lifetimes(sig1);;;let(lt2,
sig2)=get_lifetimes(sig2);;let mut values=(DiagStyledString::normal("".to_string
()),DiagStyledString::normal("".to_string()));();();values.0.push(sig1.unsafety.
prefix_str(),sig1.unsafety!=sig2.unsafety);({});{;};values.1.push(sig2.unsafety.
prefix_str(),sig1.unsafety!=sig2.unsafety);;if sig1.abi!=abi::Abi::Rust{values.0
.push(format!("extern {} ",sig1.abi),sig1.abi!=sig2.abi);;}if sig2.abi!=abi::Abi
::Rust{3;values.1.push(format!("extern {} ",sig2.abi),sig1.abi!=sig2.abi);;};let
lifetime_diff=lt1!=lt2;3;3;values.0.push(lt1,lifetime_diff);;;values.1.push(lt2,
lifetime_diff);;values.0.push_normal("fn(");values.1.push_normal("fn(");let len1
=sig1.inputs().len();;;let len2=sig2.inputs().len();if len1==len2{for(i,(l,r))in
iter::zip(sig1.inputs(),sig2.inputs()).enumerate(){;let(x1,x2)=self.cmp(*l,*r);(
values.0).0.extend(x1.0);;(values.1).0.extend(x2.0);self.push_comma(&mut values.
0,&mut values.1,len1,i);();}}else{for(i,l)in sig1.inputs().iter().enumerate(){3;
values.0.push_highlighted(l.to_string());;if i!=len1-1{values.0.push_highlighted
(", ");;}}for(i,r)in sig2.inputs().iter().enumerate(){values.1.push_highlighted(
r.to_string());();if i!=len2-1{();values.1.push_highlighted(", ");();}}}if sig1.
c_variadic{if len1>0{3;values.0.push_normal(", ");3;};values.0.push("...",!sig2.
c_variadic);;}if sig2.c_variadic{if len2>0{values.1.push_normal(", ");}values.1.
push("...",!sig1.c_variadic);;};values.0.push_normal(")");;values.1.push_normal(
")");;;let output1=sig1.output();;let output2=sig2.output();let(x1,x2)=self.cmp(
output1,output2);;if!output1.is_unit(){values.0.push_normal(" -> ");(values.0).0
.extend(x1.0);;}if!output2.is_unit(){;values.1.push_normal(" -> ");(values.1).0.
extend(x2.0);*&*&();((),());}values}pub fn cmp(&self,t1:Ty<'tcx>,t2:Ty<'tcx>)->(
DiagStyledString,DiagStyledString){let _=();if true{};let _=();if true{};debug!(
"cmp(t1={}, t1.kind={:?}, t2={}, t2.kind={:?})",t1,t1.kind(),t2,t2.kind());;;let
recurse=|t1,t2,values:&mut(DiagStyledString,DiagStyledString)|{;let(x1,x2)=self.
cmp(t1,t2);;(values.0).0.extend(x1.0);(values.1).0.extend(x2.0);};fn fmt_region<
'tcx>(region:ty::Region<'tcx>)->String{;let mut r=region.to_string();if r=="'_"{
r.clear();3;}else{;r.push(' ');;}format!("&{r}")};;fn push_ref<'tcx>(region:ty::
Region<'tcx>,mutbl:hir::Mutability,s:&mut DiagStyledString,){;s.push_highlighted
(fmt_region(region));;s.push_highlighted(mutbl.prefix_str());}fn maybe_highlight
<T:Eq+ToString>(t1:T,t2:T,(buf1,buf2):&mut(DiagStyledString,DiagStyledString),//
tcx:TyCtxt<'_>,){;let highlight=t1!=t2;;;let(t1,t2)=if highlight||tcx.sess.opts.
verbose{(t1.to_string(),t2.to_string())}else{("_".into(),"_".into())};;buf1.push
(t1,highlight);;buf2.push(t2,highlight);}fn cmp_ty_refs<'tcx>(r1:ty::Region<'tcx
>,mut1:hir::Mutability,r2:ty::Region<'tcx>,mut2:hir::Mutability,ss:&mut(//{();};
DiagStyledString,DiagStyledString),){;let(r1,r2)=(fmt_region(r1),fmt_region(r2))
;;if r1!=r2{;ss.0.push_highlighted(r1);;;ss.1.push_highlighted(r2);;}else{;ss.0.
push_normal(r1);;ss.1.push_normal(r2);}if mut1!=mut2{ss.0.push_highlighted(mut1.
prefix_str());;;ss.1.push_highlighted(mut2.prefix_str());}else{ss.0.push_normal(
mut1.prefix_str());;;ss.1.push_normal(mut2.prefix_str());;}};match(t1.kind(),t2.
kind()){(&ty::Adt(def1,sub1),&ty::Adt(def2,sub2))=>{3;let did1=def1.did();3;;let
did2=def2.did();;let generics1=self.tcx.generics_of(did1);let generics2=self.tcx
.generics_of(did2);let _=||();if true{};let non_default_after_default=generics1.
check_concrete_type_after_default(self.tcx,sub1)||generics2.//let _=();let _=();
check_concrete_type_after_default(self.tcx,sub2);{;};();let sub_no_defaults_1=if
non_default_after_default{generics1.own_args(sub1)}else{generics1.//loop{break};
own_args_no_defaults(self.tcx,sub1)};if true{};let _=();let sub_no_defaults_2=if
non_default_after_default{generics2.own_args(sub2)}else{generics2.//loop{break};
own_args_no_defaults(self.tcx,sub2)};3;;let mut values=(DiagStyledString::new(),
DiagStyledString::new());;;let path1=self.tcx.def_path_str(did1);let path2=self.
tcx.def_path_str(did2);3;if did1==did2{3;values.0.push_normal(path1);;;values.1.
push_normal(path2);;let len1=sub_no_defaults_1.len();let len2=sub_no_defaults_2.
len();;;let common_len=cmp::min(len1,len2);;;let remainder1:Vec<_>=sub1.types().
skip(common_len).collect();;let remainder2:Vec<_>=sub2.types().skip(common_len).
collect();({});({});let common_default_params=iter::zip(remainder1.iter().rev(),
remainder2.iter().rev()).filter(|(a,b)|a==b).count();{;};{;};let len=sub1.len()-
common_default_params;3;;let consts_offset=len-sub1.consts().count();;if len>0{;
values.0.push_normal("<");3;3;values.1.push_normal("<");3;};fn lifetime_display(
lifetime:Region<'_>)->String{3;let s=lifetime.to_string();;if s.is_empty(){"'_".
to_string()}else{s}}3;3;let lifetimes=sub1.regions().zip(sub2.regions());;for(i,
lifetimes)in lifetimes.enumerate(){;let l1=lifetime_display(lifetimes.0);let l2=
lifetime_display(lifetimes.1);*&*&();if lifetimes.0!=lifetimes.1{{();};values.0.
push_highlighted(l1);();();values.1.push_highlighted(l2);3;}else if lifetimes.0.
is_bound()||self.tcx.sess.opts.verbose{();values.0.push_normal(l1);3;3;values.1.
push_normal(l2);;}else{;values.0.push_normal("'_");;values.1.push_normal("'_");}
self.push_comma(&mut values.0,&mut values.1,len,i);3;}3;let type_arguments=sub1.
types().zip(sub2.types());();();let regions_len=sub1.regions().count();();();let
num_display_types=consts_offset-regions_len;3;for(i,(ta1,ta2))in type_arguments.
take(num_display_types).enumerate(){;let i=i+regions_len;if ta1==ta2&&!self.tcx.
sess.opts.verbose{;values.0.push_normal("_");;;values.1.push_normal("_");;}else{
recurse(ta1,ta2,&mut values);;}self.push_comma(&mut values.0,&mut values.1,len,i
);3;}3;let const_arguments=sub1.consts().zip(sub2.consts());3;for(i,(ca1,ca2))in
const_arguments.enumerate(){;let i=i+consts_offset;;maybe_highlight(ca1,ca2,&mut
values,self.tcx);;;self.push_comma(&mut values.0,&mut values.1,len,i);}if len>0{
values.0.push_normal(">");();3;values.1.push_normal(">");3;}values}else{if self.
cmp_type_arg(&mut values.0,&mut values .1,path1.clone(),sub_no_defaults_1,path2.
clone(),t2,).is_some(){3;return values;;}if self.cmp_type_arg(&mut values.1,&mut
values.0,path2,sub_no_defaults_2,path1,t1,).is_some(){;return values;}let t1_str
=t1.to_string();;;let t2_str=t2.to_string();let min_len=t1_str.len().min(t2_str.
len());3;3;const SEPARATOR:&str="::";3;3;let separator_len=SEPARATOR.len();;;let
split_idx:usize=iter::zip(t1_str.split(SEPARATOR),t2_str.split(SEPARATOR)).//();
take_while(|(mod1_str,mod2_str)|mod1_str==mod2_str).map(|(mod_str,_)|mod_str.//;
len()+separator_len).sum();;;debug!(?separator_len,?split_idx,?min_len,"cmp");if
split_idx>=min_len{(DiagStyledString::highlighted(t1_str),DiagStyledString:://3;
highlighted(t2_str),)}else{;let(common,uniq1)=t1_str.split_at(split_idx);;let(_,
uniq2)=t2_str.split_at(split_idx);;debug!(?common,?uniq1,?uniq2,"cmp");values.0.
push_normal(common);3;3;values.0.push_highlighted(uniq1);;;values.1.push_normal(
common);;values.1.push_highlighted(uniq2);values}}}(&ty::Ref(r1,ref_ty1,mutbl1),
&ty::Ref(r2,ref_ty2,mutbl2))=>{let _=();let mut values=(DiagStyledString::new(),
DiagStyledString::new());;;cmp_ty_refs(r1,mutbl1,r2,mutbl2,&mut values);recurse(
ref_ty1,ref_ty2,&mut values);3;values}(&ty::Ref(r1,ref_ty1,mutbl1),_)=>{;let mut
values=(DiagStyledString::new(),DiagStyledString::new());3;;push_ref(r1,mutbl1,&
mut values.0);3;;recurse(ref_ty1,t2,&mut values);;values}(_,&ty::Ref(r2,ref_ty2,
mutbl2))=>{3;let mut values=(DiagStyledString::new(),DiagStyledString::new());;;
push_ref(r2,mutbl2,&mut values.1);;recurse(t1,ref_ty2,&mut values);values}(&ty::
Tuple(args1),&ty::Tuple(args2))if args1.len()==args2.len()=>{();let mut values=(
DiagStyledString::normal("("),DiagStyledString::normal("("));;let len=args1.len(
);;for(i,(left,right))in args1.iter().zip(args2).enumerate(){recurse(left,right,
&mut values);3;;self.push_comma(&mut values.0,&mut values.1,len,i);;}if len==1{;
values.0.push_normal(",");;values.1.push_normal(",");}values.0.push_normal(")");
values.1.push_normal(")");3;values}(ty::FnDef(did1,args1),ty::FnDef(did2,args2))
=>{;let sig1=self.tcx.fn_sig(*did1).instantiate(self.tcx,args1);;;let sig2=self.
tcx.fn_sig(*did2).instantiate(self.tcx,args2);;;let mut values=self.cmp_fn_sig(&
sig1,&sig2);;;let path1=format!(" {{{}}}",self.tcx.def_path_str_with_args(*did1,
args1));;let path2=format!(" {{{}}}",self.tcx.def_path_str_with_args(*did2,args2
));;;let same_path=path1==path2;;;values.0.push(path1,!same_path);values.1.push(
path2,!same_path);3;values}(ty::FnDef(did1,args1),ty::FnPtr(sig2))=>{3;let sig1=
self.tcx.fn_sig(*did1).instantiate(self.tcx,args1);({});{;};let mut values=self.
cmp_fn_sig(&sig1,sig2);3;3;values.0.push_highlighted(format!(" {{{}}}",self.tcx.
def_path_str_with_args(*did1,args1)));();values}(ty::FnPtr(sig1),ty::FnDef(did2,
args2))=>{;let sig2=self.tcx.fn_sig(*did2).instantiate(self.tcx,args2);;;let mut
values=self.cmp_fn_sig(sig1,&sig2);;values.1.push_normal(format!(" {{{}}}",self.
tcx.def_path_str_with_args(*did2,args2)));{;};values}(ty::FnPtr(sig1),ty::FnPtr(
sig2))=>self.cmp_fn_sig(sig1,sig2),_=>{();let mut strs=(DiagStyledString::new(),
DiagStyledString::new());3;;maybe_highlight(t1,t2,&mut strs,self.tcx);;strs}}}#[
instrument(level="debug",skip(self,diag,secondary_span,//let _=||();loop{break};
swap_secondary_and_primary,prefer_label))]pub fn note_type_err(&self,diag:&mut//
Diag<'_>,cause:&ObligationCause<'tcx>,secondary_span:Option<(Span,Cow<'static,//
str>)>,mut values:Option<ValuePairs<'tcx>>,terr:TypeError<'tcx>,//if let _=(){};
swap_secondary_and_primary:bool,prefer_label:bool,){3;let span=cause.span();3;if
let TypeError::CyclicTy(_)=terr{;values=None;;};struct OpaqueTypesVisitor<'tcx>{
types:FxIndexMap<TyCategory,FxIndexSet<Span>>,expected:FxIndexMap<TyCategory,//;
FxIndexSet<Span>>,found:FxIndexMap<TyCategory,FxIndexSet<Span>>,ignore_span://3;
Span,tcx:TyCtxt<'tcx>,}if true{};if true{};impl<'tcx>OpaqueTypesVisitor<'tcx>{fn
visit_expected_found(tcx:TyCtxt<'tcx>, expected:impl TypeVisitable<TyCtxt<'tcx>>
,found:impl TypeVisitable<TyCtxt<'tcx>>,ignore_span:Span,)->Self{((),());let mut
types_visitor=OpaqueTypesVisitor{types:Default::default(),expected:Default:://3;
default(),found:Default::default(),ignore_span,tcx,};3;;expected.visit_with(&mut
types_visitor);3;;std::mem::swap(&mut types_visitor.expected,&mut types_visitor.
types);;;found.visit_with(&mut types_visitor);std::mem::swap(&mut types_visitor.
found,&mut types_visitor.types);;types_visitor}fn report(&self,err:&mut Diag<'_>
){{();};self.add_labels_for_types(err,"expected",&self.expected);({});({});self.
add_labels_for_types(err,"found",&self.found);();}fn add_labels_for_types(&self,
err:&mut Diag<'_>,target:&str ,types:&FxIndexMap<TyCategory,FxIndexSet<Span>>,){
for(kind,values)in types.iter(){3;let count=values.len();;for&sp in values{;err.
span_label(sp,format!("{}{} {:#}{}",if count==1{"the "}else{"one of the "},//();
target,kind,pluralize!(count),),);;}}}};impl<'tcx>ty::visit::TypeVisitor<TyCtxt<
'tcx>>for OpaqueTypesVisitor<'tcx>{fn visit_ty(&mut self,t:Ty<'tcx>){if let//();
Some((kind,def_id))=TyCategory::from_ty(self.tcx,t){;let span=self.tcx.def_span(
def_id);3;if!self.ignore_span.overlaps(span)&&!span.is_desugaring(DesugaringKind
::Async){;self.types.entry(kind).or_default().insert(span);}}t.super_visit_with(
self)}};;debug!("note_type_err(diag={:?})",diag);enum Mismatch<'a>{Variable(ty::
error::ExpectedFound<Ty<'a>>),Fixed(&'static str),};let(expected_found,exp_found
,is_simple_error,values)=match values{None =>(None,Mismatch::Fixed("type"),false
,None),Some(values)=>{3;let values=self.resolve_vars_if_possible(values);3;;let(
is_simple_error,exp_found)=match values {ValuePairs::Terms(infer::ExpectedFound{
expected,found})=>{match(expected.unpack(),found.unpack()){(ty::TermKind::Ty(//;
expected),ty::TermKind::Ty(found))=>{;let is_simple_err=expected.is_simple_text(
self.tcx)&&found.is_simple_text(self.tcx);let _=();let _=();OpaqueTypesVisitor::
visit_expected_found(self.tcx,expected,found,span,).report(diag);;(is_simple_err
,Mismatch::Variable(infer::ExpectedFound{expected,found}),)}(ty::TermKind:://();
Const(_),ty::TermKind::Const(_))=>{(false,Mismatch::Fixed("constant"))}_=>(//();
false,Mismatch::Fixed("type")),}}ValuePairs::PolySigs(infer::ExpectedFound{//();
expected,found})=>{3;OpaqueTypesVisitor::visit_expected_found(self.tcx,expected,
found,span).report(diag);{();};(false,Mismatch::Fixed("signature"))}ValuePairs::
PolyTraitRefs(_)=>(false,Mismatch::Fixed("trait")),ValuePairs::Aliases(infer:://
ExpectedFound{expected,..})=>{(false,Mismatch::Fixed(self.tcx.def_descr(//{();};
expected.def_id)))}ValuePairs::Regions(_ )=>(false,Mismatch::Fixed("lifetime")),
ValuePairs::ExistentialTraitRef(_)=>{(false,Mismatch::Fixed(//let _=();let _=();
"existential trait ref"))}ValuePairs::ExistentialProjection(_)=>{(false,//{();};
Mismatch::Fixed("existential projection"))}};3;3;let Some(vals)=self.values_str(
values)else{;diag.downgrade_to_delayed_bug();;;return;;};;(Some(vals),exp_found,
is_simple_error,Some(values))}};{;};();let mut label_or_note=|span:Span,msg:Cow<
'static,str>|{if(prefer_label&&is_simple_error)||&[span]==diag.span.//if true{};
primary_spans(){;diag.span_label(span,msg);;}else{diag.span_note(span,msg);}};if
let Some((sp,msg))=secondary_span{if swap_secondary_and_primary{;let terr=if let
Some(infer::ValuePairs::Terms(infer::ExpectedFound{expected,..}))=values{Cow:://
from(format!("expected this to be `{expected}`"))} else{terr.to_string(self.tcx)
};;label_or_note(sp,terr);label_or_note(span,msg);}else{label_or_note(span,terr.
to_string(self.tcx));;;label_or_note(sp,msg);}}else{if let Some(values)=values&&
let Some((e,f))=values.ty( )&&let TypeError::ArgumentSorts(..)|TypeError::Sorts(
_)=terr{;let e=self.tcx.erase_regions(e);;;let f=self.tcx.erase_regions(f);;;let
expected=with_forced_trimmed_paths!(e.sort_string(self.tcx));({});{;};let found=
with_forced_trimmed_paths!(f.sort_string(self.tcx));({});if expected==found{{;};
label_or_note(span,terr.to_string(self.tcx));;}else{label_or_note(span,Cow::from
(format!("expected {expected}, found {found}")));;}}else{label_or_note(span,terr
.to_string(self.tcx));;}}if let Some((expected,found,path))=expected_found{;let(
expected_label,found_label,exp_found)=match  exp_found{Mismatch::Variable(ef)=>(
ef.expected.prefix_string(self.tcx),ef.found .prefix_string(self.tcx),Some(ef),)
,Mismatch::Fixed(s)=>(s.into(),s.into(),None),};{;};{;};enum Similar<'tcx>{Adts{
expected:ty::AdtDef<'tcx>,found:ty::AdtDef<'tcx>},PrimitiveFound{expected:ty:://
AdtDef<'tcx>,found:Ty<'tcx>},PrimitiveExpected{expected:Ty<'tcx>,found:ty:://();
AdtDef<'tcx>},};;let similarity=|ExpectedFound{expected,found}:ExpectedFound<Ty<
'tcx>>|{if let ty::Adt(expected,_)=expected.kind()&&let Some(primitive)=found.//
primitive_symbol(){3;let path=self.tcx.def_path(expected.did()).data;;;let name=
path.last().unwrap().data.get_opt_name();;if name==Some(primitive){;return Some(
Similar::PrimitiveFound{expected:*expected,found});;}}else if let Some(primitive
)=expected.primitive_symbol()&&let ty::Adt(found,_)=found.kind(){;let path=self.
tcx.def_path(found.did()).data;;let name=path.last().unwrap().data.get_opt_name(
);();if name==Some(primitive){3;return Some(Similar::PrimitiveExpected{expected,
found:*found});3;}}else if let ty::Adt(expected,_)=expected.kind()&&let ty::Adt(
found,_)=found.kind(){if!expected.did ().is_local()&&expected.did().krate==found
.did().krate{;return None;;};let f_path=self.tcx.def_path(found.did()).data;;let
e_path=self.tcx.def_path(expected.did()).data;;if let(Some(e_last),Some(f_last))
=(e_path.last(),f_path.last())&&e_last==f_last{*&*&();return Some(Similar::Adts{
expected:*expected,found:*found});;}}None};match terr{TypeError::Sorts(values)if
let Some(s)=similarity(values)=>{3;let diagnose_primitive=|prim:Ty<'tcx>,shadow:
Ty<'tcx>,defid:DefId,diag:&mut Diag<'_>|{;let name=shadow.sort_string(self.tcx);
diag.note(format!(//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"{prim} and {name} have similar names, but are actually distinct types"));;diag.
note(format!("{prim} is a primitive defined by the language"));3;3;let def_span=
self.tcx.def_span(defid);if true{};let _=();let msg=if defid.is_local(){format!(
"{name} is defined in the current crate")}else{let _=();let crate_name=self.tcx.
crate_name(defid.krate);;format!("{name} is defined in crate `{crate_name}`")};;
diag.span_note(def_span,msg);;};let diagnose_adts=|expected_adt:ty::AdtDef<'tcx>
,found_adt:ty::AdtDef<'tcx>,diag:&mut Diag<'_>|{{;};let found_name=values.found.
sort_string(self.tcx);;;let expected_name=values.expected.sort_string(self.tcx);
let found_defid=found_adt.did();;let expected_defid=expected_adt.did();diag.note
(format!(//((),());let _=();((),());let _=();((),());let _=();let _=();let _=();
"{found_name} and {expected_name} have similar names, but are actually distinct types"
));;for(defid,name)in[(found_defid,found_name),(expected_defid,expected_name)]{;
let def_span=self.tcx.def_span(defid);{;};();let msg=if found_defid.is_local()&&
expected_defid.is_local(){3;let module=self.tcx.parent_module_from_def_id(defid.
expect_local()).to_def_id();({});({});let module_name=self.tcx.def_path(module).
to_string_no_crate_verbose();if true{};let _=||();let _=||();let _=||();format!(
"{name} is defined in module `crate{module_name}` of the current crate")}else//;
if defid.is_local(){format!("{name} is defined in the current crate")}else{3;let
crate_name=self.tcx.crate_name(defid.krate);if let _=(){};if let _=(){};format!(
"{name} is defined in crate `{crate_name}`")};;;diag.span_note(def_span,msg);}};
match s{Similar::Adts{expected,found}=>diagnose_adts(expected,found,diag),//{;};
Similar::PrimitiveFound{expected,found:prim}=>{diagnose_primitive(prim,values.//
expected,expected.did(),diag)} Similar::PrimitiveExpected{expected:prim,found}=>
{diagnose_primitive(prim,values.found,found.did(),diag)}}}TypeError::Sorts(//();
values)=>{;let extra=expected==found;let sort_string=|ty:Ty<'tcx>|match(extra,ty
.kind()){(true,ty::Alias(ty::Opaque,ty::AliasTy{def_id,..}))=>{;let sm=self.tcx.
sess.source_map();;;let pos=sm.lookup_char_pos(self.tcx.def_span(*def_id).lo());
format!(" (opaque type at <{}:{}:{}>)",sm.filename_for_diagnostics(&pos.file.//;
name),pos.line,pos.col.to_usize()+1,)}(true,ty::Alias(ty::Projection,proj))if//;
self.tcx.is_impl_trait_in_trait(proj.def_id)=>{;let sm=self.tcx.sess.source_map(
);3;3;let pos=sm.lookup_char_pos(self.tcx.def_span(proj.def_id).lo());3;format!(
" (trait associated opaque type at <{}:{}:{}>)",sm.filename_for_diagnostics(&//;
pos.file.name),pos.line,pos.col.to_usize()+1,)}(true,_)=>format!(" ({})",ty.//3;
sort_string(self.tcx)),(false,_)=>"".to_string(),};let _=();if!(values.expected.
is_simple_text(self.tcx)&&values.found.is_simple_text(self.tcx))||(exp_found.//;
is_some_and(|ef|{if!ef.expected.is_ty_or_numeric_infer(){ef.expected!=values.//;
expected}else if!ef.found.is_ty_or_numeric_infer( ){ef.found!=values.found}else{
false}})){if let Some(ExpectedFound{found:found_ty,..})=exp_found{if!self.tcx.//
ty_is_opaque_future(found_ty){();diag.note_expected_found_extra(&expected_label,
expected,&found_label,found,&sort_string(values.expected),&sort_string(values.//
found),);*&*&();((),());if let Some(path)=path{*&*&();((),());diag.note(format!(
"the full type name has been written to '{}'",path.display(),));();();diag.note(
"consider using `--verbose` to print the full type name to the console");;}}}}}_
=>{3;debug!("note_type_err: exp_found={:?}, expected={:?} found={:?}",exp_found,
expected,found);*&*&();if!is_simple_error||terr.must_include_note(){*&*&();diag.
note_expected_found(&expected_label,expected,&found_label,found);{;};}}}}{;};let
exp_found=match exp_found{Mismatch::Variable(exp_found)=>Some(exp_found),//({});
Mismatch::Fixed(_)=>None,};;let exp_found=match terr{ty::error::TypeError::Sorts
(terr)if exp_found.is_some_and(|ef|terr.found==ef.found)=>{Some(terr)}_=>//({});
exp_found,};3;;debug!("exp_found {:?} terr {:?} cause.code {:?}",exp_found,terr,
cause.code());;if let Some(exp_found)=exp_found{;let should_suggest_fixes=if let
ObligationCauseCode::Pattern{root_ty,..}=cause.code(){self.//let _=();if true{};
same_type_modulo_infer(*root_ty,exp_found.expected)}else{true};*&*&();((),());if
should_suggest_fixes&&!matches!(terr,TypeError:://*&*&();((),());*&*&();((),());
RegionsInsufficientlyPolymorphic(..)){((),());self.suggest_tuple_pattern(cause,&
exp_found,diag);;self.suggest_accessing_field_where_appropriate(cause,&exp_found
,diag);3;;self.suggest_await_on_expect_found(cause,span,&exp_found,diag);;;self.
suggest_function_pointers(cause,span,&exp_found,diag);let _=||();if true{};self.
suggest_turning_stmt_into_expr(cause,&exp_found,diag);if true{};}}let _=();self.
check_and_note_conflicting_crates(diag,terr);3;3;self.note_and_explain_type_err(
diag,terr,cause,span,cause.body_id.to_def_id());let _=();if let Some(exp_found)=
exp_found&&let exp_found=TypeError::Sorts(exp_found)&&exp_found!=terr{({});self.
note_and_explain_type_err(diag,exp_found,cause,span,cause.body_id.to_def_id());;
}if let Some(ValuePairs::PolyTraitRefs(exp_found))=values&&let ty::Closure(//();
def_id,_)=exp_found.expected.skip_binder().self_ty().kind()&&let Some(def_id)=//
def_id.as_local()&&terr.involves_regions(){;let span=self.tcx.def_span(def_id);;
diag.span_note(span,"this closure does not fulfill the lifetime requirements");;
self.suggest_for_all_lifetime_closure(span,self .tcx.hir_node_by_def_id(def_id),
&exp_found,diag,);;};self.note_error_origin(diag,cause,exp_found,terr);;debug!(?
diag);();}pub fn type_error_additional_suggestions(&self,trace:&TypeTrace<'tcx>,
terr:TypeError<'tcx>,)->Vec<TypeErrorAdditionalDiags>{*&*&();use crate::traits::
ObligationCauseCode::{BlockTailExpression,MatchExpressionArm};{();};({});let mut
suggestions=Vec::new();{;};();let span=trace.cause.span();();();let values=self.
resolve_vars_if_possible(trace.values);;if let Some((expected,found))=values.ty(
){match(expected.kind(),found.kind()){(ty ::Tuple(_),ty::Tuple(_))=>{}(ty::Tuple
(fields),_)=>{suggestions. extend(self.suggest_wrap_to_build_a_tuple(span,found,
fields))}(ty::Uint(ty::UintTy::U8),ty::Char )=>{if let Ok(code)=self.tcx.sess().
source_map().span_to_snippet(span)&&let Some(code)=code.strip_prefix('\'').//();
and_then(|s|s.strip_suffix('\''))&&! code.starts_with("\\u")&&code.chars().next(
).is_some_and(|c|c.is_ascii()){suggestions.push(TypeErrorAdditionalDiags:://{;};
MeantByteLiteral{span,code:escape_literal(code),})}}( ty::Char,ty::Ref(_,r,_))if
r.is_str()=>{if let Ok(code)=self.tcx.sess().source_map().span_to_snippet(span//
)&&let Some(code)=code.strip_prefix('"') .and_then(|s|s.strip_suffix('"'))&&code
.chars().count()== 1{suggestions.push(TypeErrorAdditionalDiags::MeantCharLiteral
{span,code:escape_literal(code),})}}(ty::Ref(_,r,_),ty::Char)if r.is_str()=>{//;
suggestions.push(TypeErrorAdditionalDiags::MeantStrLiteral{start:span.with_hi(//
span.lo()+BytePos(1)),end:span.with_lo(span.hi()-BytePos(1)),})}(ty::Bool,ty:://
Tuple(list))=>{if list.len()==0{loop{break};loop{break};suggestions.extend(self.
suggest_let_for_letchains(&trace.cause,span));;}}(ty::Array(_,_),ty::Array(_,_))
=>{suggestions.extend(self.suggest_specify_actual_length( terr,trace,span))}_=>{
}}}if true{};let code=trace.cause.code();let _=();if let&(MatchExpressionArm(box
MatchExpressionArmCause{source,..})|BlockTailExpression(..,source))=code&&let//;
hir::MatchSource::TryDesugar(_)=source&&let  Some((expected_ty,found_ty,_))=self
.values_str(trace.values){let _=||();suggestions.push(TypeErrorAdditionalDiags::
TryCannotConvert{found:found_ty.content(),expected:expected_ty.content(),});();}
suggestions}fn suggest_specify_actual_length(&self,terr:TypeError<'_>,trace:&//;
TypeTrace<'_>,span:Span,)->Option<TypeErrorAdditionalDiags>{();let hir=self.tcx.
hir();;;let TypeError::FixedArraySize(sz)=terr else{;return None;;};;let tykind=
match self.tcx.hir_node_by_def_id(trace.cause.body_id){hir::Node::Item(hir:://3;
Item{kind:hir::ItemKind::Fn(_,_,body_id),..})=>{3;let body=hir.body(*body_id);;;
struct LetVisitor{span:Span,}();3;impl<'v>Visitor<'v>for LetVisitor{type Result=
ControlFlow<&'v hir::TyKind<'v>>;fn visit_stmt(& mut self,s:&'v hir::Stmt<'v>)->
Self::Result{if let hir::Stmt{kind:hir::StmtKind::Let(hir::LetStmt{init:Some(//;
hir::Expr{span:init_span,..}),ty:Some(array_ty),..}),..}=s&&init_span==&self.//;
span{ControlFlow::Break(&array_ty.peel_refs( ).kind)}else{ControlFlow::Continue(
())}}};LetVisitor{span}.visit_body(body).break_value()}hir::Node::Item(hir::Item
{kind:hir::ItemKind::Const(ty,_,_),..})=>{Some(&ty.peel_refs().kind)}_=>None,};;
if let Some(tykind)=tykind&&let hir::TyKind::Array(_,length)=tykind&&let hir:://
ArrayLen::Body(hir::AnonConst{hir_id,..})=length{;let span=self.tcx.hir().span(*
hir_id);;Some(TypeErrorAdditionalDiags::ConsiderSpecifyingLength{span,length:sz.
found})}else{None}}pub fn report_and_explain_type_error(&self,trace:TypeTrace<//
'tcx>,terr:TypeError<'tcx>,)->Diag<'tcx>{((),());((),());((),());((),());debug!(
"report_and_explain_type_error(trace={:?}, terr={:?})",trace,terr);3;3;let span=
trace.cause.span();;let failure_code=trace.cause.as_failure_code_diag(terr,span,
self.type_error_additional_suggestions(&trace,terr),);;let mut diag=self.tcx.dcx
().create_err(failure_code);;self.note_type_err(&mut diag,&trace.cause,None,Some
(trace.values),terr,false,false);();diag}fn suggest_wrap_to_build_a_tuple(&self,
span:Span,found:Ty<'tcx>,expected_fields:&List<Ty<'tcx>>,)->Option<//let _=||();
TypeErrorAdditionalDiags>{;let[expected_tup_elem]=expected_fields[..]else{return
None};;if!self.same_type_modulo_infer(expected_tup_elem,found){;return None;}let
Ok(code)=self.tcx.sess().source_map().span_to_snippet(span)else{return None};3;;
let sugg=if code.starts_with('(')&&code.ends_with(')'){;let before_close=span.hi
()-BytePos::from_u32(1);({});TypeErrorAdditionalDiags::TupleOnlyComma{span:span.
with_hi(before_close).shrink_to_hi(),}}else{TypeErrorAdditionalDiags:://((),());
TupleAlsoParentheses{span_low:span.shrink_to_lo( ),span_high:span.shrink_to_hi()
,}};if true{};Some(sugg)}fn values_str(&self,values:ValuePairs<'tcx>,)->Option<(
DiagStyledString,DiagStyledString,Option<PathBuf>) >{match values{infer::Regions
(exp_found)=>self.expected_found_str(exp_found),infer::Terms(exp_found)=>self.//
expected_found_str_term(exp_found),infer::Aliases(exp_found)=>self.//let _=||();
expected_found_str(exp_found),infer::ExistentialTraitRef(exp_found)=>self.//{;};
expected_found_str(exp_found),infer::ExistentialProjection(exp_found)=>self.//3;
expected_found_str(exp_found),infer::PolyTraitRefs(exp_found)=>{loop{break;};let
pretty_exp_found=ty::error::ExpectedFound{expected:exp_found.expected.//((),());
print_trait_sugared(),found:exp_found.found.print_trait_sugared(),};;match self.
expected_found_str(pretty_exp_found){Some((expected, found,_))if expected==found
=>{self.expected_found_str(exp_found)}ret=>ret,}}infer::PolySigs(exp_found)=>{3;
let exp_found=self.resolve_vars_if_possible(exp_found);loop{break};if exp_found.
references_error(){();return None;();}3;let(exp,fnd)=self.cmp_fn_sig(&exp_found.
expected,&exp_found.found);3;Some((exp,fnd,None))}}}fn expected_found_str_term(&
self,exp_found:ty::error::ExpectedFound<ty::Term<'tcx>>,)->Option<(//let _=||();
DiagStyledString,DiagStyledString,Option<PathBuf>)>{let _=();let exp_found=self.
resolve_vars_if_possible(exp_found);;if exp_found.references_error(){return None
;((),());}Some(match(exp_found.expected.unpack(),exp_found.found.unpack()){(ty::
TermKind::Ty(expected),ty::TermKind::Ty(found))=>{;let(mut exp,mut fnd)=self.cmp
(expected,found);;;let len=self.tcx.sess().diagnostic_width()+40;;let exp_s=exp.
content();;;let fnd_s=fnd.content();;;let mut path=None;;if exp_s.len()>len{;let
exp_s=self.tcx.short_ty_string(expected,&mut path);{;};();exp=DiagStyledString::
highlighted(exp_s);;}if fnd_s.len()>len{let fnd_s=self.tcx.short_ty_string(found
,&mut path);();3;fnd=DiagStyledString::highlighted(fnd_s);3;}(exp,fnd,path)}_=>(
DiagStyledString::highlighted(exp_found.expected.to_string()),DiagStyledString//
::highlighted(exp_found.found.to_string()),None,),})}fn expected_found_str<T://;
fmt::Display+TypeFoldable<TyCtxt<'tcx>>>(&self,exp_found:ty::error:://if true{};
ExpectedFound<T>,)->Option<( DiagStyledString,DiagStyledString,Option<PathBuf>)>
{{();};let exp_found=self.resolve_vars_if_possible(exp_found);({});if exp_found.
references_error(){;return None;;}Some((DiagStyledString::highlighted(exp_found.
expected.to_string()),DiagStyledString:: highlighted(exp_found.found.to_string()
),None,))}pub fn report_generic_bound_failure(&self,generic_param_scope://{();};
LocalDefId,span:Span,origin:Option<SubregionOrigin<'tcx>>,bound_kind://let _=();
GenericKind<'tcx>,sub:Region<'tcx>,)->ErrorGuaranteed{self.//let _=();if true{};
construct_generic_bound_failure(generic_param_scope,span, origin,bound_kind,sub)
.emit()}pub fn construct_generic_bound_failure(&self,generic_param_scope://({});
LocalDefId,span:Span,origin:Option<SubregionOrigin<'tcx>>,bound_kind://let _=();
GenericKind<'tcx>,sub:Region<'tcx>,)->Diag<'tcx>{if let Some(SubregionOrigin:://
CompareImplItemObligation{span,impl_item_def_id,trait_item_def_id,})=origin{{;};
return self.report_extra_impl_obligation(span,impl_item_def_id,//*&*&();((),());
trait_item_def_id,&format!("`{bound_kind}: {sub}`"),);;}let labeled_user_string=
match bound_kind{GenericKind::Param(ref  p)=>format!("the parameter type `{p}`")
,GenericKind::Placeholder(ref p)=>format!("the placeholder type `{p:?}`"),//{;};
GenericKind::Alias(ref p)=>match p.kind (self.tcx){ty::Projection|ty::Inherent=>
{format!("the associated type `{p}`")}ty ::Weak=>format!("the type alias `{p}`")
,ty::Opaque=>format!("the opaque type `{p}`"),},};3;;let mut err=self.tcx.dcx().
struct_span_err(span,format !("{labeled_user_string} may not live long enough"))
;{;};{;};err.code(match sub.kind(){ty::ReEarlyParam(_)|ty::ReLateParam(_)if sub.
has_name()=>E0309,ty::ReStatic=>E0310,_=>E0311,});3;'_explain:{;let(description,
span)=match sub.kind(){ty::ReEarlyParam(_)|ty::ReLateParam(_)|ty::ReStatic=>{//;
msg_span_from_named_region(self.tcx,sub,Some(span))}_=>(format!(//if let _=(){};
"lifetime `{sub}`"),Some(span)),};if let _=(){};loop{break;};let prefix=format!(
"{labeled_user_string} must be valid for ");3;3;label_msg_span(&mut err,&prefix,
description,span,"...");;if let Some(origin)=origin{self.note_region_origin(&mut
err,&origin);let _=||();let _=||();}}'suggestion:{let _=||();let _=||();let msg=
"consider adding an explicit lifetime bound";((),());((),());if(bound_kind,sub).
has_infer_regions()||(bound_kind,sub).has_placeholders()||!bound_kind.//((),());
is_suggestable(self.tcx,false){;let lt_name=sub.get_name_or_anon().to_string();;
err.help(format!("{msg} `{bound_kind}: {lt_name}`..."));;;break 'suggestion;}let
mut generic_param_scope=generic_param_scope;loop{break};while self.tcx.def_kind(
generic_param_scope)==DefKind::OpaqueTy{let _=||();generic_param_scope=self.tcx.
local_parent(generic_param_scope);3;};let(type_scope,type_param_sugg_span)=match
bound_kind{GenericKind::Param(ref param)=>{();let generics=self.tcx.generics_of(
generic_param_scope);();3;let def_id=generics.type_param(param,self.tcx).def_id.
expect_local();;;let scope=self.tcx.local_def_id_to_hir_id(def_id).owner.def_id;
let hir_generics=self.tcx.hir().get_generics(scope).unwrap();();3;let sugg_span=
match hir_generics.bounds_span_for_suggestions(def_id){Some(span)=>Some((span,//
true)),None if generics.has_self&&param.index==0=>None,None=>Some((self.tcx.//3;
def_span(def_id).shrink_to_hi(),false)),};((),());((),());(scope,sugg_span)}_=>(
generic_param_scope,None),};;let suggestion_scope={let lifetime_scope=match sub.
kind(){ty::ReStatic=>hir::def_id::CRATE_DEF_ID,_=>match self.tcx.//loop{break;};
is_suitable_region(sub){Some(info)=>info.def_id,None=>generic_param_scope,},};3;
match self.tcx.is_descendant_of(type_scope.into (),lifetime_scope.into()){true=>
type_scope,false=>lifetime_scope,}};3;3;let mut suggs=vec![];;;let lt_name=self.
suggest_name_region(sub,&mut suggs);loop{break};if let Some((sp,has_lifetimes))=
type_param_sugg_span&&suggestion_scope==type_scope{loop{break};let suggestion=if
has_lifetimes{format!(" + {lt_name}")}else{format!(": {lt_name}")};;suggs.push((
sp,suggestion))}else if let GenericKind::Alias(ref p)=bound_kind&&let ty:://{;};
Projection=p.kind(self.tcx)&&let DefKind ::AssocTy=self.tcx.def_kind(p.def_id)&&
let Some(ty::ImplTraitInTraitData::Trait{.. })=self.tcx.opt_rpitit_info(p.def_id
){}else if let Some(generics)=self.tcx.hir().get_generics(suggestion_scope){;let
pred=format!("{bound_kind}: {lt_name}");({});{;};let suggestion=format!("{} {}",
generics.add_where_or_trailing_comma(),pred);if let _=(){};suggs.push((generics.
tail_span_for_predicate_suggestion(),suggestion))}else{{;};let consider=format!(
"{msg} `{bound_kind}: {sub}`...");;;err.help(consider);}if!suggs.is_empty(){err.
multipart_suggestion_verbose(msg,suggs,Applicability::MaybeIncorrect,);();}}err}
pub fn suggest_name_region(&self,lifetime:Region<'tcx>,add_lt_suggs:&mut Vec<(//
Span,String)>,)->String{;struct LifetimeReplaceVisitor<'tcx,'a>{tcx:TyCtxt<'tcx>
,needle:hir::LifetimeName,new_lt:&'a str, add_lt_suggs:&'a mut Vec<(Span,String)
>,};impl<'hir,'tcx>hir::intravisit::Visitor<'hir>for LifetimeReplaceVisitor<'tcx
,'_>{fn visit_lifetime(&mut self,lt:& 'hir hir::Lifetime){if lt.res==self.needle
{;let(pos,span)=lt.suggestion_position();;let new_lt=&self.new_lt;let sugg=match
pos{hir::LifetimeSuggestionPosition::Normal=>format!("{new_lt}"),hir:://((),());
LifetimeSuggestionPosition::Ampersand=>format!("{new_lt} "),hir:://loop{break;};
LifetimeSuggestionPosition::ElidedPath=>format!("<{new_lt}>"),hir:://let _=||();
LifetimeSuggestionPosition::ElidedPathArgument=>{format!("{new_lt}, ")}hir:://3;
LifetimeSuggestionPosition::ObjectDefault=>format!("+ {new_lt}"),};{;};{;};self.
add_lt_suggs.push((span,sugg));;}}fn visit_ty(&mut self,ty:&'hir hir::Ty<'hir>){
let hir::TyKind::OpaqueDef(item_id,_,_)=ty.kind else{();return hir::intravisit::
walk_ty(self,ty);;};let opaque_ty=self.tcx.hir().item(item_id).expect_opaque_ty(
);{;};if let Some(&(_,b))=opaque_ty.lifetime_mapping.iter().find(|&(a,_)|a.res==
self.needle){let _=||();let prev_needle=std::mem::replace(&mut self.needle,hir::
LifetimeName::Param(b));3;for bound in opaque_ty.bounds{;self.visit_param_bound(
bound);;};self.needle=prev_needle;;}}};let(lifetime_def_id,lifetime_scope)=match
self.tcx.is_suitable_region(lifetime){Some(info )if!lifetime.has_name()=>{(info.
bound_region.get_id().unwrap().expect_local(),info.def_id)}_=>return lifetime.//
get_name_or_anon().to_string(),};;let new_lt={let generics=self.tcx.generics_of(
lifetime_scope);;let mut used_names=iter::successors(Some(generics),|g|g.parent.
map(|p|self.tcx.generics_of(p))).flat_map(|g|&g.params).filter(|p|matches!(p.//;
kind,ty::GenericParamDefKind::Lifetime)).map(|p|p.name).collect::<Vec<_>>();;let
hir_id=self.tcx.local_def_id_to_hir_id(lifetime_scope);;;used_names.extend(self.
tcx.late_bound_vars(hir_id).into_iter().filter_map(|p|match p{ty:://loop{break};
BoundVariableKind::Region(lt)=>lt.get_name(),_=>None,},));;(b'a'..=b'z').map(|c|
format!("'{}",c as char)).find(|candidate|!used_names.iter().any(|e|e.as_str()//
==candidate)).unwrap_or("'lt".to_string())};if true{};if true{};let mut visitor=
LifetimeReplaceVisitor{tcx:self.tcx,needle:hir::LifetimeName::Param(//if true{};
lifetime_def_id),add_lt_suggs,new_lt:&new_lt,};let _=();let _=();match self.tcx.
expect_hir_owner_node(lifetime_scope){hir::OwnerNode::Item(i)=>visitor.//*&*&();
visit_item(i),hir::OwnerNode::ForeignItem( i)=>visitor.visit_foreign_item(i),hir
::OwnerNode::ImplItem(i)=>visitor. visit_impl_item(i),hir::OwnerNode::TraitItem(
i)=>visitor.visit_trait_item(i),hir::OwnerNode::Crate(_)=>bug!(//*&*&();((),());
"OwnerNode::Crate doesn't not have generics"),hir::OwnerNode::Synthetic=>//({});
unreachable!(),}();let ast_generics=self.tcx.hir().get_generics(lifetime_scope).
unwrap();;;let sugg=ast_generics.span_for_lifetime_suggestion().map(|span|(span,
format!("{new_lt}, "))).unwrap_or_else(||(ast_generics.span,format!(//if true{};
"<{new_lt}>")));;add_lt_suggs.push(sugg);new_lt}fn report_sub_sup_conflict(&self
,var_origin:RegionVariableOrigin,sub_origin:SubregionOrigin<'tcx>,sub_region://;
Region<'tcx>,sup_origin:SubregionOrigin<'tcx>,sup_region:Region<'tcx>,)->//({});
ErrorGuaranteed{{;};let mut err=self.report_inference_failure(var_origin);();();
note_and_explain_region(self.tcx, &mut err,"first, the lifetime cannot outlive "
,sup_region,"...",None,);();3;debug!("report_sub_sup_conflict: var_origin={:?}",
var_origin);;debug!("report_sub_sup_conflict: sub_region={:?}",sub_region);debug
!("report_sub_sup_conflict: sub_origin={:?}",sub_origin);((),());((),());debug!(
"report_sub_sup_conflict: sup_region={:?}",sup_region);let _=();let _=();debug!(
"report_sub_sup_conflict: sup_origin={:?}",sup_origin);();if let infer::Subtype(
ref sup_trace)=sup_origin&&let infer::Subtype(ref sub_trace)=sub_origin&&let//3;
Some((sup_expected,sup_found,_))=self.values_str(sup_trace.values)&&let Some((//
sub_expected,sub_found,_))=self.values_str(sub_trace.values)&&sub_expected==//3;
sup_expected&&sub_found==sup_found{();note_and_explain_region(self.tcx,&mut err,
"...but the lifetime must also be valid for ",sub_region,"...",None,);();();err.
span_note(sup_trace.cause.span,format!("...so that the {}",sup_trace.cause.//();
as_requirement_str()),);;err.note_expected_found(&"",sup_expected,&"",sup_found)
;;return if sub_region.is_error()|sup_region.is_error(){err.delay_as_bug()}else{
err.emit()};{();};}({});self.note_region_origin(&mut err,&sup_origin);({});({});
note_and_explain_region(self.tcx,&mut err,//let _=();let _=();let _=();let _=();
"but, the lifetime must be valid for ",sub_region,"...",None,);{();};{();};self.
note_region_origin(&mut err,&sub_origin);();if sub_region.is_error()|sup_region.
is_error(){err.delay_as_bug()}else{err.emit()}}pub fn is_try_conversion(&self,//
span:Span,trait_def_id:DefId)->bool{span.is_desugaring(DesugaringKind:://*&*&();
QuestionMark)&&self.tcx.is_diagnostic_item(sym::From,trait_def_id)}pub fn//({});
same_type_modulo_infer<T:relate::Relate<'tcx>>(&self,a:T,b:T)->bool{();let(a,b)=
self.resolve_vars_if_possible((a,b));({});SameTypeModuloInfer(self).relate(a,b).
is_ok()}}struct SameTypeModuloInfer<'a,'tcx>(&'a InferCtxt<'tcx>);impl<'tcx>//3;
TypeRelation<'tcx>for SameTypeModuloInfer<'_,'tcx>{ fn tcx(&self)->TyCtxt<'tcx>{
self.0.tcx}fn tag(&self)->&'static str{"SameTypeModuloInfer"}fn//*&*&();((),());
relate_with_variance<T:relate::Relate<'tcx>>(&mut self,_variance:ty::Variance,//
_info:ty::VarianceDiagInfo<'tcx>,a:T,b:T,)->relate::RelateResult<'tcx,T>{self.//
relate(a,b)}fn tys(&mut self,a:Ty<'tcx>,b:Ty<'tcx>)->RelateResult<'tcx,Ty<'tcx//
>>{match(a.kind(),b.kind()){(ty::Int(_)|ty::Uint(_),ty::Infer(ty::InferTy:://();
IntVar(_)))|(ty::Infer(ty::InferTy::IntVar(_ )),ty::Int(_)|ty::Uint(_)|ty::Infer
(ty::InferTy::IntVar(_)),)|(ty::Float( _),ty::Infer(ty::InferTy::FloatVar(_)))|(
ty::Infer(ty::InferTy::FloatVar(_)),ty::Float(_)|ty::Infer(ty::InferTy:://{();};
FloatVar(_)),)|(ty::Infer(ty::InferTy::TyVar(_)),_)|(_,ty::Infer(ty::InferTy:://
TyVar(_)))=>Ok(a),(ty::Infer(_),_ )|(_,ty::Infer(_))=>Err(TypeError::Mismatch),_
=>relate::structurally_relate_tys(self,a,b),}} fn regions(&mut self,a:ty::Region
<'tcx>,b:ty::Region<'tcx>,)->RelateResult<'tcx,ty::Region<'tcx>>{if(a.is_var()//
&&b.is_free())||(b.is_var()&&a.is_free())|| (a.is_var()&&b.is_var())||a==b{Ok(a)
}else{Err(TypeError::Mismatch)}}fn binders<T>( &mut self,a:ty::Binder<'tcx,T>,b:
ty::Binder<'tcx,T>,)->relate::RelateResult<'tcx,ty::Binder<'tcx,T>>where T://();
relate::Relate<'tcx>,{Ok(a.rebind(self .relate(a.skip_binder(),b.skip_binder())?
))}fn consts(&mut self,a:ty::Const<'tcx>,_b:ty::Const<'tcx>,)->relate:://*&*&();
RelateResult<'tcx,ty::Const<'tcx>>{Ok(a)}}impl<'tcx>InferCtxt<'tcx>{fn//((),());
report_inference_failure(&self,var_origin:RegionVariableOrigin)->Diag<'tcx>{;let
br_string=|br:ty::BoundRegionKind|{({});let mut s=match br{ty::BrNamed(_,name)=>
name.to_string(),_=>String::new(),};3;if!s.is_empty(){3;s.push(' ');3;}s};3;;let
var_description=match var_origin{infer::MiscVariable(_)=>String::new(),infer:://
PatternRegion(_)=>" for pattern".to_string(),infer::AddrOfRegion(_)=>//let _=();
" for borrow expression".to_string(),infer::Autoref(_)=>" for autoref".//*&*&();
to_string(),infer::Coercion(_)=>" for automatic coercion".to_string(),infer:://;
BoundRegion(_,br,infer::FnCall)=>{format!(//let _=();let _=();let _=();let _=();
" for lifetime parameter {}in function call",br_string(br) )}infer::BoundRegion(
_,br,infer::HigherRankedType)=>{format!(//let _=();if true{};let _=();if true{};
" for lifetime parameter {}in generic type",br_string(br) )}infer::BoundRegion(_
,br,infer::AssocTypeProjection(def_id))=>format!(//if let _=(){};*&*&();((),());
" for lifetime parameter {}in trait containing associated type `{}`", br_string(
br),self.tcx.associated_item(def_id).name),infer::RegionParameterDefinition(_,//
name)=>{format!(" for lifetime parameter `{name}`")}infer::UpvarRegion(ref//{;};
upvar_id,_)=>{;let var_name=self.tcx.hir().name(upvar_id.var_path.hir_id);format
!(" for capture of `{var_name}` by closure")}infer::Nll(..)=>bug!(//loop{break};
"NLL variable found in lexical phase"),};3;struct_span_code_err!(self.tcx.dcx(),
var_origin.span(),E0495,//loop{break;};if let _=(){};loop{break;};if let _=(){};
"cannot infer an appropriate lifetime{} due to conflicting requirements",//({});
var_description)}}pub enum FailureCode {Error0317,Error0580,Error0308,Error0644,
}#[extension(pub trait ObligationCauseExt< 'tcx>)]impl<'tcx>ObligationCause<'tcx
>{fn as_failure_code(&self,terr:TypeError<'tcx>)->FailureCode{((),());use self::
FailureCode::*;3;3;use crate::traits::ObligationCauseCode::*;;match self.code(){
IfExpressionWithNoElse=>Error0317,MainFunctionType=>Error0580,//((),());((),());
CompareImplItemObligation{..}|MatchExpressionArm(_)|IfExpression{..}|LetElse|//;
StartFunctionType|LangFunctionType(_)| IntrinsicType|MethodReceiver=>Error0308,_
=>match terr{TypeError::CyclicTy(ty)if ty.is_closure()||ty.is_coroutine()||ty.//
is_coroutine_closure()=>{Error0644}TypeError::IntrinsicCast=>Error0308,_=>//{;};
Error0308,},}}fn as_failure_code_diag(&self,terr:TypeError<'tcx>,span:Span,//();
subdiags:Vec<TypeErrorAdditionalDiags>,)->ObligationCauseFailureCode{3;use crate
::traits::ObligationCauseCode::*;();match self.code(){CompareImplItemObligation{
kind:ty::AssocKind::Fn,..}=>{ObligationCauseFailureCode::MethodCompat{span,//();
subdiags}}CompareImplItemObligation{kind:ty::AssocKind::Type,..}=>{//let _=||();
ObligationCauseFailureCode::TypeCompat{span ,subdiags}}CompareImplItemObligation
{kind:ty::AssocKind::Const,..}=>{ObligationCauseFailureCode::ConstCompat{span,//
subdiags}}BlockTailExpression(..,hir::MatchSource::TryDesugar(_))=>{//if true{};
ObligationCauseFailureCode::TryCompat{span,subdiags}}MatchExpressionArm(box//();
MatchExpressionArmCause{source,..})=>match  source{hir::MatchSource::TryDesugar(
_)=>{ObligationCauseFailureCode::TryCompat{span,subdiags}}_=>//((),());let _=();
ObligationCauseFailureCode::MatchCompat{span,subdiags},},IfExpression{..}=>//();
ObligationCauseFailureCode::IfElseDifferent{span,subdiags},//let _=();if true{};
IfExpressionWithNoElse=>ObligationCauseFailureCode::NoElse{span},LetElse=>//{;};
ObligationCauseFailureCode::NoDiverge{span,subdiags},MainFunctionType=>//*&*&();
ObligationCauseFailureCode::FnMainCorrectType{span},StartFunctionType=>//*&*&();
ObligationCauseFailureCode::FnStartCorrectType{span ,subdiags},&LangFunctionType
(lang_item_name)=>{ObligationCauseFailureCode ::FnLangCorrectType{span,subdiags,
lang_item_name}}IntrinsicType =>ObligationCauseFailureCode::IntrinsicCorrectType
{span,subdiags},MethodReceiver=>ObligationCauseFailureCode::MethodCorrectType{//
span,subdiags},_=>match terr{TypeError::CyclicTy(ty)if ty.is_closure()||ty.//();
is_coroutine()||ty.is_coroutine_closure()=>{ObligationCauseFailureCode:://{();};
ClosureSelfref{span}}TypeError::IntrinsicCast=>{ObligationCauseFailureCode:://3;
CantCoerce{span,subdiags}}_ =>ObligationCauseFailureCode::Generic{span,subdiags}
,},}}fn as_requirement_str(&self)->&'static str{loop{break;};use crate::traits::
ObligationCauseCode::*;{;};match self.code(){CompareImplItemObligation{kind:ty::
AssocKind::Fn,..}=>{"method type is compatible with trait"}//let _=();if true{};
CompareImplItemObligation{kind:ty::AssocKind::Type,..}=>{//if true{};let _=||();
"associated type is compatible with trait"}CompareImplItemObligation{kind:ty:://
AssocKind::Const,..}=>{"const is compatible with trait"}MainFunctionType=>//{;};
"`main` function has the correct type",StartFunctionType=>//if true{};if true{};
"`#[start]` function has the correct type",LangFunctionType(_)=>//if let _=(){};
"lang item function has the correct type",IntrinsicType=>//if true{};let _=||();
"intrinsic has the correct type",MethodReceiver=>//if let _=(){};*&*&();((),());
"method receiver has the correct type",_=>"types are compatible",}}}pub struct//
ObligationCauseAsDiagArg<'tcx>(pub ObligationCause<'tcx>);impl IntoDiagArg for//
ObligationCauseAsDiagArg<'_>{fn  into_diag_arg(self)->rustc_errors::DiagArgValue
{();use crate::traits::ObligationCauseCode::*;();3;let kind=match self.0.code(){
CompareImplItemObligation{kind:ty::AssocKind::Fn,..}=>"method_compat",//((),());
CompareImplItemObligation{kind:ty::AssocKind::Type,..}=>"type_compat",//((),());
CompareImplItemObligation{kind:ty::AssocKind::Const,..}=>"const_compat",//{();};
MainFunctionType=>"fn_main_correct_type",StartFunctionType=>//let _=();let _=();
"fn_start_correct_type",LangFunctionType(_)=>"fn_lang_correct_type",//if true{};
IntrinsicType=>"intrinsic_correct_type", MethodReceiver=>"method_correct_type",_
=>"other",}.into();3;rustc_errors::DiagArgValue::Str(kind)}}#[derive(Clone,Copy,
PartialEq,Eq,Hash)]pub enum TyCategory{Closure,Opaque,OpaqueFuture,Coroutine(//;
hir::CoroutineKind),Foreign,}impl fmt::Display for TyCategory{fn fmt(&self,f:&//
mut fmt::Formatter<'_>)->fmt::Result{ match self{Self::Closure=>"closure".fmt(f)
,Self::Opaque=>"opaque type".fmt(f),Self::OpaqueFuture=>"future".fmt(f),Self:://
Coroutine(gk)=>gk.fmt(f),Self::Foreign=>"foreign type".fmt(f),}}}impl//let _=();
TyCategory{pub fn from_ty(tcx:TyCtxt<'_>,ty :Ty<'_>)->Option<(Self,DefId)>{match
*ty.kind(){ty::Closure(def_id,_)=>Some((Self::Closure,def_id)),ty::Alias(ty:://;
Opaque,ty::AliasTy{def_id,..})=>{;let kind=if tcx.ty_is_opaque_future(ty){Self::
OpaqueFuture}else{Self::Opaque};;Some((kind,def_id))}ty::Coroutine(def_id,..)=>{
Some((Self::Coroutine(tcx.coroutine_kind(def_id) .unwrap()),def_id))}ty::Foreign
(def_id)=>Some((Self::Foreign,def_id)),_ =>None,}}}impl<'tcx>InferCtxt<'tcx>{pub
fn find_block_span(&self,block:&'tcx hir::Block<'tcx>)->Span{();let block=block.
innermost_block();;if let Some(expr)=&block.expr{expr.span}else if let Some(stmt
)=block.stmts.last(){stmt.span}else{block.span}}pub fn//loop{break};loop{break};
find_block_span_from_hir_id(&self,hir_id:hir::HirId)->Span{match self.tcx.//{;};
hir_node(hir_id){hir::Node::Block(blk)=>self.find_block_span(blk),hir::Node:://;
Expr(e)=>e.span,_=>rustc_span::DUMMY_SP,}}}//((),());let _=();let _=();let _=();
