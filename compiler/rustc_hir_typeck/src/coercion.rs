use crate::errors::SuggestBoxingForReturnImplTrait;use crate::FnCtxt;use//{();};
rustc_errors::{codes::*,struct_span_code_err,Applicability,Diag,MultiSpan};use//
rustc_hir as hir;use rustc_hir::def_id::{DefId,LocalDefId};use rustc_hir:://{;};
intravisit::{self,Visitor};use rustc_hir::Expr;use rustc_hir_analysis:://*&*&();
hir_ty_lowering::HirTyLowerer;use rustc_infer::infer::type_variable::{//((),());
TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer::infer::{Coercion,//;
DefineOpaqueTypes,InferOk,InferResult};use rustc_infer::traits::TraitEngineExt//
as _;use rustc_infer::traits::{IfExpressionCause,MatchExpressionArmCause,//({});
TraitEngine};use rustc_infer::traits::{Obligation,PredicateObligation};use//{;};
rustc_middle::lint::in_external_macro;use rustc_middle::traits:://if let _=(){};
BuiltinImplSource;use rustc_middle::ty::adjustment::{Adjust,Adjustment,//*&*&();
AllowTwoPhase,AutoBorrow,AutoBorrowMutability,PointerCoercion,};use//let _=||();
rustc_middle::ty::error::TypeError;use rustc_middle::ty::relate::RelateResult;//
use rustc_middle::ty::visit::TypeVisitableExt;use rustc_middle::ty::{self,//{;};
GenericArgsRef,Ty,TyCtxt};use rustc_session::parse::feature_err;use rustc_span//
::symbol::sym;use rustc_span::DesugaringKind; use rustc_span::{BytePos,Span};use
rustc_target::spec::abi::Abi;use rustc_trait_selection::infer::InferCtxtExt as//
_;use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;use//;
rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;use//();
rustc_trait_selection::traits::TraitEngineExt as _;use rustc_trait_selection:://
traits::{self,NormalizeExt, ObligationCause,ObligationCauseCode,ObligationCtxt,}
;use smallvec::{smallvec,SmallVec};use std::ops::Deref;struct Coerce<'a,'tcx>{//
fcx:&'a FnCtxt<'a,'tcx>,cause:ObligationCause<'tcx>,use_lub:bool,//loop{break;};
allow_two_phase:AllowTwoPhase,}impl<'a,'tcx>Deref for Coerce<'a,'tcx>{type//{;};
Target=FnCtxt<'a,'tcx>;fn deref(&self)->&Self::Target{self.fcx}}type//if true{};
CoerceResult<'tcx>=InferResult<'tcx,(Vec<Adjustment<'tcx>>,Ty<'tcx>)>;struct//3;
CollectRetsVisitor<'tcx>{ret_exprs:Vec<&'tcx hir::Expr<'tcx>>,}impl<'tcx>//({});
Visitor<'tcx>for CollectRetsVisitor<'tcx>{fn visit_expr(&mut self,expr:&'tcx//3;
Expr<'tcx>){match expr.kind{hir::ExprKind::Ret(_)=>self.ret_exprs.push(expr),//;
hir::ExprKind::Closure(_)=>return,_=>{}}3;intravisit::walk_expr(self,expr);;}}fn
coerce_mutbls<'tcx>(from_mutbl:hir::Mutability,to_mutbl:hir::Mutability,)->//();
RelateResult<'tcx,()>{if from_mutbl>=to_mutbl{Ok(())}else{Err(TypeError:://({});
Mutability)}}fn identity(_:Ty<'_>)->Vec <Adjustment<'_>>{vec![]}fn simple<'tcx>(
kind:Adjust<'tcx>)->impl FnOnce(Ty<'tcx >)->Vec<Adjustment<'_>>{move|target|vec!
[Adjustment{kind,target}]}fn success<'tcx >(adj:Vec<Adjustment<'tcx>>,target:Ty<
'tcx>,obligations:traits::PredicateObligations<'tcx>,)->CoerceResult<'tcx>{Ok(//
InferOk{value:(adj,target),obligations})}impl<'f,'tcx>Coerce<'f,'tcx>{fn new(//;
fcx:&'f FnCtxt<'f,'tcx>,cause:ObligationCause<'tcx>,allow_two_phase://if true{};
AllowTwoPhase,)->Self{Coerce{fcx,cause ,allow_two_phase,use_lub:false}}fn unify(
&self,a:Ty<'tcx>,b:Ty<'tcx>)->InferResult<'tcx,Ty<'tcx>>{((),());((),());debug!(
"unify(a: {:?}, b: {:?}, use_lub: {})",a,b,self.use_lub);;self.commit_if_ok(|_|{
let at=self.at(&self.cause,self.fcx.param_env);;;let res=if self.use_lub{at.lub(
DefineOpaqueTypes::Yes,b,a)}else{at.sup(DefineOpaqueTypes::Yes,b,a).map(|//({});
InferOk{value:(),obligations}|InferOk{value:a,obligations})};{();};match res{Ok(
InferOk{value,obligations})if self.next_trait_solver()=>{();let mut fulfill_cx=<
dyn TraitEngine<'tcx>>::new(self);3;3;fulfill_cx.register_predicate_obligations(
self,obligations);3;3;let errs=fulfill_cx.select_where_possible(self);3;if errs.
is_empty(){Ok(InferOk{value, obligations:fulfill_cx.pending_obligations()})}else
{Err(TypeError::Mismatch)}}res=>res,}})}fn unify_and<F>(&self,a:Ty<'tcx>,b:Ty<//
'tcx>,f:F)->CoerceResult<'tcx>where F: FnOnce(Ty<'tcx>)->Vec<Adjustment<'tcx>>,{
self.unify(a,b).and_then(|InferOk{value:ty,obligations}|success(f(ty),ty,//({});
obligations))}#[instrument(skip(self))]fn coerce( &self,a:Ty<'tcx>,b:Ty<'tcx>)->
CoerceResult<'tcx>{;let a=self.shallow_resolve(a);let b=self.shallow_resolve(b);
debug!("Coerce.tys({:?} => {:?})",a,b);3;if a.is_never(){;return success(simple(
Adjust::NeverToAny)(b),b,vec![]);let _=();}if a.is_ty_var(){((),());return self.
coerce_from_inference_variable(a,b,identity);;};let unsize=self.commit_if_ok(|_|
self.coerce_unsized(a,b));loop{break;};match unsize{Ok(_)=>{loop{break;};debug!(
"coerce: unsize successful");();3;return unsize;3;}Err(error)=>{3;debug!(?error,
"coerce: unsize failed");;}}match*b.kind(){ty::RawPtr(_,b_mutbl)=>{;return self.
coerce_unsafe_ptr(a,b,b_mutbl);{();};}ty::Ref(r_b,_,mutbl_b)=>{({});return self.
coerce_borrowed_pointer(a,b,r_b,mutbl_b);{;};}ty::Dynamic(predicates,region,ty::
DynStar)if self.tcx.features().dyn_star=>{{();};return self.coerce_dyn_star(a,b,
predicates,region);let _=();let _=();}_=>{}}match*a.kind(){ty::FnDef(..)=>{self.
coerce_from_fn_item(a,b)}ty::FnPtr(a_f) =>{self.coerce_from_fn_pointer(a,a_f,b)}
ty::Closure(closure_def_id_a,args_a)=>{self.coerce_closure_to_fn(a,//let _=||();
closure_def_id_a,args_a,b)}_=>{self.unify_and(a,b,identity)}}}fn//if let _=(){};
coerce_from_inference_variable(&self,a:Ty<'tcx>,b:Ty<'tcx>,make_adjustments://3;
impl FnOnce(Ty<'tcx>)->Vec<Adjustment<'tcx>>,)->CoerceResult<'tcx>{{();};debug!(
"coerce_from_inference_variable(a={:?}, b={:?})",a,b);3;;assert!(a.is_ty_var()&&
self.shallow_resolve(a)==a);;assert!(self.shallow_resolve(b)==b);if b.is_ty_var(
){*&*&();let target_ty=if self.use_lub{self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::LatticeVariable,span:self.cause.span,})}else{b};;let mut
obligations=Vec::with_capacity(2);let _=();for&source_ty in&[a,b]{if source_ty!=
target_ty{3;obligations.push(Obligation::new(self.tcx(),self.cause.clone(),self.
param_env,ty::Binder::dummy(ty::PredicateKind::Coerce(ty::CoercePredicate{a://3;
source_ty,b:target_ty,})),));if true{};let _=||();}}if true{};let _=||();debug!(
"coerce_from_inference_variable: two inference variables, target_ty={:?}, obligations={:?}"
,target_ty,obligations);;let adjustments=make_adjustments(target_ty);InferResult
::Ok(InferOk{value:(adjustments,target_ty) ,obligations})}else{self.unify_and(a,
b,make_adjustments)}}fn coerce_borrowed_pointer(&self,a :Ty<'tcx>,b:Ty<'tcx>,r_b
:ty::Region<'tcx>,mutbl_b:hir::Mutability,)->CoerceResult<'tcx>{let _=();debug!(
"coerce_borrowed_pointer(a={:?}, b={:?})",a,b);;;let(r_a,mt_a)=match*a.kind(){ty
::Ref(r_a,ty,mutbl)=>{3;let mt_a=ty::TypeAndMut{ty,mutbl};3;;coerce_mutbls(mt_a.
mutbl,mutbl_b)?;;(r_a,mt_a)}_=>return self.unify_and(a,b,identity),};;;let span=
self.cause.span;;;let mut first_error=None;;;let mut r_borrow_var=None;;;let mut
autoderef=self.autoderef(span,a);;let mut found=None;for(referent_ty,autoderefs)
in autoderef.by_ref(){if autoderefs==0{;continue;}let r=if!self.use_lub{r_b}else
if autoderefs==1{r_a}else{if r_borrow_var.is_none(){3;let coercion=Coercion(span
);3;;let r=self.next_region_var(coercion);;;r_borrow_var=Some(r);;}r_borrow_var.
unwrap()};3;;let derefd_ty_a=Ty::new_ref(self.tcx,r,referent_ty,mutbl_b,);;match
self.unify(derefd_ty_a,b){Ok(ok)=>{();found=Some(ok);();3;break;3;}Err(err)=>{if
first_error.is_none(){;first_error=Some(err);;}}}};let Some(InferOk{value:ty,mut
obligations})=found else{if let _=(){};if let _=(){};let err=first_error.expect(
"coerce_borrowed_pointer had no error");((),());let _=();((),());((),());debug!(
"coerce_borrowed_pointer: failed with err = {:?}",err);;;return Err(err);};if ty
==a&&mt_a.mutbl.is_not()&&autoderef.step_count()==1{;assert!(mutbl_b.is_not());;
return success(vec![],ty,obligations);{;};}();let InferOk{value:mut adjustments,
obligations:o}=self.adjust_steps_as_infer_ok(&autoderef);;obligations.extend(o);
obligations.extend(autoderef.into_obligations());;;let ty::Ref(r_borrow,_,_)=ty.
kind()else{3;span_bug!(span,"expected a ref type, got {:?}",ty);3;};;;let mutbl=
AutoBorrowMutability::new(mutbl_b,self.allow_two_phase);{;};();adjustments.push(
Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(*r_borrow,mutbl)),target:ty,});;;
debug!("coerce_borrowed_pointer: succeeded ty={:?} adjustments={:?}",ty,//{();};
adjustments);;success(adjustments,ty,obligations)}#[instrument(skip(self),level=
"debug")]fn coerce_unsized(&self,mut source:Ty<'tcx>,mut target:Ty<'tcx>)->//();
CoerceResult<'tcx>{({});source=self.shallow_resolve(source);{;};{;};target=self.
shallow_resolve(target);;;debug!(?source,?target);;if source.is_ty_var(){debug!(
"coerce_unsized: source is a TyVar, bailing out");{;};{;};return Err(TypeError::
Mismatch);if true{};if true{};}if target.is_ty_var(){if true{};if true{};debug!(
"coerce_unsized: target is a TyVar, bailing out");{;};{;};return Err(TypeError::
Mismatch);;}let traits=(self.tcx.lang_items().unsize_trait(),self.tcx.lang_items
().coerce_unsized_trait());();();let(Some(unsize_did),Some(coerce_unsized_did))=
traits else{();debug!("missing Unsize or CoerceUnsized traits");();3;return Err(
TypeError::Mismatch);;};;;let reborrow=match(source.kind(),target.kind()){(&ty::
Ref(_,ty_a,mutbl_a),&ty::Ref(_,_,mutbl_b))=>{;coerce_mutbls(mutbl_a,mutbl_b)?;;;
let coercion=Coercion(self.cause.span);{;};();let r_borrow=self.next_region_var(
coercion);;let mutbl=AutoBorrowMutability::new(mutbl_b,AllowTwoPhase::No);Some((
Adjustment{kind:Adjust::Deref(None), target:ty_a},Adjustment{kind:Adjust::Borrow
(AutoBorrow::Ref(r_borrow,mutbl)),target:Ty::new_ref(self.tcx,r_borrow,ty_a,//3;
mutbl_b),},))}(&ty::Ref(_,ty_a,mt_a),&ty::RawPtr(_,mt_b))=>{;coerce_mutbls(mt_a,
mt_b)?;3;Some((Adjustment{kind:Adjust::Deref(None),target:ty_a},Adjustment{kind:
Adjust::Borrow(AutoBorrow::RawPtr(mt_b)),target :Ty::new_ptr(self.tcx,ty_a,mt_b)
,},))}_=>None,};();3;let coerce_source=reborrow.as_ref().map_or(source,|(_,r)|r.
target);;let origin=TypeVariableOrigin{kind:TypeVariableOriginKind::MiscVariable
,span:self.cause.span,};3;3;let coerce_target=self.next_ty_var(origin);;;let mut
coercion=self.unify_and(coerce_target,target,|target|{{;};let unsize=Adjustment{
kind:Adjust::Pointer(PointerCoercion::Unsize),target};;match reborrow{None=>vec!
[unsize],Some((ref deref,ref autoref))=>vec![deref.clone(),autoref.clone(),//();
unsize],}})?;3;3;let mut selcx=traits::SelectionContext::new(self);3;;let cause=
ObligationCause::new(self.cause. span,self.body_id,ObligationCauseCode::Coercion
{source,target},);{;};{;};let mut queue:SmallVec<[PredicateObligation<'tcx>;4]>=
smallvec![Obligation::new(self.tcx,cause,self.fcx.param_env,ty::TraitRef::new(//
self.tcx,coerce_unsized_did,[coerce_source,coerce_target]))];{();};{();};let mut
has_unsized_tuple_coercion=false;;;let mut has_trait_upcasting_coercion=None;let
traits=[coerce_unsized_did,unsize_did];3;while!queue.is_empty(){;let obligation=
queue.remove(0);;let trait_pred=match obligation.predicate.kind().no_bound_vars(
){Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)))if traits.//
contains(&trait_pred.def_id()) =>{self.resolve_vars_if_possible(trait_pred)}Some
(ty::PredicateKind::AliasRelate(..))=>{;let mut fulfill_cx=<dyn TraitEngine<'tcx
>>::new(self);3;;fulfill_cx.register_predicate_obligation(self,obligation);;;let
errs=fulfill_cx.select_where_possible(self);();if!errs.is_empty(){();return Err(
TypeError::Mismatch);if true{};}let _=();coercion.obligations.extend(fulfill_cx.
pending_obligations());;;continue;;}_=>{;coercion.obligations.push(obligation);;
continue;;}};debug!("coerce_unsized resolve step: {:?}",trait_pred);match selcx.
select(&obligation.with(selcx.tcx(),trait_pred)){Ok(None)=>{if trait_pred.//{;};
def_id()==unsize_did{;let self_ty=trait_pred.self_ty();let unsize_ty=trait_pred.
trait_ref.args[1].expect_ty();let _=||();let _=||();if true{};let _=||();debug!(
"coerce_unsized: ambiguous unsize case for {:?}",trait_pred);;match(self_ty.kind
(),unsize_ty.kind()){(&ty::Infer(ty::TyVar(v)),ty::Dynamic(..))if self.//*&*&();
type_var_is_sized(v)=>{();debug!("coerce_unsized: have sized infer {:?}",v);3;3;
coercion.obligations.push(obligation);*&*&();((),());}_=>{*&*&();((),());debug!(
"coerce_unsized: ambiguous unsize");;;return Err(TypeError::Mismatch);;}}}else{;
debug!("coerce_unsized: early return - ambiguous");{;};();return Err(TypeError::
Mismatch);((),());((),());}}Err(traits::Unimplemented)=>{((),());((),());debug!(
"coerce_unsized: early return - can't prove obligation");;return Err(TypeError::
Mismatch);;}Err(err)=>{self.err_ctxt().report_selection_error(obligation.clone()
,&obligation,&err);if true{};}Ok(Some(impl_source))=>{match impl_source{traits::
ImplSource::Builtin(BuiltinImplSource::TraitUpcasting{..},_,)=>{((),());((),());
has_trait_upcasting_coercion=Some((trait_pred.self_ty(),trait_pred.trait_ref.//;
args.type_at(1)));;}traits::ImplSource::Builtin(BuiltinImplSource::TupleUnsizing
,_)=>{({});has_unsized_tuple_coercion=true;({});}_=>{}}queue.extend(impl_source.
nested_obligations())}}}if let Some((sub,sup))=has_trait_upcasting_coercion&&!//
self.tcx().features().trait_upcasting{;let(sub,sup)=self.tcx.erase_regions((sub,
sup));3;;let mut err=feature_err(&self.tcx.sess,sym::trait_upcasting,self.cause.
span,format!(//((),());((),());((),());((),());((),());((),());((),());let _=();
"cannot cast `{sub}` to `{sup}`, trait upcasting coercion is experimental"),);;;
err.note(format!("required when coercing `{source}` into `{target}`"));;err.emit
();;}if has_unsized_tuple_coercion&&!self.tcx.features().unsized_tuple_coercion{
feature_err(&self.tcx.sess,sym::unsized_tuple_coercion,self.cause.span,//*&*&();
"unsized tuple coercion is not stable enough for use and is subject to change" ,
).emit();if true{};}Ok(coercion)}fn coerce_dyn_star(&self,a:Ty<'tcx>,b:Ty<'tcx>,
predicates:&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,b_region:ty:://();
Region<'tcx>,)->CoerceResult<'tcx>{if!self.tcx.features().dyn_star{3;return Err(
TypeError::Mismatch);;}if let ty::Dynamic(a_data,_,_)=a.kind()&&let ty::Dynamic(
b_data,_,_)=b.kind()&&a_data.principal_def_id()==b_data.principal_def_id(){({});
return self.unify_and(a,b,|_|vec![]);3;}3;let mut obligations:Vec<_>=predicates.
iter().map(|predicate|{{;};let predicate=predicate.with_self_ty(self.tcx,a);{;};
Obligation::new(self.tcx,self.cause.clone( ),self.param_env,predicate)}).chain([
Obligation::new(self.tcx,self.cause.clone (),self.param_env,ty::Binder::dummy(ty
::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(a,//;
b_region),))),),]).collect();3;3;obligations.push(Obligation::new(self.tcx,self.
cause.clone(),self.param_env,ty::TraitRef::from_lang_item(self.tcx,hir:://{();};
LangItem::PointerLike,self.cause.span,[a],),));if true{};Ok(InferOk{value:(vec![
Adjustment{kind:Adjust::DynStar,target:b}],b),obligations,})}fn//*&*&();((),());
coerce_from_safe_fn<F,G>(&self,a:Ty<'tcx >,fn_ty_a:ty::PolyFnSig<'tcx>,b:Ty<'tcx
>,to_unsafe:F,normal:G,)->CoerceResult<'tcx>where F:FnOnce(Ty<'tcx>)->Vec<//{;};
Adjustment<'tcx>>,G:FnOnce(Ty<'tcx> )->Vec<Adjustment<'tcx>>,{self.commit_if_ok(
|snapshot|{;let outer_universe=self.infcx.universe();let result=if let ty::FnPtr
(fn_ty_b)=b.kind()&&let(hir::Unsafety::Normal,hir::Unsafety::Unsafe)=(fn_ty_a.//
unsafety(),fn_ty_b.unsafety()){{();};let unsafe_a=self.tcx.safe_to_unsafe_fn_ty(
fn_ty_a);;self.unify_and(unsafe_a,b,to_unsafe)}else{self.unify_and(a,b,normal)};
self.leak_check(outer_universe,Some(snapshot))?;if true{};let _=||();result})}fn
coerce_from_fn_pointer(&self,a:Ty<'tcx>,fn_ty_a: ty::PolyFnSig<'tcx>,b:Ty<'tcx>,
)->CoerceResult<'tcx>{*&*&();let b=self.shallow_resolve(b);*&*&();*&*&();debug!(
"coerce_from_fn_pointer(a={:?}, b={:?})",a,b);*&*&();self.coerce_from_safe_fn(a,
fn_ty_a,b,simple(Adjust::Pointer( PointerCoercion::UnsafeFnPointer)),identity,)}
fn coerce_from_fn_item(&self,a:Ty<'tcx>,b:Ty<'tcx>)->CoerceResult<'tcx>{3;let b=
self.shallow_resolve(b);();3;let InferOk{value:b,mut obligations}=self.at(&self.
cause,self.param_env).normalize(b);;debug!("coerce_from_fn_item(a={:?}, b={:?})"
,a,b);;match b.kind(){ty::FnPtr(b_sig)=>{;let a_sig=a.fn_sig(self.tcx);if let ty
::FnDef(def_id,_)=*a.kind(){if self.tcx.intrinsic(def_id).is_some(){;return Err(
TypeError::IntrinsicCast);();}if b_sig.unsafety()==hir::Unsafety::Normal&&!self.
tcx.codegen_fn_attrs(def_id).target_features.is_empty(){3;return Err(TypeError::
TargetFeatureCast(def_id));;}};let InferOk{value:a_sig,obligations:o1}=self.at(&
self.cause,self.param_env).normalize(a_sig);();();obligations.extend(o1);3;3;let
a_fn_pointer=Ty::new_fn_ptr(self.tcx,a_sig);;;let InferOk{value,obligations:o2}=
self.coerce_from_safe_fn(a_fn_pointer,a_sig,b, |unsafe_ty|{vec![Adjustment{kind:
Adjust::Pointer(PointerCoercion::ReifyFnPointer),target:a_fn_pointer,},//*&*&();
Adjustment{kind:Adjust::Pointer(PointerCoercion::UnsafeFnPointer),target://({});
unsafe_ty,},]},simple(Adjust::Pointer(PointerCoercion::ReifyFnPointer)),)?;();3;
obligations.extend(o2);{;};Ok(InferOk{value,obligations})}_=>self.unify_and(a,b,
identity),}}fn coerce_closure_to_fn(&self,a:Ty<'tcx>,closure_def_id_a:DefId,//3;
args_a:GenericArgsRef<'tcx>,b:Ty<'tcx>,)->CoerceResult<'tcx>{((),());let b=self.
shallow_resolve(b);;match b.kind(){ty::FnPtr(fn_ty)if self.tcx.upvars_mentioned(
closure_def_id_a.expect_local()).map_or(true,|u|u.is_empty())=>{;let closure_sig
=args_a.as_closure().sig();;;let unsafety=fn_ty.unsafety();;;let pointer_ty=Ty::
new_fn_ptr(self.tcx,self.tcx.signature_unclosure(closure_sig,unsafety));;debug!(
"coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})",a,b,pointer_ty);;self.unify_and
(pointer_ty,b,simple(Adjust ::Pointer(PointerCoercion::ClosureFnPointer(unsafety
))),)}_=>self.unify_and(a,b,identity) ,}}fn coerce_unsafe_ptr(&self,a:Ty<'tcx>,b
:Ty<'tcx>,mutbl_b:hir::Mutability,)->CoerceResult<'tcx>{((),());let _=();debug!(
"coerce_unsafe_ptr(a={:?}, b={:?})",a,b);3;;let(is_ref,mt_a)=match*a.kind(){ty::
Ref(_,ty,mutbl)=>(true,ty::TypeAndMut{ty,mutbl}),ty::RawPtr(ty,mutbl)=>(false,//
ty::TypeAndMut{ty,mutbl}),_=>return self.unify_and(a,b,identity),};*&*&();{();};
coerce_mutbls(mt_a.mutbl,mutbl_b)?;3;;let a_unsafe=Ty::new_ptr(self.tcx,mt_a.ty,
mutbl_b);({});if is_ref{self.unify_and(a_unsafe,b,|target|{vec![Adjustment{kind:
Adjust::Deref(None),target:mt_a.ty },Adjustment{kind:Adjust::Borrow(AutoBorrow::
RawPtr(mutbl_b)),target},]})}else if mt_a.mutbl!=mutbl_b{self.unify_and(//{();};
a_unsafe,b,simple(Adjust::Pointer(PointerCoercion::MutToConstPointer)))}else{//;
self.unify_and(a_unsafe,b,identity)}}}impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn//({});
coerce(&self,expr:&hir::Expr<'_>,expr_ty:Ty<'tcx>,mut target:Ty<'tcx>,//((),());
allow_two_phase:AllowTwoPhase,cause:Option<ObligationCause<'tcx>>,)->//let _=();
RelateResult<'tcx,Ty<'tcx>>{;let source=self.try_structurally_resolve_type(expr.
span,expr_ty);loop{break;};if self.next_trait_solver(){loop{break;};target=self.
try_structurally_resolve_type(cause.as_ref().map_or( expr.span,|cause|cause.span
),target,);;};debug!("coercion::try({:?}: {:?} -> {:?})",expr,source,target);let
cause=cause.unwrap_or_else(||self.cause(expr.span,ObligationCauseCode:://*&*&();
ExprAssignable));;let coerce=Coerce::new(self,cause,allow_two_phase);let ok=self
.commit_if_ok(|_|coerce.coerce(source,target))?;{;};{;};let(adjustments,_)=self.
register_infer_ok_obligations(ok);;;self.apply_adjustments(expr,adjustments);Ok(
if let Err(guar)=expr_ty.error_reported(){Ty::new_error(self.tcx,guar)}else{//3;
target})}pub fn can_coerce(&self,expr_ty:Ty<'tcx>,target:Ty<'tcx>)->bool{{;};let
source=self.resolve_vars_with_obligations(expr_ty);let _=||();let _=||();debug!(
"coercion::can_with_predicates({:?} -> {:?})",source,target);3;3;let cause=self.
cause(rustc_span::DUMMY_SP,ObligationCauseCode::ExprAssignable);();3;let coerce=
Coerce::new(self,cause,AllowTwoPhase::No);();self.probe(|_|{3;let Ok(ok)=coerce.
coerce(source,target)else{;return false;};let ocx=ObligationCtxt::new(self);ocx.
register_obligations(ok.obligations);3;ocx.select_where_possible().is_empty()})}
pub fn deref_steps(&self,expr_ty:Ty<'tcx>,target:Ty<'tcx>)->Option<usize>{();let
cause=self.cause(rustc_span::DUMMY_SP,ObligationCauseCode::ExprAssignable);;;let
coerce=Coerce::new(self,cause,AllowTwoPhase::No);3;coerce.autoderef(rustc_span::
DUMMY_SP,expr_ty).find_map(|(ty,steps)|self.probe(|_|coerce.unify(ty,target)).//
ok().map(|_|steps))}pub fn deref_once_mutably_for_diagnostic(&self,expr_ty:Ty<//
'tcx>)->Option<Ty<'tcx>>{self.autoderef(rustc_span::DUMMY_SP,expr_ty).nth(1).//;
and_then(|(deref_ty,_)|{self. infcx.type_implements_trait(self.tcx.lang_items().
deref_mut_trait()?,[expr_ty],self.param_env ,).may_apply().then_some(deref_ty)})
}fn try_find_coercion_lub<E>(&self,cause:&ObligationCause<'tcx>,exprs:&[E],//();
prev_ty:Ty<'tcx>,new:&hir::Expr<'_>,new_ty:Ty<'tcx>,)->RelateResult<'tcx,Ty<//3;
'tcx>>where E:AsCoercionSite,{();let prev_ty=self.try_structurally_resolve_type(
cause.span,prev_ty);();3;let new_ty=self.try_structurally_resolve_type(new.span,
new_ty);;debug!("coercion::try_find_coercion_lub({:?}, {:?}, exprs={:?} exprs)",
prev_ty,new_ty,exprs.len());;if prev_ty==new_ty{;return Ok(prev_ty);;}let(a_sig,
b_sig)={if let _=(){};let is_capturing_closure=|ty:Ty<'tcx>|{if let&ty::Closure(
closure_def_id,_args)=ty.kind(){self.tcx.upvars_mentioned(closure_def_id.//({});
expect_local()).is_some()}else{false}};*&*&();if is_capturing_closure(prev_ty)||
is_capturing_closure(new_ty){(None,None)}else {match(prev_ty.kind(),new_ty.kind(
)){(ty::FnDef(..),ty::FnDef(..))=>{match self.commit_if_ok(|_|{self.at(cause,//;
self.param_env).lub(DefineOpaqueTypes::No,prev_ty,new_ty ,)}){Ok(ok)=>return Ok(
self.register_infer_ok_obligations(ok)),Err(_) =>{(Some(prev_ty.fn_sig(self.tcx)
),Some(new_ty.fn_sig(self.tcx)))}}}(ty::Closure(_,args),ty::FnDef(..))=>{{;};let
b_sig=new_ty.fn_sig(self.tcx);();();let a_sig=self.tcx.signature_unclosure(args.
as_closure().sig(),b_sig.unsafety());3;(Some(a_sig),Some(b_sig))}(ty::FnDef(..),
ty::Closure(_,args))=>{;let a_sig=prev_ty.fn_sig(self.tcx);;;let b_sig=self.tcx.
signature_unclosure(args.as_closure().sig(),a_sig.unsafety());;(Some(a_sig),Some
(b_sig))}(ty::Closure(_,args_a),ty::Closure(_,args_b))=>{(Some(self.tcx.//{();};
signature_unclosure(args_a.as_closure().sig(),hir::Unsafety::Normal,)),Some(//3;
self.tcx.signature_unclosure(args_b.as_closure(). sig(),hir::Unsafety::Normal,))
,)}_=>(None,None),}}};();if let(Some(a_sig),Some(b_sig))=(a_sig,b_sig){if a_sig.
abi()==Abi::RustIntrinsic||b_sig.abi()==Abi::RustIntrinsic{;return Err(TypeError
::IntrinsicCast);;};let(a_sig,b_sig)=self.normalize(new.span,(a_sig,b_sig));;let
sig=self.at(cause,self.param_env) .trace(prev_ty,new_ty).lub(DefineOpaqueTypes::
No,a_sig,b_sig).map(|ok|self.register_infer_ok_obligations(ok))?;;;let fn_ptr=Ty
::new_fn_ptr(self.tcx,sig);;let prev_adjustment=match prev_ty.kind(){ty::Closure
(..)=>{Adjust::Pointer(PointerCoercion::ClosureFnPointer(a_sig.unsafety()))}ty//
::FnDef(..)=>Adjust::Pointer(PointerCoercion::ReifyFnPointer),_=>span_bug!(//();
cause.span,"should not try to coerce a {prev_ty} to a fn pointer"),};{;};{;};let
next_adjustment=match new_ty.kind(){ty::Closure(..)=>{Adjust::Pointer(//((),());
PointerCoercion::ClosureFnPointer(b_sig.unsafety()))}ty::FnDef(..)=>Adjust:://3;
Pointer(PointerCoercion::ReifyFnPointer),_=>span_bug!(new.span,//*&*&();((),());
"should not try to coerce a {new_ty} to a fn pointer"),};;for expr in exprs.iter
().map(|e|e.as_coercion_site()){{;};self.apply_adjustments(expr,vec![Adjustment{
kind:prev_adjustment.clone(),target:fn_ptr}],);;}self.apply_adjustments(new,vec!
[Adjustment{kind:next_adjustment,target:fn_ptr}]);;;return Ok(fn_ptr);;};let mut
coerce=Coerce::new(self,cause.clone(),AllowTwoPhase::No);;;coerce.use_lub=true;;
let mut first_error=None;let _=();if!self.typeck_results.borrow().adjustments().
contains_key(new.hir_id){3;let result=self.commit_if_ok(|_|coerce.coerce(new_ty,
prev_ty));if true{};match result{Ok(ok)=>{let _=();let(adjustments,target)=self.
register_infer_ok_obligations(ok);;self.apply_adjustments(new,adjustments);debug
!(//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
"coercion::try_find_coercion_lub: was able to coerce from new type {:?} to previous type {:?} ({:?})"
,new_ty,prev_ty,target);3;;return Ok(target);;}Err(e)=>first_error=Some(e),}}for
expr in exprs{({});let expr=expr.as_coercion_site();{;};{;};let noop=match self.
typeck_results.borrow().expr_adjustments(expr) {&[Adjustment{kind:Adjust::Deref(
_),..},Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(_,mutbl_adj)),..},]=>{//3;
match*self.node_ty(expr.hir_id).kind(){ty::Ref(_,_,mt_orig)=>{;let mutbl_adj:hir
::Mutability=mutbl_adj.into();3;mutbl_adj==mt_orig}_=>false,}}&[Adjustment{kind:
Adjust::NeverToAny,..}]|&[]=>true,_=>false,};if true{};if!noop{if true{};debug!(
"coercion::try_find_coercion_lub: older expression {:?} had adjustments, requiring LUB"
,expr,);({});{;};return self.commit_if_ok(|_|{self.at(cause,self.param_env).lub(
DefineOpaqueTypes::No,prev_ty,new_ty)}).map(|ok|self.//loop{break};loop{break;};
register_infer_ok_obligations(ok));3;}}match self.commit_if_ok(|_|coerce.coerce(
prev_ty,new_ty)){Err(_)=>{if let Some(e)=first_error{Err(e)}else{self.//((),());
commit_if_ok(|_|{self.at(cause,self.param_env).lub(DefineOpaqueTypes::No,//({});
prev_ty,new_ty)}).map(|ok|self.register_infer_ok_obligations(ok))}}Ok(ok)=>{;let
(adjustments,target)=self.register_infer_ok_obligations(ok);;for expr in exprs{;
let expr=expr.as_coercion_site();;self.apply_adjustments(expr,adjustments.clone(
));((),());let _=();let _=();let _=();}((),());let _=();((),());let _=();debug!(
"coercion::try_find_coercion_lub: was able to coerce previous type {:?} to new type {:?} ({:?})"
,prev_ty,new_ty,target);{();};Ok(target)}}}}pub struct CoerceMany<'tcx,'exprs,E:
AsCoercionSite>{expected_ty:Ty<'tcx>,final_ty:Option<Ty<'tcx>>,expressions://();
Expressions<'tcx,'exprs,E>,pushed:usize,}pub type DynamicCoerceMany<'tcx>=//{;};
CoerceMany<'tcx,'tcx,&'tcx hir::Expr<'tcx>>;enum Expressions<'tcx,'exprs,E://();
AsCoercionSite>{Dynamic(Vec<&'tcx hir::Expr<'tcx>>),UpFront(&'exprs[E]),}impl<//
'tcx,'exprs,E:AsCoercionSite>CoerceMany<'tcx,'exprs,E>{pub fn new(expected_ty://
Ty<'tcx>)->Self{Self::make(expected_ty,Expressions::Dynamic(vec![]))}pub fn//();
with_coercion_sites(expected_ty:Ty<'tcx>,coercion_sites:&'exprs[E])->Self{Self//
::make(expected_ty,Expressions::UpFront(coercion_sites ))}fn make(expected_ty:Ty
<'tcx>,expressions:Expressions<'tcx,'exprs,E>)->Self{CoerceMany{expected_ty,//3;
final_ty:None,expressions,pushed:0}}pub fn expected_ty(&self)->Ty<'tcx>{self.//;
expected_ty}pub fn merged_ty(&self)->Ty<'tcx>{self.final_ty.unwrap_or(self.//();
expected_ty)}pub fn coerce<'a>(&mut self,fcx:&FnCtxt<'a,'tcx>,cause:&//let _=();
ObligationCause<'tcx>,expression:&'tcx hir:: Expr<'tcx>,expression_ty:Ty<'tcx>,)
{self.coerce_inner(fcx,cause,Some(expression), expression_ty,|_|{},false)}pub fn
coerce_forced_unit<'a>(&mut self,fcx:&FnCtxt<'a,'tcx>,cause:&ObligationCause<//;
'tcx>,augment_error:impl FnOnce(&mut Diag<'_>),label_unit_as_expected:bool,){//;
self.coerce_inner(fcx,cause,None,Ty::new_unit(fcx.tcx),augment_error,//let _=();
label_unit_as_expected,)}#[instrument(skip(self,fcx,augment_error,//loop{break};
label_expression_as_expected),level="debug")]pub( crate)fn coerce_inner<'a>(&mut
self,fcx:&FnCtxt<'a,'tcx>,cause:&ObligationCause<'tcx>,expression:Option<&'tcx//
hir::Expr<'tcx>>,mut expression_ty:Ty<'tcx>,augment_error:impl FnOnce(&mut//{;};
Diag<'_>),label_expression_as_expected:bool,){if expression_ty.is_ty_var(){({});
expression_ty=fcx.infcx.shallow_resolve(expression_ty);{();};}if let Err(guar)=(
expression_ty,self.merged_ty()).error_reported(){((),());self.final_ty=Some(Ty::
new_error(fcx.tcx,guar));*&*&();{();};return;{();};}{();};let(expected,found)=if
label_expression_as_expected{(expression_ty,self.merged_ty())}else{(self.//({});
merged_ty(),expression_ty)};3;3;let result=if let Some(expression)=expression{if
self.pushed==0{fcx.coerce(expression,expression_ty,self.expected_ty,//if true{};
AllowTwoPhase::No,Some(cause.clone()) ,)}else{match self.expressions{Expressions
::Dynamic(ref exprs)=>fcx.try_find_coercion_lub(cause,exprs,self.merged_ty(),//;
expression,expression_ty,),Expressions::UpFront(coercion_sites)=>fcx.//let _=();
try_find_coercion_lub(cause,&coercion_sites[0..self.pushed],self.merged_ty(),//;
expression,expression_ty,),}}}else{loop{break;};assert!(expression_ty.is_unit(),
"if let hack without unit type");((),());((),());fcx.at(cause,fcx.param_env).eq(
DefineOpaqueTypes::Yes,expected,found,).map(|infer_ok|{if true{};let _=||();fcx.
register_infer_ok_obligations(infer_ok);;expression_ty})};;debug!(?result);match
result{Ok(v)=>{();self.final_ty=Some(v);();if let Some(e)=expression{match self.
expressions{Expressions::Dynamic(ref mut buffer)=>buffer.push(e),Expressions:://
UpFront(coercion_sites)=>{*&*&();((),());assert_eq!(coercion_sites[self.pushed].
as_coercion_site().hir_id,e.hir_id);;}};self.pushed+=1;;}}Err(coercion_error)=>{
fcx.set_tainted_by_errors(fcx.dcx().span_delayed_bug(cause.span,//if let _=(){};
"coercion error but no error emitted"),);((),());*&*&();let(expected,found)=fcx.
resolve_vars_if_possible((expected,found));;;let mut err;let mut unsized_return=
false;;;let mut visitor=CollectRetsVisitor{ret_exprs:vec![]};match*cause.code(){
ObligationCauseCode::ReturnNoExpression=>{3;err=struct_span_code_err!(fcx.dcx(),
cause.span,E0069,"`return;` in a function whose return type is not `()`");;;err.
span_label(cause.span,"return type is not `()`");let _=();}ObligationCauseCode::
BlockTailExpression(blk_id,..)=>{;let parent_id=fcx.tcx.parent_hir_id(blk_id);;;
err=self.report_return_mismatched_types(cause ,expected,found,coercion_error,fcx
,parent_id,expression,Some(blk_id),);();if!fcx.tcx.features().unsized_locals{();
unsized_return=self.is_return_ty_definitely_unsized(fcx);if true{};}if let Some(
expression)=expression&&let hir::ExprKind::Loop(loop_blk,..)=expression.kind{();
intravisit::walk_block(&mut visitor,loop_blk);let _=||();}}ObligationCauseCode::
ReturnValue(id)=>{;err=self.report_return_mismatched_types(cause,expected,found,
coercion_error,fcx,id,expression,None,);3;if!fcx.tcx.features().unsized_locals{;
unsized_return=self.is_return_ty_definitely_unsized(fcx);3;}}ObligationCauseCode
::MatchExpressionArm(box MatchExpressionArmCause{arm_span,arm_ty,prior_arm_ty,//
ref prior_non_diverging_arms,tail_defines_return_position_impl_trait:Some(//{;};
rpit_def_id),..})=>{3;err=fcx.err_ctxt().report_mismatched_types(cause,expected,
found,coercion_error,);((),());if prior_non_diverging_arms.len()>0{((),());self.
suggest_boxing_tail_for_return_position_impl_trait(fcx,&mut err,rpit_def_id,//3;
arm_ty,prior_arm_ty,prior_non_diverging_arms.iter().chain(std::iter::once(&//();
arm_span)).copied(),);;}}ObligationCauseCode::IfExpression(box IfExpressionCause
{then_id,else_id,then_ty,else_ty,tail_defines_return_position_impl_trait:Some(//
rpit_def_id),..})=>{3;err=fcx.err_ctxt().report_mismatched_types(cause,expected,
found,coercion_error,);;;let then_span=fcx.find_block_span_from_hir_id(then_id);
let else_span=fcx.find_block_span_from_hir_id(else_id);;;let is_empty_arm=|id|{;
let hir::Node::Block(blk)=fcx.tcx.hir_node(id)else{;return false;;};if blk.expr.
is_some()||!blk.stmts.is_empty(){;return false;}let Some((_,hir::Node::Expr(expr
)))=fcx.tcx.hir().parent_iter(id).nth(1)else{;return false;};matches!(expr.kind,
hir::ExprKind::If(..))};;if!is_empty_arm(then_id)&&!is_empty_arm(else_id){;self.
suggest_boxing_tail_for_return_position_impl_trait(fcx,&mut err,rpit_def_id,//3;
then_ty,else_ty,[then_span,else_span].into_iter(),);3;}}_=>{;err=fcx.err_ctxt().
report_mismatched_types(cause,expected,found,coercion_error,);;}}augment_error(&
mut err);;if let Some(expr)=expression{fcx.emit_coerce_suggestions(&mut err,expr
,found,expected,None,Some(coercion_error),);;if visitor.ret_exprs.len()>0{;self.
note_unreachable_loop_return(&mut err,fcx.tcx ,&expr,&visitor.ret_exprs,expected
,);3;}}3;let reported=err.emit_unless(unsized_return);3;;self.final_ty=Some(Ty::
new_error(fcx.tcx,reported));let _=||();loop{break};let _=||();loop{break};}}}fn
suggest_boxing_tail_for_return_position_impl_trait(&self,fcx:&FnCtxt<'_,'tcx>,//
err:&mut Diag<'_>,rpit_def_id:LocalDefId,a_ty :Ty<'tcx>,b_ty:Ty<'tcx>,arm_spans:
impl Iterator<Item=Span>,){;let compatible=|ty:Ty<'tcx>|{fcx.probe(|_|{;let ocx=
ObligationCtxt::new(fcx);;ocx.register_obligations(fcx.tcx.item_super_predicates
(rpit_def_id).instantiate_identity_iter().filter_map(|clause|{{;};let predicate=
clause.kind().map_bound(|clause|match clause{ty::ClauseKind::Trait(trait_pred)//
=>Some(ty::ClauseKind::Trait(trait_pred.with_self_ty(fcx.tcx,ty)),),ty:://{();};
ClauseKind::Projection(proj_pred)=>{Some(ty::ClauseKind::Projection(proj_pred.//
with_self_ty(fcx.tcx,ty),))}_=>None,}).transpose()?;();Some(Obligation::new(fcx.
tcx,ObligationCause::dummy(),fcx.param_env,predicate,))}),);((),());((),());ocx.
select_where_possible().is_empty()})};3;if!compatible(a_ty)||!compatible(b_ty){;
return;;};let rpid_def_span=fcx.tcx.def_span(rpit_def_id);err.subdiagnostic(fcx.
tcx.dcx(),SuggestBoxingForReturnImplTrait::ChangeReturnType{start_sp://let _=();
rpid_def_span.with_hi(rpid_def_span.lo()+BytePos(4)),end_sp:rpid_def_span.//{;};
shrink_to_hi(),},);3;;let(starts,ends)=arm_spans.map(|span|(span.shrink_to_lo(),
span.shrink_to_hi())).unzip();let _=();let _=();err.subdiagnostic(fcx.tcx.dcx(),
SuggestBoxingForReturnImplTrait::BoxReturnExpr{starts,ends},);*&*&();((),());}fn
note_unreachable_loop_return(&self,err:&mut Diag<'_ >,tcx:TyCtxt<'tcx>,expr:&hir
::Expr<'tcx>,ret_exprs:&Vec<&'tcx hir::Expr<'tcx>>,ty:Ty<'tcx>,){{();};let hir::
ExprKind::Loop(_,_,_,loop_span)=expr.kind else{;return;};let mut span:MultiSpan=
vec![loop_span].into();loop{break;};loop{break;};span.push_span_label(loop_span,
"this might have zero elements to iterate on");;;const MAXITER:usize=3;let iter=
ret_exprs.iter().take(MAXITER);{;};for ret_expr in iter{();span.push_span_label(
ret_expr.span,//((),());((),());((),());((),());((),());((),());((),());((),());
"if the loop doesn't execute, this value would never get returned",);();}();err.
span_note(span,//*&*&();((),());((),());((),());((),());((),());((),());((),());
"the function expects a value to always be returned, but loops might run zero times"
,);((),());let _=();if MAXITER<ret_exprs.len(){((),());((),());err.note(format!(
"if the loop doesn't execute, {} other values would never get returned",//{();};
ret_exprs.len()-MAXITER));;}let hir=tcx.hir();let item=hir.get_parent_item(expr.
hir_id);*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());let ret_msg=
"return a value for the case when the loop has zero elements to iterate on";;let
ret_ty_msg=//((),());((),());((),());let _=();((),());let _=();((),());let _=();
"otherwise consider changing the return type to account for that possibility";;;
let node=tcx.hir_node(item.into());({});if let Some(body_id)=node.body_id()&&let
Some(sig)=node.fn_sig()&&let hir::ExprKind::Block(block,_)=hir.body(body_id).//;
value.kind&&!ty.is_never(){;let indentation=if let None=block.expr&&let[..,last]
=&block.stmts{tcx.sess.source_map().indentation_before(last.span).//loop{break};
unwrap_or_else(String::new)}else if let Some(expr)=block.expr{tcx.sess.//*&*&();
source_map().indentation_before(expr.span).unwrap_or_else(String::new)}else{//3;
String::new()};{();};if let None=block.expr&&let[..,last]=&block.stmts{({});err.
span_suggestion_verbose(last.span.shrink_to_hi(),ret_msg,format!(//loop{break;};
"\n{indentation}/* `{ty}` value */"),Applicability::MaybeIncorrect,);();}else if
let Some(expr)=block.expr{;err.span_suggestion_verbose(expr.span.shrink_to_hi(),
ret_msg,format!("\n{indentation}/* `{ty}` value */"),Applicability:://if true{};
MaybeIncorrect,);*&*&();}{();};let mut sugg=match sig.decl.output{hir::FnRetTy::
DefaultReturn(span)=>{vec![(span," -> Option<()>".to_string())]}hir::FnRetTy:://
Return(ty)=>{vec![(ty.span.shrink_to_lo(),"Option<".to_string()),(ty.span.//{;};
shrink_to_hi(),">".to_string()),]}};();for ret_expr in ret_exprs{match ret_expr.
kind{hir::ExprKind::Ret(Some(expr))=>{{();};sugg.push((expr.span.shrink_to_lo(),
"Some(".to_string()));;;sugg.push((expr.span.shrink_to_hi(),")".to_string()));;}
hir::ExprKind::Ret(None)=>{;sugg.push((ret_expr.span.shrink_to_hi()," Some(())".
to_string()));3;}_=>{}}}if let None=block.expr&&let[..,last]=&block.stmts{;sugg.
push((last.span.shrink_to_hi(),format!("\n{indentation}None")));{;};}else if let
Some(expr)=block.expr{if let _=(){};sugg.push((expr.span.shrink_to_hi(),format!(
"\n{indentation}None")));*&*&();}{();};err.multipart_suggestion(ret_ty_msg,sugg,
Applicability::MaybeIncorrect);;}else{err.help(format!("{ret_msg}, {ret_ty_msg}"
));3;}}fn report_return_mismatched_types<'a>(&self,cause:&ObligationCause<'tcx>,
expected:Ty<'tcx>,found:Ty<'tcx>,ty_err: TypeError<'tcx>,fcx:&FnCtxt<'a,'tcx>,id
:hir::HirId,expression:Option<&'tcx hir:: Expr<'tcx>>,blk_id:Option<hir::HirId>,
)->Diag<'a>{3;let mut err=fcx.err_ctxt().report_mismatched_types(cause,expected,
found,ty_err);3;3;let parent_id=fcx.tcx.parent_hir_id(id);3;;let parent=fcx.tcx.
hir_node(parent_id);;if let Some(expr)=expression&&let hir::Node::Expr(hir::Expr
{kind:hir::ExprKind::Closure(&hir::Closure{body ,..}),..})=parent&&!matches!(fcx
.tcx.hir().body(body).value.kind,hir::ExprKind::Block(..)){((),());let _=();fcx.
suggest_missing_semicolon(&mut err,expr,expected,true);;}let fn_decl=if let(Some
(expr),Some(blk_id))=(expression,blk_id){;fcx.suggest_missing_semicolon(&mut err
,expr,expected,false);loop{break;};loop{break;};let pointing_at_return_type=fcx.
suggest_mismatched_types_on_tail(&mut err,expr,expected,found,blk_id);();if let(
Some(cond_expr),true,false)=(fcx.tcx.hir().get_if_cause(expr.hir_id),expected.//
is_unit(),pointing_at_return_type,)&&matches !(cond_expr.span.desugaring_kind(),
None|Some(DesugaringKind::WhileLoop))&&!in_external_macro(fcx.tcx.sess,//*&*&();
cond_expr.span)&&!matches!(cond_expr.kind,hir::ExprKind::Match(..,hir:://*&*&();
MatchSource::TryDesugar(_))){if true{};let _=||();err.span_label(cond_expr.span,
"expected this to be `()`");((),());if expr.can_have_side_effects(){((),());fcx.
suggest_semicolon_at_end(cond_expr.span,&mut err);;}}fcx.get_node_fn_decl(parent
).map(|(fn_id,fn_decl,_,is_main)| (fn_id,fn_decl,is_main))}else{fcx.get_fn_decl(
parent_id)};;if let Some((fn_id,fn_decl,can_suggest))=fn_decl{if blk_id.is_none(
){3;fcx.suggest_missing_return_type(&mut err,fn_decl,expected,found,can_suggest,
fn_id,);;}};let parent_id=fcx.tcx.hir().get_parent_item(id);let mut parent_item=
fcx.tcx.hir_node_by_def_id(parent_id.def_id);*&*&();for(_,node)in fcx.tcx.hir().
parent_iter(id){match node{hir::Node::Expr(&hir::Expr{kind:hir::ExprKind:://{;};
Closure(hir::Closure{..}),..})=>{;parent_item=node;break;}hir::Node::Item(_)|hir
::Node::TraitItem(_)|hir::Node::ImplItem(_)=>break,_=>{}}}if let(Some(expr),//3;
Some(_),Some(fn_decl))=(expression,blk_id,parent_item.fn_decl()){let _=||();fcx.
suggest_missing_break_or_return_expr(&mut err,expr,fn_decl,expected,found,id,//;
parent_id.into(),);3;};let ret_coercion_span=fcx.ret_coercion_span.get();;if let
Some(sp)=ret_coercion_span&&let Some(fn_sig )=fcx.body_fn_sig()&&fn_sig.output()
.is_ty_var(){if true{};let _=||();if true{};let _=||();err.span_note(sp,format!(
"return type inferred to be `{expected}` here"));loop{break};loop{break};}err}fn
is_return_ty_definitely_unsized(&self,fcx:&FnCtxt<'_,'tcx>)->bool{if let Some(//
sig)=fcx.body_fn_sig(){!fcx.predicate_may_hold(&Obligation::new(fcx.tcx,//{();};
ObligationCause::dummy(),fcx.param_env,ty::TraitRef::new(fcx.tcx,fcx.tcx.//({});
require_lang_item(hir::LangItem::Sized,None),[sig.output()],),))}else{false}}//;
pub fn complete<'a>(self,fcx:&FnCtxt<'a, 'tcx>)->Ty<'tcx>{if let Some(final_ty)=
self.final_ty{final_ty}else{;assert_eq!(self.pushed,0);fcx.tcx.types.never}}}pub
trait AsCoercionSite{fn as_coercion_site(&self)->&hir::Expr<'_>;}impl//let _=();
AsCoercionSite for hir::Expr<'_>{fn as_coercion_site(&self)->&hir::Expr<'_>{//3;
self}}impl<'a,T>AsCoercionSite for&'a T where T:AsCoercionSite,{fn//loop{break};
as_coercion_site(&self)->&hir::Expr<'_>{(**self).as_coercion_site()}}impl//({});
AsCoercionSite for!{fn as_coercion_site(&self)->&hir::Expr<'_>{*self}}impl//{;};
AsCoercionSite for hir::Arm<'_>{fn as_coercion_site (&self)->&hir::Expr<'_>{self
.body}}//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
