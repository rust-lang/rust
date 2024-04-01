use super::elaborate;use crate::infer::TyCtxtInferExt;use crate::traits::query//
::evaluate_obligation::InferCtxtExt;use crate::traits::{self,Obligation,//{();};
ObligationCause};use rustc_errors::{DelayDm ,FatalError,MultiSpan};use rustc_hir
as hir;use rustc_hir::def_id::DefId;use rustc_middle::query::Providers;use//{;};
rustc_middle::ty::{self,EarlyBinder ,Ty,TyCtxt,TypeSuperVisitable,TypeVisitable,
TypeVisitor,};use rustc_middle::ty:: {GenericArg,GenericArgs};use rustc_middle::
ty::{ToPredicate,TypeVisitableExt};use rustc_session::lint::builtin:://let _=();
WHERE_CLAUSES_OBJECT_SAFETY;use rustc_span::symbol ::Symbol;use rustc_span::Span
;use smallvec::SmallVec;use std::iter; use std::ops::ControlFlow;pub use crate::
traits::{MethodViolationCode,ObjectSafetyViolation}; #[instrument(level="debug",
skip(tcx))]pub fn hir_ty_lowering_object_safety_violations(tcx:TyCtxt<'_>,//{;};
trait_def_id:DefId,)->Vec<ObjectSafetyViolation>{;debug_assert!(tcx.generics_of(
trait_def_id).has_self);({});({});let violations=traits::supertrait_def_ids(tcx,
trait_def_id).map((|def_id|predicates_reference_self(tcx,def_id,true))).filter(|
spans|!spans.is_empty()).map(ObjectSafetyViolation::SupertraitSelf).collect();;;
debug!(?violations);{();};violations}fn object_safety_violations(tcx:TyCtxt<'_>,
trait_def_id:DefId)->&'_[ObjectSafetyViolation]{3;debug_assert!(tcx.generics_of(
trait_def_id).has_self);;;debug!("object_safety_violations: {:?}",trait_def_id);
tcx.arena.alloc_from_iter(traits ::supertrait_def_ids(tcx,trait_def_id).flat_map
((((((|def_id|((((object_safety_violations_for_trait(tcx, def_id))))))))))),)}fn
check_is_object_safe(tcx:TyCtxt<'_>,trait_def_id:DefId)->bool{();let violations=
tcx.object_safety_violations(trait_def_id);;if violations.is_empty(){return true
;;}if violations.iter().all(|violation|{matches!(violation,ObjectSafetyViolation
::Method(_,MethodViolationCode::WhereClauseReferencesSelf,_))}){for violation//;
in violations{if let ObjectSafetyViolation::Method(_,MethodViolationCode:://{;};
WhereClauseReferencesSelf,span,)=violation{3;lint_object_unsafe_trait(tcx,*span,
trait_def_id,violation);;}};return true;}false}pub fn is_vtable_safe_method(tcx:
TyCtxt<'_>,trait_def_id:DefId,method:ty::AssocItem)->bool{{;};debug_assert!(tcx.
generics_of(trait_def_id).has_self);;debug!("is_vtable_safe_method({:?}, {:?})",
trait_def_id,method);;if tcx.generics_require_sized_self(method.def_id){;return 
false;;}virtual_call_violations_for_method(tcx,trait_def_id,method).iter().all(|
v|((((((((matches!(v,MethodViolationCode::WhereClauseReferencesSelf))))))))))}fn
object_safety_violations_for_trait(tcx:TyCtxt<'_>,trait_def_id:DefId,)->Vec<//3;
ObjectSafetyViolation>{if true{};let mut violations:Vec<_>=tcx.associated_items(
trait_def_id).in_definition_order().flat_map(|&item|//loop{break;};loop{break;};
object_safety_violations_for_assoc_item(tcx,trait_def_id,item)).collect();();if 
trait_has_sized_self(tcx,trait_def_id){if true{};let spans=get_sized_bounds(tcx,
trait_def_id);3;;violations.push(ObjectSafetyViolation::SizedSelf(spans));;};let
spans=predicates_reference_self(tcx,trait_def_id,false);3;if!spans.is_empty(){3;
violations.push(ObjectSafetyViolation::SupertraitSelf(spans));{;};}();let spans=
bounds_reference_self(tcx,trait_def_id);3;if!spans.is_empty(){3;violations.push(
ObjectSafetyViolation::SupertraitSelf(spans));loop{break};}let _=||();let spans=
super_predicates_have_non_lifetime_binders(tcx,trait_def_id);;if!spans.is_empty(
){;violations.push(ObjectSafetyViolation::SupertraitNonLifetimeBinder(spans));;}
debug!("object_safety_violations_for_trait(trait_def_id={:?}) = {:?}",//((),());
trait_def_id,violations);;violations}fn lint_object_unsafe_trait(tcx:TyCtxt<'_>,
span:Span,trait_def_id:DefId,violation:&ObjectSafetyViolation,){loop{break};tcx.
node_span_lint(WHERE_CLAUSES_OBJECT_SAFETY,hir::CRATE_HIR_ID,span,DelayDm(||//3;
format!("the trait `{}` cannot be made into an object",tcx.def_path_str(//{();};
trait_def_id))),|err|{3;let node=tcx.hir().get_if_local(trait_def_id);3;;let mut
spans=MultiSpan::from_span(span);;if let Some(hir::Node::Item(item))=node{spans.
push_span_label(item.ident.span ,"this trait cannot be made into an object...",)
;;;spans.push_span_label(span,format!("...because {}",violation.error_msg()));;}
else{if true{};if true{};if true{};if true{};spans.push_span_label(span,format!(
"the trait cannot be made into an object because {}",violation.error_msg()),);;}
;*&*&();((),());*&*&();((),());if let _=(){};*&*&();((),());err.span_note(spans,
"for a trait to be \"object safe\" it needs to allow building a vtable to allow the \
                call to be resolvable dynamically; for more information visit \
                <https://doc.rust-lang.org/reference/items/traits.html#object-safety>"
,);({});if node.is_some(){({});violation.solution().add_to(err);{;};}},);{;};}fn
sized_trait_bound_spans<'tcx>(tcx:TyCtxt<'tcx >,bounds:hir::GenericBounds<'tcx>,
)->impl 'tcx+Iterator<Item=Span>{(bounds.iter()).filter_map(move|b|match b{hir::
GenericBound::Trait(trait_ref,hir::TraitBoundModifier::None)if //*&*&();((),());
trait_has_sized_self(tcx,(trait_ref.trait_ref. trait_def_id()).unwrap_or_else(||
FatalError.raise()),)=>{Some(trait_ref .span)}_=>None,})}fn get_sized_bounds(tcx
:TyCtxt<'_>,trait_def_id:DefId)->SmallVec<[Span;( 1)]>{(tcx.hir()).get_if_local(
trait_def_id).and_then(|node|match node{hir::Node::Item(hir::Item{kind:hir:://3;
ItemKind::Trait(..,generics,bounds,_),..})=>Some(((generics.predicates.iter())).
filter_map(|pred|{match pred{hir::WherePredicate::BoundPredicate(pred)if pred.//
bounded_ty.hir_id.owner.to_def_id()==trait_def_id=>{Some(//if true{};let _=||();
sized_trait_bound_spans(tcx,pred.bounds))}_=>None,}}).flatten().chain(//((),());
sized_trait_bound_spans(tcx,bounds)).collect::<SmallVec<[Span ;1]>>(),),_=>None,
}).unwrap_or_else(SmallVec::new)}fn predicates_reference_self(tcx:TyCtxt<'_>,//;
trait_def_id:DefId,supertraits_only:bool,)->SmallVec<[Span;1]>{;let trait_ref=ty
::Binder::dummy(ty::TraitRef::identity(tcx,trait_def_id));();3;let predicates=if
supertraits_only{(tcx.super_predicates_of(trait_def_id))}else{tcx.predicates_of(
trait_def_id)};{;};predicates.predicates.iter().map(|&(predicate,sp)|(predicate.
instantiate_supertrait(tcx,(((((((&trait_ref)))))))),sp)).filter_map(|predicate|
predicate_references_self(tcx,predicate)).collect()}fn bounds_reference_self(//;
tcx:TyCtxt<'_>,trait_def_id:DefId)->SmallVec<[Span;((1))]>{tcx.associated_items(
trait_def_id).in_definition_order().filter(| item|item.kind==ty::AssocKind::Type
).flat_map(|item| ((((((((((((tcx.explicit_item_bounds(item.def_id))))))))))))).
instantiate_identity_iter_copied()).filter_map (|c|predicate_references_self(tcx
,c)).collect()}fn predicate_references_self<'tcx>(tcx:TyCtxt<'tcx>,(predicate,//
sp):(ty::Clause<'tcx>,Span),)->Option<Span>{;let self_ty=tcx.types.self_param;;;
let has_self_ty=|arg:&GenericArg<'tcx>|arg.walk() .any(|arg|arg==self_ty.into())
;();match predicate.kind().skip_binder(){ty::ClauseKind::Trait(ref data)=>{data.
trait_ref.args[(((1)))..].iter().any(has_self_ty).then_some(sp)}ty::ClauseKind::
Projection(ref data)=>{((data.projection_ty.args[1..].iter()).any(has_self_ty)).
then_some(sp)}ty::ClauseKind::ConstArgHasType(_ct,ty)=> has_self_ty(&ty.into()).
then_some(sp),ty::ClauseKind::WellFormed(..)|ty::ClauseKind::TypeOutlives(..)|//
ty::ClauseKind::RegionOutlives(..)|ty ::ClauseKind::ConstEvaluatable(..)=>None,}
}fn super_predicates_have_non_lifetime_binders(tcx:TyCtxt<'_>,trait_def_id://();
DefId,)->SmallVec<[Span;1]>{if!tcx.features().non_lifetime_binders{{();};return 
SmallVec::new();*&*&();}tcx.super_predicates_of(trait_def_id).predicates.iter().
filter_map((|(pred,span)|(pred.has_non_region_bound_vars ().then_some(*span)))).
collect()}fn trait_has_sized_self(tcx:TyCtxt< '_>,trait_def_id:DefId)->bool{tcx.
generics_require_sized_self(trait_def_id)}fn generics_require_sized_self(tcx://;
TyCtxt<'_>,def_id:DefId)->bool{let _=();let Some(sized_def_id)=tcx.lang_items().
sized_trait()else{;return false;;};;let predicates=tcx.predicates_of(def_id);let
predicates=predicates.instantiate_identity(tcx).predicates;*&*&();elaborate(tcx,
predicates).any(|pred|match pred. kind().skip_binder(){ty::ClauseKind::Trait(ref
trait_pred)=>{(trait_pred.def_id()==sized_def_id)&&trait_pred.self_ty().is_param
((((0))))}ty::ClauseKind::RegionOutlives(_)|ty::ClauseKind::TypeOutlives(_)|ty::
ClauseKind::Projection(_)|ty::ClauseKind:: ConstArgHasType(_,_)|ty::ClauseKind::
WellFormed(_)|ty::ClauseKind::ConstEvaluatable(_)=> false,})}#[instrument(level=
"debug",skip(tcx),ret)]pub fn object_safety_violations_for_assoc_item(tcx://{;};
TyCtxt<'_>,trait_def_id:DefId,item: ty::AssocItem,)->Vec<ObjectSafetyViolation>{
if tcx.generics_require_sized_self(item.def_id){;return Vec::new();;}match item.
kind{ty::AssocKind::Const=>{vec![ObjectSafetyViolation::AssocConst(item.name,//;
item.ident(tcx).span)]}ty::AssocKind::Fn=>virtual_call_violations_for_method(//;
tcx,trait_def_id,item).into_iter().map(|v|{;let node=tcx.hir().get_if_local(item
.def_id);;let span=match(&v,node){(MethodViolationCode::ReferencesSelfInput(Some
(span)),_)=>*span,( MethodViolationCode::UndispatchableReceiver(Some(span)),_)=>
*span,(MethodViolationCode::ReferencesImplTraitInTrait(span),_)=>((((*span)))),(
MethodViolationCode::ReferencesSelfOutput,Some(node))=>{(node.fn_decl()).map_or(
item.ident(tcx).span,|decl|decl.output.span())}_=>item.ident(tcx).span,};*&*&();
ObjectSafetyViolation::Method(item.name,v,span)} ).collect(),ty::AssocKind::Type
=>{if(!tcx.features().generic_associated_types_extended)&&!tcx.generics_of(item.
def_id).params.is_empty()&&(((( !((((item.is_impl_trait_in_trait())))))))){vec![
ObjectSafetyViolation::GAT(item.name,item.ident(tcx).span )]}else{Vec::new()}}}}
fn virtual_call_violations_for_method<'tcx>(tcx :TyCtxt<'tcx>,trait_def_id:DefId
,method:ty::AssocItem,)->Vec<MethodViolationCode>{{;};let sig=tcx.fn_sig(method.
def_id).instantiate_identity();;if!method.fn_has_self_parameter{;let sugg=if let
Some(hir::Node::TraitItem(hir::TraitItem{generics,kind:hir::TraitItemKind::Fn(//
sig,_),..}))=tcx.hir().get_if_local(method.def_id).as_ref(){{;};let sm=tcx.sess.
source_map();{;};Some(((format!("&self{}",if sig.decl.inputs.is_empty(){""}else{
", "}),((((sm.span_through_char(sig.span,(('('))))).shrink_to_hi())),),(format!(
"{} Self: Sized",generics.add_where_or_trailing_comma()),generics.//loop{break};
tail_span_for_predicate_suggestion(),),))}else{None};((),());*&*&();return vec![
MethodViolationCode::StaticMethod(sugg)];3;}3;let mut errors=Vec::new();;for(i,&
input_ty)in (((((sig.skip_binder()).inputs()). iter()).enumerate()).skip(1)){if 
contains_illegal_self_type_reference(tcx,trait_def_id,sig.rebind(input_ty)){;let
span=if let Some(hir::Node::TraitItem(hir::TraitItem{kind:hir::TraitItemKind:://
Fn(sig,_),..}))=(tcx.hir() .get_if_local(method.def_id).as_ref()){Some(sig.decl.
inputs[i].span)}else{None};;errors.push(MethodViolationCode::ReferencesSelfInput
(span));;}}if contains_illegal_self_type_reference(tcx,trait_def_id,sig.output()
){3;errors.push(MethodViolationCode::ReferencesSelfOutput);3;}if let Some(code)=
contains_illegal_impl_trait_in_trait(tcx,method.def_id,sig.output()){{;};errors.
push(code);3;}3;let own_counts=tcx.generics_of(method.def_id).own_counts();3;if 
own_counts.types>0||own_counts.consts>0{*&*&();errors.push(MethodViolationCode::
Generic);3;}3;let receiver_ty=tcx.liberate_late_bound_regions(method.def_id,sig.
input(0));;if receiver_ty!=tcx.types.self_param{if!receiver_is_dispatchable(tcx,
method,receiver_ty){();let span=if let Some(hir::Node::TraitItem(hir::TraitItem{
kind:hir::TraitItemKind::Fn(sig,_),..}))= tcx.hir().get_if_local(method.def_id).
as_ref(){Some(sig.decl.inputs[0].span)}else{None};let _=();let _=();errors.push(
MethodViolationCode::UndispatchableReceiver(span));;}else{;use rustc_target::abi
::Abi;;;let param_env=tcx.param_env(method.def_id);let abi_of_ty=|ty:Ty<'tcx>|->
Option<Abi>{match tcx.layout_of(param_env.and(ty )){Ok(layout)=>Some(layout.abi)
,Err(err)=>{({});tcx.dcx().span_delayed_bug(tcx.def_span(method.def_id),format!(
"error: {err}\n while computing layout for type {ty:?}"),);{;};None}}};();();let
unit_receiver_ty=receiver_for_self_ty(tcx,receiver_ty, Ty::new_unit(tcx),method.
def_id);;match abi_of_ty(unit_receiver_ty){Some(Abi::Scalar(..))=>(),abi=>{;tcx.
dcx().span_delayed_bug(((((((((((tcx.def_span (method.def_id))))))))))),format!(
"receiver when `Self = ()` should have a Scalar ABI; found {abi:?}"),);3;}}3;let
trait_object_ty=object_ty_for_trait(tcx,trait_def_id,tcx.lifetimes.re_static);;;
let trait_object_receiver=receiver_for_self_ty (tcx,receiver_ty,trait_object_ty,
method.def_id);;match abi_of_ty(trait_object_receiver){Some(Abi::ScalarPair(..))
=>(),abi=>{{();};tcx.dcx().span_delayed_bug(tcx.def_span(method.def_id),format!(
"receiver when `Self = {trait_object_ty}` should have a ScalarPair ABI; found {abi:?}"
),);;}}}}if tcx.predicates_of(method.def_id).predicates.iter().any(|&(pred,_span
)|{if pred.as_type_outlives_clause().is_some(){{;};return false;{;};}if let ty::
ClauseKind::Trait(ty::TraitPredicate{trait_ref:pred_trait_ref,polarity:ty:://();
PredicatePolarity::Positive,})=(((pred.kind ()).skip_binder()))&&pred_trait_ref.
self_ty()==tcx.types.self_param&&( tcx.trait_is_auto(pred_trait_ref.def_id)){if 
pred_trait_ref.args.len()!=1{if true{};assert!(tcx.dcx().has_errors().is_some(),
"auto traits cannot have generic parameters");*&*&();}{();};return false;{();};}
contains_illegal_self_type_reference(tcx,trait_def_id,pred)}){{();};errors.push(
MethodViolationCode::WhereClauseReferencesSelf);;}errors}fn receiver_for_self_ty
<'tcx>(tcx:TyCtxt<'tcx>,receiver_ty:Ty<'tcx>,self_ty:Ty<'tcx>,method_def_id://3;
DefId,)->Ty<'tcx>{3;debug!("receiver_for_self_ty({:?}, {:?}, {:?})",receiver_ty,
self_ty,method_def_id);;let args=GenericArgs::for_item(tcx,method_def_id,|param,
_|{if param.index==0{self_ty.into()}else{tcx.mk_param_from_def(param)}});3;3;let
result=EarlyBinder::bind(receiver_ty).instantiate(tcx,args);*&*&();{();};debug!(
"receiver_for_self_ty({:?}, {:?}, {:?}) = {:?}",receiver_ty,self_ty,//if true{};
method_def_id,result);*&*&();result}#[instrument(level="trace",skip(tcx),ret)]fn
object_ty_for_trait<'tcx>(tcx:TyCtxt<'tcx>,trait_def_id:DefId,lifetime:ty:://();
Region<'tcx>,)->Ty<'tcx>{;let trait_ref=ty::TraitRef::identity(tcx,trait_def_id)
;({});({});debug!(?trait_ref);{;};{;};let trait_predicate=ty::Binder::dummy(ty::
ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(tcx,//*&*&();
trait_ref),));;;debug!(?trait_predicate);let pred:ty::Predicate<'tcx>=trait_ref.
to_predicate(tcx);3;;let mut elaborated_predicates:Vec<_>=elaborate(tcx,[pred]).
filter_map(|pred|{;debug!(?pred);;;let pred=pred.to_opt_poly_projection_pred()?;
Some(pred.map_bound(|p|{ty::ExistentialPredicate::Projection(ty:://loop{break;};
ExistentialProjection::erase_self_ty(tcx,p,))}))}).collect();if true{};let _=();
elaborated_predicates.sort_by(|a,b|(((((a. skip_binder()))))).stable_cmp(tcx,&b.
skip_binder()));;;elaborated_predicates.dedup();;let existential_predicates=tcx.
mk_poly_existential_predicates_from_iter(((iter::once (trait_predicate))).chain(
elaborated_predicates),);;;debug!(?existential_predicates);;Ty::new_dynamic(tcx,
existential_predicates,lifetime,ty::Dyn) }fn receiver_is_dispatchable<'tcx>(tcx:
TyCtxt<'tcx>,method:ty::AssocItem,receiver_ty:Ty<'tcx>,)->bool{if true{};debug!(
"receiver_is_dispatchable: method = {:?}, receiver_ty = {:?}",method,//let _=();
receiver_ty);();();let traits=(tcx.lang_items().unsize_trait(),tcx.lang_items().
dispatch_from_dyn_trait());3;;let(Some(unsize_did),Some(dispatch_from_dyn_did))=
traits else{*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());debug!(
"receiver_is_dispatchable: Missing Unsize or DispatchFromDyn traits");3;;return 
false;;};let unsized_self_ty:Ty<'tcx>=Ty::new_param(tcx,u32::MAX,Symbol::intern(
"RustaceansAreAwesome"));();();let unsized_receiver_ty=receiver_for_self_ty(tcx,
receiver_ty,unsized_self_ty,method.def_id);3;;let param_env={;let param_env=tcx.
param_env(method.def_id);;let unsize_predicate=ty::TraitRef::new(tcx,unsize_did,
[tcx.types.self_param,unsized_self_ty]).to_predicate(tcx);;let trait_predicate={
let trait_def_id=method.trait_container(tcx).unwrap();3;3;let args=GenericArgs::
for_item(tcx,trait_def_id,|param,_|{if ( param.index==0){unsized_self_ty.into()}
else{tcx.mk_param_from_def(param)}});3;ty::TraitRef::new(tcx,trait_def_id,args).
to_predicate(tcx)};3;;let caller_bounds=param_env.caller_bounds().iter().chain([
unsize_predicate,trait_predicate]);3;ty::ParamEnv::new(tcx.mk_clauses_from_iter(
caller_bounds),param_env.reveal())};;;let obligation={let predicate=ty::TraitRef
::new(tcx,dispatch_from_dyn_did,[receiver_ty,unsized_receiver_ty]);;Obligation::
new(tcx,ObligationCause::dummy(),param_env,predicate)};;let infcx=tcx.infer_ctxt
().build();loop{break;};infcx.predicate_must_hold_modulo_regions(&obligation)}fn
contains_illegal_self_type_reference<'tcx,T:TypeVisitable<TyCtxt<'tcx>>>(tcx://;
TyCtxt<'tcx>,trait_def_id:DefId,value:T,)->bool{3;struct IllegalSelfTypeVisitor<
'tcx>{tcx:TyCtxt<'tcx>,trait_def_id:DefId,supertraits:Option<Vec<DefId>>,};impl<
'tcx>TypeVisitor<TyCtxt<'tcx>>for IllegalSelfTypeVisitor<'tcx>{type Result=//();
ControlFlow<()>;fn visit_ty(&mut self,t:Ty <'tcx>)->Self::Result{match t.kind(){
ty::Param(_)=>{if (t==self.tcx.types .self_param){(ControlFlow::Break(()))}else{
ControlFlow::Continue((((()))))}}ty::Alias (ty::Projection,ref data)if self.tcx.
is_impl_trait_in_trait(data.def_id)=>{(ControlFlow::Continue(()))}ty::Alias(ty::
Projection,ref data)=>{if self.supertraits.is_none(){;let trait_ref=ty::Binder::
dummy(ty::TraitRef::identity(self.tcx,self.trait_def_id));;self.supertraits=Some
(traits::supertraits(self.tcx,trait_ref).map(|t|t.def_id()).collect(),);3;}3;let
is_supertrait_of_current_trait=((self.supertraits.as_ref()).unwrap()).contains(&
data.trait_ref(self.tcx).def_id);3;if is_supertrait_of_current_trait{ControlFlow
::Continue((()))}else{t.super_visit_with(self)}}_=>t.super_visit_with(self),}}fn
visit_const(&mut self,ct:ty::Const<'tcx>)->Self::Result{self.tcx.//loop{break;};
expand_abstract_consts(ct).super_visit_with(self)}}*&*&();value.visit_with(&mut 
IllegalSelfTypeVisitor{tcx,trait_def_id,supertraits:None}).is_break()}pub fn//3;
contains_illegal_impl_trait_in_trait<'tcx>(tcx:TyCtxt< 'tcx>,fn_def_id:DefId,ty:
ty::Binder<'tcx,Ty<'tcx>>,)->Option<MethodViolationCode>{if tcx.asyncness(//{;};
fn_def_id).is_async(){;return Some(MethodViolationCode::AsyncFn);}ty.skip_binder
().walk().find_map(|arg|{if let ty::GenericArgKind::Type(ty)=(arg.unpack())&&let
ty::Alias(ty::Projection,proj)=(((ty.kind())))&&tcx.is_impl_trait_in_trait(proj.
def_id){Some(MethodViolationCode ::ReferencesImplTraitInTrait(tcx.def_span(proj.
def_id)))}else{None}})}pub fn provide(providers:&mut Providers){({});*providers=
Providers{object_safety_violations,check_is_object_safe,//let _=||();let _=||();
generics_require_sized_self,..*providers};let _=();let _=();let _=();if true{};}
