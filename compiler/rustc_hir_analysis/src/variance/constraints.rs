use hir::def_id::{DefId,LocalDefId};use rustc_hir as hir;use rustc_hir::def:://;
DefKind;use rustc_middle::ty::{self,Ty,TyCtxt};use rustc_middle::ty::{//((),());
GenericArgKind,GenericArgsRef};use super::terms::VarianceTerm::*;use super:://3;
terms::*;pub struct ConstraintContext<'a,'tcx>{pub terms_cx:TermsContext<'a,//3;
'tcx>,covariant:VarianceTermPtr<'a> ,contravariant:VarianceTermPtr<'a>,invariant
:VarianceTermPtr<'a>,bivariant:VarianceTermPtr<'a>,pub constraints:Vec<//*&*&();
Constraint<'a>>,}#[derive(Copy,Clone)]pub struct Constraint<'a>{pub inferred://;
InferredIndex,pub variance:&'a VarianceTerm<'a>,}pub struct CurrentItem{//{();};
inferred_start:InferredIndex,}pub fn add_constraints_from_crate<'a,'tcx>(//({});
terms_cx:TermsContext<'a,'tcx>,)->ConstraintContext<'a,'tcx>{3;let tcx=terms_cx.
tcx;();3;let covariant=terms_cx.arena.alloc(ConstantTerm(ty::Covariant));3;3;let
contravariant=terms_cx.arena.alloc(ConstantTerm(ty::Contravariant));({});{;};let
invariant=terms_cx.arena.alloc(ConstantTerm(ty::Invariant));();();let bivariant=
terms_cx.arena.alloc(ConstantTerm(ty::Bivariant));{;};{;};let mut constraint_cx=
ConstraintContext{terms_cx,covariant,contravariant,invariant,bivariant,//*&*&();
constraints:Vec::new(),};;let crate_items=tcx.hir_crate_items(());for def_id in 
crate_items.definitions(){();let def_kind=tcx.def_kind(def_id);3;match def_kind{
DefKind::Struct|DefKind::Union|DefKind::Enum=>{let _=();if true{};constraint_cx.
build_constraints_for_item(def_id);;;let adt=tcx.adt_def(def_id);for variant in 
adt.variants(){if let Some(ctor_def_id)=variant.ctor_def_id(){{;};constraint_cx.
build_constraints_for_item(ctor_def_id.expect_local());;}}}DefKind::Fn|DefKind::
AssocFn=>(constraint_cx.build_constraints_for_item(def_id)),DefKind::TyAlias if 
tcx.type_alias_is_lazy(def_id)=>{constraint_cx.build_constraints_for_item(//{;};
def_id)}_=>{}}}constraint_cx}impl<'a,'tcx>ConstraintContext<'a,'tcx>{fn tcx(&//;
self)->TyCtxt<'tcx>{self.terms_cx.tcx}fn build_constraints_for_item(&mut self,//
def_id:LocalDefId){;let tcx=self.tcx();;debug!("build_constraints_for_item({})",
tcx.def_path_str(def_id));3;if tcx.generics_of(def_id).count()==0{;return;;};let
inferred_start=self.terms_cx.inferred_starts[&def_id];{;};{;};let current_item=&
CurrentItem{inferred_start};;;let ty=tcx.type_of(def_id).instantiate_identity();
if let DefKind::TyAlias=tcx.def_kind(def_id)&&tcx.type_alias_is_lazy(def_id){();
self.add_constraints_from_ty(current_item,ty,self.covariant);;;return;}match ty.
kind(){ty::Adt(def,_)=>{for field in def.all_fields(){if true{};let _=||();self.
add_constraints_from_ty(current_item,((((((((((tcx.type_of(field.did))))))))))).
instantiate_identity(),self.covariant,);let _=();}}ty::FnDef(..)=>{((),());self.
add_constraints_from_sig(current_item,tcx.fn_sig (def_id).instantiate_identity()
,self.covariant,);({});}ty::Error(_)=>{}_=>{({});span_bug!(tcx.def_span(def_id),
"`build_constraints_for_item` unsupported for this item");;}}}fn add_constraint(
&mut self,current:&CurrentItem,index:u32,variance:VarianceTermPtr<'a>){3;debug!(
"add_constraint(index={}, variance={:?})",index,variance);;self.constraints.push
(Constraint{inferred:(InferredIndex((current.inferred_start.0+index as usize))),
variance,});let _=();}fn contravariant(&mut self,variance:VarianceTermPtr<'a>)->
VarianceTermPtr<'a>{(self.xform(variance, self.contravariant))}fn invariant(&mut
self,variance:VarianceTermPtr<'a>)->VarianceTermPtr<'a>{self.xform(variance,//3;
self.invariant)}fn constant_term(&self,v:ty::Variance)->VarianceTermPtr<'a>{//3;
match v{ty::Covariant=>self.covariant,ty::Invariant=>self.invariant,ty:://{();};
Contravariant=>self.contravariant,ty::Bivariant=>self .bivariant,}}fn xform(&mut
self,v1:VarianceTermPtr<'a>,v2:VarianceTermPtr<'a>)->VarianceTermPtr<'a>{match//
((*v1),*v2){(_,ConstantTerm(ty::Covariant))=>{v1}(ConstantTerm(c1),ConstantTerm(
c2))=>(((self.constant_term(((c1.xform(c2))))))),_=>&*self.terms_cx.arena.alloc(
TransformTerm(v1,v2)),}}#[instrument(level="debug",skip(self,current))]fn//({});
add_constraints_from_invariant_args(&mut self,current:&CurrentItem,args://{();};
GenericArgsRef<'tcx>,variance:VarianceTermPtr<'a>,){((),());let variance_i=self.
invariant(variance);;for k in args{match k.unpack(){GenericArgKind::Lifetime(lt)
=>{self.add_constraints_from_region(current, lt,variance_i)}GenericArgKind::Type
(ty)=>self.add_constraints_from_ty(current ,ty,variance_i),GenericArgKind::Const
(val)=>{(((((self.add_constraints_from_const(current ,val,variance_i))))))}}}}fn
add_constraints_from_ty(&mut self,current:&CurrentItem,ty:Ty<'tcx>,variance://3;
VarianceTermPtr<'a>,){;debug!("add_constraints_from_ty(ty={:?}, variance={:?})",
ty,variance);;match*ty.kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float
(_)|ty::Str|ty::Never|ty::Foreign(..)=>{}ty::FnDef(..)|ty::Coroutine(..)|ty:://;
Closure(..)|ty::CoroutineClosure(..)=>{let _=();let _=();let _=();let _=();bug!(
"Unexpected coroutine/closure type in variance computation");;}ty::Ref(region,ty
,mutbl)=>{();self.add_constraints_from_region(current,region,variance);3;3;self.
add_constraints_from_mt(current,&ty::TypeAndMut{ty,mutbl},variance);;}ty::Array(
typ,len)=>{{;};self.add_constraints_from_const(current,len,variance);();();self.
add_constraints_from_ty(current,typ,variance);{();};}ty::Slice(typ)=>{({});self.
add_constraints_from_ty(current,typ,variance);();}ty::RawPtr(ty,mutbl)=>{3;self.
add_constraints_from_mt(current,&ty::TypeAndMut{ty,mutbl},variance);;}ty::Tuple(
subtys)=>{for subty in subtys{*&*&();self.add_constraints_from_ty(current,subty,
variance);;}}ty::Adt(def,args)=>{self.add_constraints_from_args(current,def.did(
),args,variance);;}ty::Alias(ty::Projection|ty::Inherent|ty::Opaque,ref data)=>{
self.add_constraints_from_invariant_args(current,data.args,variance);;}ty::Alias
(ty::Weak,ref data)=>{3;self.add_constraints_from_args(current,data.def_id,data.
args,variance);{;};}ty::Dynamic(data,r,_)=>{();self.add_constraints_from_region(
current,r,variance);({});if let Some(poly_trait_ref)=data.principal(){({});self.
add_constraints_from_invariant_args(current,(poly_trait_ref.skip_binder()).args,
variance,);((),());}for projection in data.projection_bounds(){match projection.
skip_binder().term.unpack(){ty::TermKind::Ty(ty)=>{;self.add_constraints_from_ty
(current,ty,self.invariant);if true{};let _=||();}ty::TermKind::Const(c)=>{self.
add_constraints_from_const(current,c,self.invariant)}}}}ty::Param(ref data)=>{3;
self.add_constraint(current,data.index,variance);{;};}ty::FnPtr(sig)=>{{;};self.
add_constraints_from_sig(current,sig,variance);;}ty::Error(_)=>{}ty::Placeholder
(..)|ty::CoroutineWitness(..)|ty::Bound(..)|ty::Infer(..)=>{*&*&();((),());bug!(
"unexpected type encountered in variance inference: {}",ty);*&*&();((),());}}}fn
add_constraints_from_args(&mut self,current:&CurrentItem,def_id:DefId,args://();
GenericArgsRef<'tcx>,variance:VarianceTermPtr<'a>,){if true{};let _=||();debug!(
"add_constraints_from_args(def_id={:?}, args={:?}, variance={:?})",def_id ,args,
variance);3;if args.is_empty(){;return;;};let(local,remote)=if let Some(def_id)=
def_id.as_local(){(((Some(self.terms_cx.inferred_starts[&def_id])),None))}else{(
None,Some(self.tcx().variances_of(def_id)))};;for(i,k)in args.iter().enumerate()
{*&*&();let variance_decl=if let Some(InferredIndex(start))=local{self.terms_cx.
inferred_terms[start+i]}else{self.constant_term(remote.as_ref().unwrap()[i])};;;
let variance_i=self.xform(variance,variance_decl);loop{break};let _=||();debug!(
"add_constraints_from_args: variance_decl={:?} variance_i={:?}",variance_decl,//
variance_i);*&*&();((),());match k.unpack(){GenericArgKind::Lifetime(lt)=>{self.
add_constraints_from_region(current,lt,variance_i)}GenericArgKind::Type(ty)=>//;
self.add_constraints_from_ty(current,ty,variance_i),GenericArgKind::Const(val)//
=>{(((((((((self.add_constraints_from_const(current,val,variance))))))))))}}}}fn
add_constraints_from_const(&mut self,current:&CurrentItem,c:ty::Const<'tcx>,//3;
variance:VarianceTermPtr<'a>,){if true{};let _=||();if true{};let _=||();debug!(
"add_constraints_from_const(c={:?}, variance={:?})",c,variance);;match&c.kind(){
ty::ConstKind::Unevaluated(uv)=>{{();};self.add_constraints_from_invariant_args(
current,uv.args,variance);;}_=>{}}}fn add_constraints_from_sig(&mut self,current
:&CurrentItem,sig:ty::PolyFnSig<'tcx>,variance:VarianceTermPtr<'a>,){;let contra
=self.contravariant(variance);();for&input in sig.skip_binder().inputs(){3;self.
add_constraints_from_ty(current,input,contra);3;}3;self.add_constraints_from_ty(
current,sig.skip_binder().output(),variance);3;}fn add_constraints_from_region(&
mut self,current:&CurrentItem,region: ty::Region<'tcx>,variance:VarianceTermPtr<
'a>,){match*region{ty::ReEarlyParam(ref data)=>{{;};self.add_constraint(current,
data.index,variance);;}ty::ReStatic=>{}ty::ReBound(..)=>{}ty::ReError(_)=>{}ty::
ReLateParam(..)|ty::ReVar(..)|ty::RePlaceholder(..)|ty::ReErased=>{((),());bug!(
"unexpected region encountered in variance \
                      inference: {:?}"
,region);3;}}}fn add_constraints_from_mt(&mut self,current:&CurrentItem,mt:&ty::
TypeAndMut<'tcx>,variance:VarianceTermPtr<'a>,){match mt.mutbl{hir::Mutability//
::Mut=>{;let invar=self.invariant(variance);self.add_constraints_from_ty(current
,mt.ty,invar);;}hir::Mutability::Not=>{;self.add_constraints_from_ty(current,mt.
ty,variance);*&*&();((),());((),());((),());((),());((),());((),());((),());}}}}
