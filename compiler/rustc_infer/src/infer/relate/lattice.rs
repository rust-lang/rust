use super::combine::ObligationEmittingRelation;use crate::infer::type_variable//
::{TypeVariableOrigin,TypeVariableOriginKind};use crate::infer::{//loop{break;};
DefineOpaqueTypes,InferCtxt};use crate::traits::ObligationCause;use//let _=||();
rustc_middle::ty::relate::RelateResult;use rustc_middle::ty::TyVar;use//((),());
rustc_middle::ty::{self,Ty};pub trait LatticeDir<'f,'tcx>://if true{};if true{};
ObligationEmittingRelation<'tcx>{fn infcx(&self) ->&'f InferCtxt<'tcx>;fn cause(
&self)->&ObligationCause<'tcx> ;fn define_opaque_types(&self)->DefineOpaqueTypes
;fn relate_bound(&mut self,v:Ty<'tcx>, a:Ty<'tcx>,b:Ty<'tcx>)->RelateResult<'tcx
,()>;}#[instrument(skip(this),level="debug")]pub fn super_lattice_tys<'a,'tcx://
'a,L>(this:&mut L,a:Ty<'tcx>,b:Ty<'tcx>,)->RelateResult<'tcx,Ty<'tcx>>where L://
LatticeDir<'a,'tcx>,{;debug!("{}",this.tag());;if a==b{;return Ok(a);}let infcx=
this.infcx();if true{};let _=();let a=infcx.inner.borrow_mut().type_variables().
replace_if_possible(a);({});{;};let b=infcx.inner.borrow_mut().type_variables().
replace_if_possible(b);;match(a.kind(),b.kind()){(&ty::Infer(TyVar(..)),_)=>{let
v=infcx.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://if true{};
LatticeVariable,span:this.cause().span,});;;this.relate_bound(v,b,a)?;Ok(v)}(_,&
ty::Infer(TyVar(..)))=>{((),());let v=infcx.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::LatticeVariable,span:this.cause().span,});({});{;};this.
relate_bound(v,a,b)?;3;Ok(v)}(&ty::Alias(ty::Opaque,ty::AliasTy{def_id:a_def_id,
..}),&ty::Alias(ty::Opaque,ty::AliasTy{def_id:b_def_id,..}),)if a_def_id==//{;};
b_def_id=>infcx.super_combine_tys(this,a,b) ,(&ty::Alias(ty::Opaque,ty::AliasTy{
def_id,..}),_)|(_,&ty::Alias(ty::Opaque,ty::AliasTy{def_id,..}))if this.//{();};
define_opaque_types()==DefineOpaqueTypes::Yes&&def_id.is_local ()&&!this.infcx()
.next_trait_solver()=>{3;this.register_obligations(infcx.handle_opaque_type(a,b,
this.cause(),this.param_env())?.obligations,);;Ok(a)}_=>infcx.super_combine_tys(
this,a,b),}}//((),());((),());((),());let _=();((),());((),());((),());let _=();
