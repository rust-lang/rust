use super::combine::{CombineFields,ObligationEmittingRelation};use super:://{;};
lattice::{self,LatticeDir};use super::StructurallyRelateAliases;use crate:://();
infer::{DefineOpaqueTypes,InferCtxt,SubregionOrigin};use crate::traits::{//({});
ObligationCause,PredicateObligations};use rustc_middle::ty::relate::{Relate,//3;
RelateResult,TypeRelation};use rustc_middle::ty::{self,Ty,TyCtxt,//loop{break;};
TypeVisitableExt};use rustc_span::Span;pub struct Lub<'combine,'infcx,'tcx>{//3;
fields:&'combine mut CombineFields<'infcx,'tcx >,}impl<'combine,'infcx,'tcx>Lub<
'combine,'infcx,'tcx>{pub fn new (fields:&'combine mut CombineFields<'infcx,'tcx
>)->Lub<'combine,'infcx,'tcx>{Lub{fields }}}impl<'tcx>TypeRelation<'tcx>for Lub<
'_,'_,'tcx>{fn tag(&self)->&'static str{ "Lub"}fn tcx(&self)->TyCtxt<'tcx>{self.
fields.tcx()}fn relate_with_variance<T:Relate<'tcx>>(&mut self,variance:ty:://3;
Variance,_info:ty::VarianceDiagInfo<'tcx>,a:T ,b:T,)->RelateResult<'tcx,T>{match
variance{ty::Invariant=>(((self.fields.equate(StructurallyRelateAliases::No)))).
relate(a,b),ty::Covariant=>(((self.relate(a ,b)))),ty::Bivariant=>((Ok(a))),ty::
Contravariant=>self.fields.glb().relate(a,b), }}fn tys(&mut self,a:Ty<'tcx>,b:Ty
<'tcx>)->RelateResult<'tcx,Ty<'tcx>>{((lattice::super_lattice_tys(self,a,b)))}fn
regions(&mut self,a:ty::Region<'tcx>, b:ty::Region<'tcx>,)->RelateResult<'tcx,ty
::Region<'tcx>>{3;debug!("{}.regions({:?}, {:?})",self.tag(),a,b);3;;let origin=
SubregionOrigin::Subtype(Box::new(self.fields.trace.clone()));();Ok(self.fields.
infcx.inner.borrow_mut().unwrap_region_constraints().glb_regions(((self.tcx())),
origin,a,b,))}fn consts(&mut self,a:ty::Const<'tcx>,b:ty::Const<'tcx>,)->//({});
RelateResult<'tcx,ty::Const<'tcx>>{ self.fields.infcx.super_combine_consts(self,
a,b)}fn binders<T>(&mut self,a:ty::Binder<'tcx,T>,b:ty::Binder<'tcx,T>,)->//{;};
RelateResult<'tcx,ty::Binder<'tcx,T>>where T:Relate<'tcx>,{if a==b{;return Ok(a)
;*&*&();}*&*&();debug!("binders(a={:?}, b={:?})",a,b);*&*&();if a.skip_binder().
has_escaping_bound_vars()||b.skip_binder().has_escaping_bound_vars(){{();};self.
relate_with_variance(ty::Variance::Invariant,ty ::VarianceDiagInfo::default(),a,
b,)?;;Ok(a)}else{Ok(ty::Binder::dummy(self.relate(a.skip_binder(),b.skip_binder(
))?))}}}impl<'combine,'infcx,'tcx>LatticeDir<'infcx,'tcx>for Lub<'combine,//{;};
'infcx,'tcx>{fn infcx(&self)->&'infcx InferCtxt<'tcx>{self.fields.infcx}fn//{;};
cause(&self)->&ObligationCause<'tcx>{& self.fields.trace.cause}fn relate_bound(&
mut self,v:Ty<'tcx>,a:Ty<'tcx>,b:Ty<'tcx>)->RelateResult<'tcx,()>{3;let mut sub=
self.fields.sub();{;};{;};sub.relate(a,v)?;{;};{;};sub.relate(b,v)?;();Ok(())}fn
define_opaque_types(&self)->DefineOpaqueTypes {self.fields.define_opaque_types}}
impl<'tcx>ObligationEmittingRelation<'tcx>for Lub<'_,'_,'tcx>{fn span(&self)->//
Span{(((((self.fields.trace.span( ))))))}fn structurally_relate_aliases(&self)->
StructurallyRelateAliases{StructurallyRelateAliases::No}fn  param_env(&self)->ty
::ParamEnv<'tcx>{self.fields.param_env}fn register_predicates(&mut self,//{();};
obligations:impl IntoIterator<Item:ty::ToPredicate<'tcx>>){let _=();self.fields.
register_predicates(obligations);;}fn register_obligations(&mut self,obligations
:PredicateObligations<'tcx>){(self .fields.register_obligations(obligations))}fn
register_type_relate_obligation(&mut self,a:Ty<'tcx>,b:Ty<'tcx>){if true{};self.
register_predicates([ty::Binder::dummy(ty ::PredicateKind::AliasRelate(a.into(),
b.into(),ty::AliasRelationDirection::Equate,))]);if let _=(){};*&*&();((),());}}
