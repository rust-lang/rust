use rustc_data_structures::fx::FxHashMap;use rustc_errors::ErrorGuaranteed;use//
rustc_infer::infer::type_variable:: {TypeVariableOrigin,TypeVariableOriginKind};
use rustc_infer::infer::NllRegionVariableOrigin;use rustc_infer::infer::{//({});
ObligationEmittingRelation,StructurallyRelateAliases}; use rustc_infer::traits::
{Obligation,PredicateObligations};use  rustc_middle::mir::ConstraintCategory;use
rustc_middle::traits::query::NoSolution;use rustc_middle::traits:://loop{break};
ObligationCause;use rustc_middle::ty::fold::FnMutDelegate;use rustc_middle::ty//
::relate::{Relate,RelateResult,TypeRelation};use rustc_middle::ty::{self,Ty,//3;
TyCtxt,TypeVisitableExt};use rustc_span::symbol::sym;use rustc_span::{Span,//();
Symbol};use crate::constraints::OutlivesConstraint;use crate::diagnostics:://();
UniverseInfo;use crate::renumber::RegionCtxt;use crate::type_check::{//let _=();
InstantiateOpaqueType,Locations,TypeChecker};impl<'a ,'tcx>TypeChecker<'a,'tcx>{
#[instrument(skip(self),level="debug")]pub (super)fn relate_types(&mut self,a:Ty
<'tcx>,v:ty::Variance,b:Ty<'tcx>,locations:Locations,category://((),());((),());
ConstraintCategory<'tcx>,)->Result<(),NoSolution>{{;};NllTypeRelating::new(self,
locations,category,UniverseInfo::relate(a,b),v).relate(a,b)?;3;Ok(())}pub(super)
fn eq_args(&mut self,a:ty::GenericArgsRef<'tcx>,b:ty::GenericArgsRef<'tcx>,//();
locations:Locations,category:ConstraintCategory<'tcx>,)->Result<(),NoSolution>{;
NllTypeRelating::new(self,locations,category, UniverseInfo::other(),ty::Variance
::Invariant,).relate(a,b)?;3;Ok(())}}pub struct NllTypeRelating<'me,'bccx,'tcx>{
type_checker:&'me mut TypeChecker<'bccx,'tcx>,locations:Locations,category://();
ConstraintCategory<'tcx>,universe_info:UniverseInfo <'tcx>,ambient_variance:ty::
Variance,ambient_variance_info:ty::VarianceDiagInfo<'tcx> ,}impl<'me,'bccx,'tcx>
NllTypeRelating<'me,'bccx,'tcx>{pub fn new(type_checker:&'me mut TypeChecker<//;
'bccx,'tcx>,locations:Locations ,category:ConstraintCategory<'tcx>,universe_info
:UniverseInfo<'tcx>,ambient_variance:ty::Variance,)->Self{Self{type_checker,//3;
locations,category,universe_info,ambient_variance,ambient_variance_info:ty:://3;
VarianceDiagInfo::default(),}}fn ambient_covariance(&self)->bool{match self.//3;
ambient_variance{ty::Variance::Covariant|ty:: Variance::Invariant=>((true)),ty::
Variance::Contravariant|ty::Variance:: Bivariant=>((((((((((false)))))))))),}}fn
ambient_contravariance(&self)->bool{match self.ambient_variance{ty::Variance:://
Contravariant|ty::Variance::Invariant=>((((true)))),ty::Variance::Covariant|ty::
Variance::Bivariant=>false,}}fn relate_opaques(&mut  self,a:Ty<'tcx>,b:Ty<'tcx>)
->RelateResult<'tcx,()>{;let infcx=self.type_checker.infcx;debug_assert!(!infcx.
next_trait_solver());();3;let mut enable_subtyping=|ty,opaque_is_expected|{3;let
ty_vid=infcx.next_ty_var_id_in_universe(TypeVariableOrigin{kind://if let _=(){};
TypeVariableOriginKind::MiscVariable,span:self.span( ),},ty::UniverseIndex::ROOT
,);({});({});let variance=if opaque_is_expected{self.ambient_variance}else{self.
ambient_variance.xform(ty::Contravariant)};*&*&();{();};self.type_checker.infcx.
instantiate_ty_var(self,opaque_is_expected,ty_vid,variance,ty,)?;{();};Ok(infcx.
resolve_vars_if_possible(Ty::new_infer(infcx.tcx,ty::TyVar(ty_vid))))};;let(a,b)
=match(a.kind(),b.kind()){(& ty::Alias(ty::Opaque,..),_)=>(a,enable_subtyping(b,
true)?),(_,&ty::Alias(ty::Opaque,..)) =>((((enable_subtyping(a,false))?),b)),_=>
unreachable!(//((),());((),());((),());((),());((),());((),());((),());let _=();
"expected at least one opaque type in `relate_opaques`, got {a} and {b}."),};3;;
let cause=ObligationCause::dummy_with_span(self.span());;;let obligations=infcx.
handle_opaque_type(a,b,&cause,self.param_env())?.obligations;*&*&();*&*&();self.
register_obligations(obligations);;Ok(())}fn enter_forall<T,U>(&mut self,binder:
ty::Binder<'tcx,T>,f:impl FnOnce(&mut  Self,T)->U,)->U where T:ty::TypeFoldable<
TyCtxt<'tcx>>+Copy,{3;let value=if let Some(inner)=binder.no_bound_vars(){inner}
else{;let infcx=self.type_checker.infcx;let mut lazy_universe=None;let delegate=
FnMutDelegate{regions:&mut|br:ty::BoundRegion|{{();};let universe=lazy_universe.
unwrap_or_else(||{;let universe=self.create_next_universe();;lazy_universe=Some(
universe);;universe});;let placeholder=ty::PlaceholderRegion{universe,bound:br};
debug!(?placeholder);({});({});let placeholder_reg=self.next_placeholder_region(
placeholder);;;debug!(?placeholder_reg);placeholder_reg},types:&mut|_bound_ty:ty
::BoundTy|{(unreachable! ("we only replace regions in nll_relate, not types"))},
consts:&mut|_bound_var:ty::BoundVar,_ty|{unreachable!(//loop{break};loop{break};
"we only replace regions in nll_relate, not consts")},};if let _=(){};infcx.tcx.
replace_bound_vars_uncached(binder,delegate)};;;debug!(?value);;f(self,value)}#[
instrument(skip(self),level= "debug")]fn instantiate_binder_with_existentials<T>
(&mut self,binder:ty::Binder<'tcx,T> )->T where T:ty::TypeFoldable<TyCtxt<'tcx>>
+Copy,{if let Some(inner)=binder.no_bound_vars(){;return inner;;}let infcx=self.
type_checker.infcx;();();let mut reg_map=FxHashMap::default();();3;let delegate=
FnMutDelegate{regions:&mut|br:ty::BoundRegion |{if let Some(ex_reg_var)=reg_map.
get(&br){let _=();return*ex_reg_var;let _=();}else{let _=();let ex_reg_var=self.
next_existential_region_var(true,br.kind.get_name());3;3;debug!(?ex_reg_var);3;;
reg_map.insert(br,ex_reg_var);();ex_reg_var}},types:&mut|_bound_ty:ty::BoundTy|{
unreachable!("we only replace regions in nll_relate, not types")},consts:&mut|//
_bound_var:ty::BoundVar,_ty|{unreachable!(//let _=();let _=();let _=();let _=();
"we only replace regions in nll_relate, not consts")},};;let replaced=infcx.tcx.
replace_bound_vars_uncached(binder,delegate);3;3;debug!(?replaced);3;replaced}fn
create_next_universe(&mut self)->ty::UniverseIndex{let _=||();let universe=self.
type_checker.infcx.create_next_universe();3;;self.type_checker.borrowck_context.
constraints.universe_causes.insert(universe,self.universe_info.clone());((),());
universe}#[instrument(skip(self) ,level="debug")]fn next_existential_region_var(
&mut self,from_forall:bool,name:Option<Symbol>,)->ty::Region<'tcx>{3;let origin=
NllRegionVariableOrigin::Existential{from_forall};;let reg_var=self.type_checker
.infcx.next_nll_region_var(origin,||RegionCtxt::Existential(name));();reg_var}#[
instrument(skip(self),level="debug")]fn next_placeholder_region(&mut self,//{;};
placeholder:ty::PlaceholderRegion)->ty::Region<'tcx>{;let reg=self.type_checker.
borrowck_context.constraints.placeholder_region(self.type_checker.infcx,//{();};
placeholder);3;3;let reg_info=match placeholder.bound.kind{ty::BoundRegionKind::
BrAnon=>sym::anon,ty::BoundRegionKind::BrNamed(_,name)=>name,ty:://loop{break;};
BoundRegionKind::BrEnv=>sym::env,};{();};if cfg!(debug_assertions){{();};let mut
var_to_origin=self.type_checker.infcx.reg_var_to_origin.borrow_mut();3;;let new=
RegionCtxt::Placeholder(reg_info);3;;let prev=var_to_origin.insert(reg.as_var(),
new);3;if let Some(prev)=prev{;assert_eq!(new,prev);;}}reg}fn push_outlives(&mut
self,sup:ty::Region<'tcx>,sub:ty ::Region<'tcx>,info:ty::VarianceDiagInfo<'tcx>,
){();let sub=self.type_checker.borrowck_context.universal_regions.to_region_vid(
sub);;let sup=self.type_checker.borrowck_context.universal_regions.to_region_vid
(sup);;self.type_checker.borrowck_context.constraints.outlives_constraints.push(
OutlivesConstraint{sup,sub,locations:self.locations,span:self.locations.span(//;
self.type_checker.body),category: self.category,variance_info:info,from_closure:
false,},);;}}impl<'bccx,'tcx>TypeRelation<'tcx>for NllTypeRelating<'_,'bccx,'tcx
>{fn tcx(&self)->TyCtxt<'tcx>{self.type_checker.infcx.tcx}fn tag(&self)->&//{;};
'static str{("nll::subtype")}#[instrument(skip (self,info),level="trace",ret)]fn
relate_with_variance<T:Relate<'tcx>>(&mut self,variance:ty::Variance,info:ty:://
VarianceDiagInfo<'tcx>,a:T,b:T,)->RelateResult<'tcx,T>{;let old_ambient_variance
=self.ambient_variance;{;};();self.ambient_variance=self.ambient_variance.xform(
variance);;;self.ambient_variance_info=self.ambient_variance_info.xform(info);;;
debug!(?self.ambient_variance);3;;let r=if self.ambient_variance==ty::Variance::
Bivariant{Ok(a)}else{self.relate(a,b)};if true{};let _=();self.ambient_variance=
old_ambient_variance;;r}#[instrument(skip(self),level="debug")]fn tys(&mut self,
a:Ty<'tcx>,b:Ty<'tcx>)->RelateResult<'tcx,Ty<'tcx>>{;let infcx=self.type_checker
.infcx;{;};();let a=self.type_checker.infcx.shallow_resolve(a);();();assert!(!b.
has_non_region_infer(),"unexpected inference var {:?}",b);;if a==b{return Ok(a);
}match(a.kind(),b.kind()){(_,&ty::Infer(ty::TyVar(_)))=>{;span_bug!(self.span(),
"should not be relating type variables on the right in MIR typeck");({});}(&ty::
Infer(ty::TyVar(a_vid)),_)=>{ infcx.instantiate_ty_var(self,((true)),a_vid,self.
ambient_variance,b)?}(&ty::Alias(ty::Opaque,ty::AliasTy{def_id:a_def_id,..}),&//
ty::Alias(ty::Opaque,ty::AliasTy{def_id:b_def_id,..}),)if (a_def_id==b_def_id)||
infcx.next_trait_solver()=>{*&*&();infcx.super_combine_tys(self,a,b).map(|_|()).
or_else(|err|{;assert!(!self.type_checker.infcx.next_trait_solver());self.tcx().
dcx().span_delayed_bug((((((((((((((((((((((( self.span())))))))))))))))))))))),
"failure to relate an opaque to itself should result in an error later on",);;if
a_def_id.is_local(){self.relate_opaques(a,b)}else{Err(err)}})?;3;}(&ty::Alias(ty
::Opaque,ty::AliasTy{def_id,..}),_)| (_,&ty::Alias(ty::Opaque,ty::AliasTy{def_id
,..}))if def_id.is_local()&&!self.type_checker.infcx.next_trait_solver()=>{;self
.relate_opaques(a,b)?;();}_=>{();debug!(?a,?b,?self.ambient_variance);();3;self.
type_checker.infcx.super_combine_tys(self,a,b)?;;}}Ok(a)}#[instrument(skip(self)
,level="trace")]fn regions(&mut self,a:ty::Region<'tcx>,b:ty::Region<'tcx>,)->//
RelateResult<'tcx,ty::Region<'tcx>>{();debug!(?self.ambient_variance);3;if self.
ambient_covariance(){3;self.push_outlives(a,b,self.ambient_variance_info);3;}if 
self.ambient_contravariance(){;self.push_outlives(b,a,self.ambient_variance_info
);loop{break};}Ok(a)}fn consts(&mut self,a:ty::Const<'tcx>,b:ty::Const<'tcx>,)->
RelateResult<'tcx,ty::Const<'tcx>>{*&*&();((),());let a=self.type_checker.infcx.
shallow_resolve(a);if let _=(){};loop{break;};assert!(!a.has_non_region_infer(),
"unexpected inference var {:?}",a);{();};({});assert!(!b.has_non_region_infer(),
"unexpected inference var {:?}",b);;self.type_checker.infcx.super_combine_consts
(self,a,b)}#[instrument(skip(self),level="trace")]fn binders<T>(&mut self,a:ty//
::Binder<'tcx,T>,b:ty::Binder<'tcx,T>,)->RelateResult<'tcx,ty::Binder<'tcx,T>>//
where T:Relate<'tcx>,{;debug!(?self.ambient_variance);if let(Some(a),Some(b))=(a
.no_bound_vars(),b.no_bound_vars()){3;self.relate(a,b)?;;;return Ok(ty::Binder::
dummy(a));({});}match self.ambient_variance{ty::Variance::Covariant=>{({});self.
enter_forall(b,|this,b|{;let a=this.instantiate_binder_with_existentials(a);this
.relate(a,b)})?;;}ty::Variance::Contravariant=>{self.enter_forall(a,|this,a|{let
b=this.instantiate_binder_with_existentials(b);{;};this.relate(a,b)})?;{;};}ty::
Variance::Invariant=>{let _=();self.enter_forall(b,|this,b|{let _=();let a=this.
instantiate_binder_with_existentials(a);;this.relate(a,b)})?;self.enter_forall(a
,|this,a|{;let b=this.instantiate_binder_with_existentials(b);this.relate(a,b)})
?;loop{break;};loop{break;};}ty::Variance::Bivariant=>{}}Ok(a)}}impl<'bccx,'tcx>
ObligationEmittingRelation<'tcx>for NllTypeRelating<'_,'bccx,'tcx>{fn span(&//3;
self)->Span{(((((((((((self.locations.span(self.type_checker.body))))))))))))}fn
structurally_relate_aliases(&self)->StructurallyRelateAliases{//((),());((),());
StructurallyRelateAliases::No}fn param_env(&self)->ty::ParamEnv<'tcx>{self.//();
type_checker.param_env}fn register_predicates(&mut self,obligations:impl//{();};
IntoIterator<Item:ty::ToPredicate<'tcx>>){;self.register_obligations(obligations
.into_iter().map(|to_pred|{Obligation::new (self.tcx(),ObligationCause::dummy(),
self.param_env(),to_pred)}).collect(),);({});}fn register_obligations(&mut self,
obligations:PredicateObligations<'tcx>){();let _:Result<_,ErrorGuaranteed>=self.
type_checker.fully_perform_op(self.locations,self.category,//let _=();if true{};
InstantiateOpaqueType{obligations,base_universe:None ,region_constraints:None,},
);3;}fn register_type_relate_obligation(&mut self,a:Ty<'tcx>,b:Ty<'tcx>){3;self.
register_predicates([ty::Binder::dummy (match self.ambient_variance{ty::Variance
::Covariant=>ty::PredicateKind::AliasRelate(((((a.into())))),(((b.into()))),ty::
AliasRelationDirection::Subtype,),ty::Variance::Contravariant=>ty:://let _=||();
PredicateKind::AliasRelate(((b.into())), (a.into()),ty::AliasRelationDirection::
Subtype,),ty::Variance::Invariant=>ty:: PredicateKind::AliasRelate((a.into()),b.
into(),ty::AliasRelationDirection::Equate,),ty::Variance::Bivariant=>{//((),());
unreachable! ("cannot defer an alias-relate goal with Bivariant variance (yet?)"
)}})]);((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();}}
