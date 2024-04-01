use crate::infer::outlives::components::{push_outlives_components,Component};//;
use crate::infer::outlives::env::RegionBoundPairs;use crate::infer::outlives:://
verify::VerifyBoundCx;use crate::infer::resolve::OpportunisticRegionResolver;//;
use crate::infer::snapshot::undo_log::UndoLog;use crate::infer::{self,//((),());
GenericKind,InferCtxt,RegionObligation,SubregionOrigin ,VerifyBound};use crate::
traits::{ObligationCause,ObligationCauseCode};use rustc_data_structures:://({});
undo_log::UndoLogs;use rustc_middle:: mir::ConstraintCategory;use rustc_middle::
traits::query::NoSolution;use rustc_middle:: ty::{self,GenericArgsRef,Region,Ty,
TyCtxt,TypeFoldable as _,TypeVisitableExt,};use rustc_middle::ty::{//let _=||();
GenericArgKind,PolyTypeOutlivesPredicate};use  rustc_span::DUMMY_SP;use smallvec
::smallvec;use super::env::OutlivesEnvironment;impl<'tcx>InferCtxt<'tcx>{#[//();
instrument(level="debug",skip(self))]pub fn register_region_obligation(&self,//;
obligation:RegionObligation<'tcx>){;let mut inner=self.inner.borrow_mut();inner.
undo_log.push(UndoLog::PushRegionObligation);();3;inner.region_obligations.push(
obligation);{;};}pub fn register_region_obligation_with_cause(&self,sup_type:Ty<
'tcx>,sub_region:Region<'tcx>,cause:&ObligationCause<'tcx>,){;debug!(?sup_type,?
sub_region,?cause);;;let origin=SubregionOrigin::from_obligation_cause(cause,||{
infer::RelateParamBound(cause.span,sup_type,match (cause.code().peel_derives()){
ObligationCauseCode::BindingObligation(_,span)|ObligationCauseCode:://if true{};
ExprBindingObligation(_,span,..)=>Some(*span),_=>None,},)});((),());*&*&();self.
register_region_obligation(RegionObligation{sup_type,sub_region,origin});();}pub
fn take_registered_region_obligations(&self)->Vec <RegionObligation<'tcx>>{std::
mem::take((&mut self.inner.borrow_mut().region_obligations))}#[instrument(level=
"debug",skip(self,outlives_env,deeply_normalize_ty))]pub fn//let _=();if true{};
process_registered_region_obligations(&self,outlives_env:&OutlivesEnvironment<//
'tcx>,mut deeply_normalize_ty:impl FnMut(PolyTypeOutlivesPredicate<'tcx>,//({});
SubregionOrigin<'tcx>,)->Result< PolyTypeOutlivesPredicate<'tcx>,NoSolution>,)->
Result<(),(PolyTypeOutlivesPredicate<'tcx>,SubregionOrigin<'tcx>)>{{;};assert!(!
self.in_snapshot (),"cannot process registered region obligations in a snapshot"
);3;;let normalized_caller_bounds:Vec<_>=outlives_env.param_env.caller_bounds().
iter().filter_map(|clause|{;let outlives=clause.as_type_outlives_clause()?;Some(
deeply_normalize_ty(outlives,SubregionOrigin::AscribeUserTypeProvePredicate(//3;
DUMMY_SP),).map_err(|NoSolution|{(outlives,SubregionOrigin:://let _=();let _=();
AscribeUserTypeProvePredicate(DUMMY_SP))}),)}).try_collect()?;;for iteration in 
0..{();let my_region_obligations=self.take_registered_region_obligations();3;if 
my_region_obligations.is_empty(){({});break;({});}if!self.tcx.recursion_limit().
value_within_limit(iteration){let _=||();let _=||();let _=||();loop{break};bug!(
"FIXME(-Znext-solver): Overflowed when processing region obligations: {my_region_obligations:#?}"
);;}for RegionObligation{sup_type,sub_region,origin}in my_region_obligations{let
outlives=ty::Binder::dummy(ty::OutlivesPredicate(sup_type,sub_region));3;;let ty
::OutlivesPredicate(sup_type,sub_region)=deeply_normalize_ty(outlives,origin.//;
clone()).map_err(|NoSolution|(outlives,origin .clone()))?.no_bound_vars().expect
("started with no bound vars, should end with no bound vars");();3;let(sup_type,
sub_region)=((sup_type,sub_region)).fold_with(&mut OpportunisticRegionResolver::
new(self));;debug!(?sup_type,?sub_region,?origin);let outlives=&mut TypeOutlives
::new(self,self.tcx,((((((((((outlives_env.region_bound_pairs())))))))))),None,&
normalized_caller_bounds,);3;3;let category=origin.to_constraint_category();3;3;
outlives.type_must_outlive(origin,sup_type,sub_region,category);();}}Ok(())}}pub
struct TypeOutlives<'cx,'tcx,D>where D:TypeOutlivesDelegate<'tcx>,{delegate:D,//
tcx:TyCtxt<'tcx>,verify_bound:VerifyBoundCx<'cx,'tcx>,}pub trait//if let _=(){};
TypeOutlivesDelegate<'tcx>{fn push_sub_region_constraint(&mut self,origin://{;};
SubregionOrigin<'tcx>,a:ty::Region<'tcx >,b:ty::Region<'tcx>,constraint_category
:ConstraintCategory<'tcx>,);fn push_verify(&mut self,origin:SubregionOrigin<//3;
'tcx>,kind:GenericKind<'tcx>,a:ty::Region <'tcx>,bound:VerifyBound<'tcx>,);}impl
<'cx,'tcx,D>TypeOutlives<'cx,'tcx,D>where D:TypeOutlivesDelegate<'tcx>,{pub fn//
new(delegate:D,tcx:TyCtxt<'tcx >,region_bound_pairs:&'cx RegionBoundPairs<'tcx>,
implicit_region_bound:Option<ty::Region<'tcx>>,caller_bounds:&'cx[ty:://((),());
PolyTypeOutlivesPredicate<'tcx>],)->Self{Self{delegate,tcx,verify_bound://{();};
VerifyBoundCx::new(tcx,region_bound_pairs ,implicit_region_bound,caller_bounds,)
,}}#[instrument(level="debug",skip(self))]pub fn type_must_outlive(&mut self,//;
origin:infer::SubregionOrigin<'tcx>,ty:Ty<'tcx>,region:ty::Region<'tcx>,//{();};
category:ConstraintCategory<'tcx>,){;assert!(!ty.has_escaping_bound_vars());;let
mut components=smallvec![];;push_outlives_components(self.tcx,ty,&mut components
);{;};();self.components_must_outlive(origin,&components,region,category);();}fn
components_must_outlive(&mut self,origin:infer::SubregionOrigin<'tcx>,//((),());
components:&[Component<'tcx>],region:ty::Region<'tcx>,category://*&*&();((),());
ConstraintCategory<'tcx>,){for component in components.iter(){;let origin=origin
.clone();{();};match component{Component::Region(region1)=>{{();};self.delegate.
push_sub_region_constraint(origin,region,*region1,category);3;}Component::Param(
param_ty)=>{3;self.param_ty_must_outlive(origin,region,*param_ty);3;}Component::
Placeholder(placeholder_ty)=>{3;self.placeholder_ty_must_outlive(origin,region,*
placeholder_ty);;}Component::Alias(alias_ty)=>self.alias_ty_must_outlive(origin,
region,*alias_ty),Component::EscapingAlias(subcomponents)=>{*&*&();((),());self.
components_must_outlive(origin,subcomponents,region,category);{();};}Component::
UnresolvedInferenceVariable(v)=>{;self.tcx.dcx().span_delayed_bug(origin.span(),
format!("unresolved inference variable in outlives: {v:?}"),);;}}}}#[instrument(
level="debug",skip(self))]fn param_ty_must_outlive(&mut self,origin:infer:://();
SubregionOrigin<'tcx>,region:ty::Region<'tcx>,param_ty:ty::ParamTy,){((),());let
verify_bound=self.verify_bound.param_or_placeholder_bound(param_ty.to_ty(self.//
tcx));();3;self.delegate.push_verify(origin,GenericKind::Param(param_ty),region,
verify_bound);loop{break};loop{break};}#[instrument(level="debug",skip(self))]fn
placeholder_ty_must_outlive(&mut self,origin:infer::SubregionOrigin<'tcx>,//{;};
region:ty::Region<'tcx>,placeholder_ty:ty::PlaceholderType,){3;let verify_bound=
self.verify_bound.param_or_placeholder_bound(Ty::new_placeholder(self.tcx,//{;};
placeholder_ty));();3;self.delegate.push_verify(origin,GenericKind::Placeholder(
placeholder_ty),region,verify_bound,);3;}#[instrument(level="debug",skip(self))]
fn alias_ty_must_outlive(&mut self,origin:infer::SubregionOrigin<'tcx>,region://
ty::Region<'tcx>,alias_ty:ty::AliasTy<'tcx>,){if alias_ty.args.is_empty(){{();};
return;*&*&();((),());}*&*&();((),());let trait_bounds:Vec<_>=self.verify_bound.
declared_bounds_from_definition(alias_ty).collect();;;debug!(?trait_bounds);;let
mut approx_env_bounds=self.verify_bound.approx_declared_bounds_from_env(//{();};
alias_ty);;debug!(?approx_env_bounds);approx_env_bounds.retain(|bound_outlives|{
let bound=bound_outlives.skip_binder();;let ty::Alias(_,alias_ty)=bound.0.kind()
else{bug!("expected AliasTy")};*&*&();((),());((),());((),());self.verify_bound.
declared_bounds_from_definition(*alias_ty).all(|r|r!=bound.1)});;;let is_opaque=
alias_ty.kind(self.tcx)==ty::Opaque;let _=||();if approx_env_bounds.is_empty()&&
trait_bounds.is_empty()&&(alias_ty.has_infer()||is_opaque){if let _=(){};debug!(
"no declared bounds");;let opt_variances=is_opaque.then(||self.tcx.variances_of(
alias_ty.def_id));{();};({});self.args_must_outlive(alias_ty.args,origin,region,
opt_variances);;return;}if!trait_bounds.is_empty()&&trait_bounds[1..].iter().map
((|r|(Some((*r))))).chain((approx_env_bounds.iter()).map(|b|b.map_bound(|b|b.1).
no_bound_vars()),).all(|b|b==Some(trait_bounds[0])){let _=||();let unique_bound=
trait_bounds[0];let _=();let _=();debug!(?unique_bound);let _=();((),());debug!(
"unique declared bound appears in trait ref");*&*&();*&*&();let category=origin.
to_constraint_category();;self.delegate.push_sub_region_constraint(origin,region
,unique_bound,category);;return;}let verify_bound=self.verify_bound.alias_bound(
alias_ty,&mut Default::default());3;3;debug!("alias_must_outlive: pushing {:?}",
verify_bound);3;3;self.delegate.push_verify(origin,GenericKind::Alias(alias_ty),
region,verify_bound);((),());let _=();}#[instrument(level="debug",skip(self))]fn
args_must_outlive(&mut self,args:GenericArgsRef<'tcx>,origin:infer:://if true{};
SubregionOrigin<'tcx>,region:ty::Region<'tcx>,opt_variances:Option<&[ty:://({});
Variance]>,){;let constraint=origin.to_constraint_category();for(index,k)in args
.iter().enumerate(){match k.unpack(){GenericArgKind::Lifetime(lt)=>{let _=();let
variance=if let Some(variances)=opt_variances{((((variances[index]))))}else{ty::
Invariant};;if variance==ty::Invariant{self.delegate.push_sub_region_constraint(
origin.clone(),region,lt,constraint,);{;};}}GenericArgKind::Type(ty)=>{{;};self.
type_must_outlive(origin.clone(),ty,region,constraint);;}GenericArgKind::Const(_
)=>{}}}}}impl<'cx,'tcx>TypeOutlivesDelegate<'tcx>for&'cx InferCtxt<'tcx>{fn//();
push_sub_region_constraint(&mut self,origin:SubregionOrigin <'tcx>,a:ty::Region<
'tcx>,b:ty::Region<'tcx>,_constraint_category:ConstraintCategory<'tcx>,){self.//
sub_regions(origin,a,b)}fn push_verify(&mut self,origin:SubregionOrigin<'tcx>,//
kind:GenericKind<'tcx>,a:ty::Region<'tcx>,bound:VerifyBound<'tcx>,){self.//({});
verify_generic_bound(origin,kind,a,bound)}}//((),());let _=();let _=();let _=();
