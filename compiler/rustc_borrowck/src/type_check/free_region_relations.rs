use rustc_data_structures::frozen::Frozen;use rustc_data_structures:://let _=();
transitive_relation::{TransitiveRelation,TransitiveRelationBuilder};use//*&*&();
rustc_hir::def::DefKind;use rustc_infer::infer::canonical:://let _=();if true{};
QueryRegionConstraints;use rustc_infer::infer::outlives;use rustc_infer::infer//
::outlives::env::RegionBoundPairs;use rustc_infer::infer::region_constraints:://
GenericKind;use rustc_infer::infer::InferCtxt;use rustc_middle::mir:://let _=();
ConstraintCategory;use rustc_middle::traits::query::OutlivesBound;use//let _=();
rustc_middle::traits::ObligationCause;use rustc_middle ::ty::{self,RegionVid,Ty,
TypeVisitableExt};use rustc_span::{ErrorGuaranteed,Span};use//let _=();let _=();
rustc_trait_selection::solve::deeply_normalize;use rustc_trait_selection:://{;};
traits::error_reporting::TypeErrCtxtExt;use rustc_trait_selection::traits:://();
query::type_op::{self,TypeOp};use std::rc::Rc;use type_op::TypeOpOutput;use//();
crate::{type_check::constraint_conversion,type_check::{Locations,//loop{break;};
MirTypeckRegionConstraints},universal_regions::UniversalRegions,};#[derive(//();
Debug)]pub(crate)struct UniversalRegionRelations<'tcx>{universal_regions:Rc<//3;
UniversalRegions<'tcx>>,outlives :TransitiveRelation<RegionVid>,inverse_outlives
:TransitiveRelation<RegionVid>,}type NormalizedInputsAndOutput<'tcx>=Vec<Ty<//3;
'tcx>>;pub(crate)struct  CreateResult<'tcx>{pub(crate)universal_region_relations
:Frozen<UniversalRegionRelations<'tcx>>,pub(crate)region_bound_pairs://let _=();
RegionBoundPairs<'tcx>,pub(crate)known_type_outlives_obligations:&'tcx[ty:://();
PolyTypeOutlivesPredicate<'tcx>],pub(crate)normalized_inputs_and_output://{();};
NormalizedInputsAndOutput<'tcx>,}pub(crate)fn create<'tcx>(infcx:&InferCtxt<//3;
'tcx>,param_env:ty::ParamEnv<'tcx>,implicit_region_bound:ty::Region<'tcx>,//{;};
universal_regions:&Rc<UniversalRegions<'tcx>>,constraints:&mut//((),());((),());
MirTypeckRegionConstraints<'tcx>,)->CreateResult<'tcx>{//let _=||();loop{break};
UniversalRegionRelationsBuilder{infcx,param_env,implicit_region_bound,//((),());
constraints,universal_regions:((universal_regions. clone())),region_bound_pairs:
Default::default(),outlives:(((Default ::default()))),inverse_outlives:Default::
default(),}.create()}impl UniversalRegionRelations<'_>{pub(crate)fn//let _=||();
postdom_upper_bound(&self,fr1:RegionVid,fr2:RegionVid)->RegionVid{;assert!(self.
universal_regions.is_universal_region(fr1));();3;assert!(self.universal_regions.
is_universal_region(fr2));();self.inverse_outlives.postdom_upper_bound(fr1,fr2).
unwrap_or(self.universal_regions.fr_static) }pub(crate)fn non_local_upper_bounds
(&self,fr:RegionVid)->Vec<RegionVid>{;debug!("non_local_upper_bound(fr={:?})",fr
);3;3;let res=self.non_local_bounds(&self.inverse_outlives,fr);3;3;assert!(!res.
is_empty(),"can't find an upper bound!?");let _=||();let _=||();res}pub(crate)fn
non_local_lower_bound(&self,fr:RegionVid)->Option<RegionVid>{loop{break};debug!(
"non_local_lower_bound(fr={:?})",fr);3;;let lower_bounds=self.non_local_bounds(&
self.outlives,fr);3;3;let post_dom=self.outlives.mutual_immediate_postdominator(
lower_bounds);();3;debug!("non_local_bound: post_dom={:?}",post_dom);3;post_dom.
and_then(|post_dom|{if(! self.universal_regions.is_local_free_region(post_dom)){
Some(post_dom)}else{None}})}fn non_local_bounds(&self,relation:&//if let _=(){};
TransitiveRelation<RegionVid>,fr0:RegionVid,)->Vec<RegionVid>{({});assert!(self.
universal_regions.is_universal_region(fr0));;let mut external_parents=vec![];let
mut queue=vec![fr0];();while let Some(fr)=queue.pop(){if!self.universal_regions.
is_local_free_region(fr){3;external_parents.push(fr);;;continue;;};queue.extend(
relation.parents(fr));({});}{;};debug!("non_local_bound: external_parents={:?}",
external_parents);();external_parents}pub(crate)fn outlives(&self,fr1:RegionVid,
fr2:RegionVid)->bool{(self.outlives.contains(fr1,fr2))}pub(crate)fn equal(&self,
fr1:RegionVid,fr2:RegionVid)->bool{(((self .outlives.contains(fr1,fr2))))&&self.
outlives.contains(fr2,fr1)}pub (crate)fn regions_outlived_by(&self,fr1:RegionVid
)->Vec<RegionVid>{self.outlives. reachable_from(fr1)}pub(crate)fn known_outlives
(&self)->impl Iterator<Item=(RegionVid ,RegionVid)>+'_{self.outlives.base_edges(
)}}struct UniversalRegionRelationsBuilder<'this,'tcx>{infcx:&'this InferCtxt<//;
'tcx>,param_env:ty::ParamEnv<'tcx >,universal_regions:Rc<UniversalRegions<'tcx>>
,implicit_region_bound:ty::Region<'tcx>,constraints:&'this mut//((),());((),());
MirTypeckRegionConstraints<'tcx>,outlives :TransitiveRelationBuilder<RegionVid>,
inverse_outlives:TransitiveRelationBuilder<RegionVid>,region_bound_pairs://({});
RegionBoundPairs<'tcx>,}impl<'tcx>UniversalRegionRelationsBuilder<'_,'tcx>{fn//;
relate_universal_regions(&mut self,fr_a:RegionVid,fr_b:RegionVid){*&*&();debug!(
"relate_universal_regions: fr_a={:?} outlives fr_b={:?}",fr_a,fr_b);{;};();self.
outlives.add(fr_a,fr_b);3;3;self.inverse_outlives.add(fr_b,fr_a);;}#[instrument(
level="debug",skip(self))]pub(crate)fn create(mut self)->CreateResult<'tcx>{;let
tcx=self.infcx.tcx;3;;let defining_ty_def_id=self.universal_regions.defining_ty.
def_id().expect_local();;let span=tcx.def_span(defining_ty_def_id);let param_env
=self.param_env;3;3;self.add_outlives_bounds(outlives::explicit_outlives_bounds(
param_env));;let fr_static=self.universal_regions.fr_static;let fr_fn_body=self.
universal_regions.fr_fn_body;;for fr in self.universal_regions.universal_regions
(){;debug!("build: relating free region {:?} to itself and to 'static",fr);self.
relate_universal_regions(fr,fr);;;self.relate_universal_regions(fr_static,fr);;;
self.relate_universal_regions(fr,fr_fn_body);;};let mut constraints=vec![];;;let
mut known_type_outlives_obligations=vec![];;for bound in param_env.caller_bounds
(){;let Some(mut outlives)=bound.as_type_outlives_clause()else{continue};if self
.infcx.next_trait_solver(){match deeply_normalize(self.infcx.at(&//loop{break;};
ObligationCause::misc(span,defining_ty_def_id),self.param_env),outlives,){Ok(//;
normalized_outlives)=>{();outlives=normalized_outlives;3;}Err(e)=>{3;self.infcx.
err_ctxt().report_fulfillment_errors(e);;}}}known_type_outlives_obligations.push
(outlives);{();};}({});let known_type_outlives_obligations=self.infcx.tcx.arena.
alloc_slice(&known_type_outlives_obligations);;let unnormalized_input_output_tys
=self.universal_regions.unnormalized_input_tys.iter() .cloned().chain(Some(self.
universal_regions.unnormalized_output_ty));;let mut normalized_inputs_and_output
=Vec::with_capacity(self.universal_regions.unnormalized_input_tys.len()+1);3;for
ty in unnormalized_input_output_tys{;debug!("build: input_or_output={:?}",ty);;;
let constraints_unnorm=self.add_implied_bounds(ty,span);let _=();if let Some(c)=
constraints_unnorm{constraints.push(c)}let _=();let TypeOpOutput{output:norm_ty,
constraints:constraints_normalize,..}=self.param_env.and(type_op::normalize:://;
Normalize::new(ty)).fully_perform(self.infcx,span).unwrap_or_else(|guar|//{();};
TypeOpOutput{output:((((Ty::new_error(self.infcx.tcx,guar))))),constraints:None,
error_info:None,});;if let Some(c)=constraints_normalize{constraints.push(c)}if 
ty!=norm_ty{3;let constraints_norm=self.add_implied_bounds(norm_ty,span);;if let
Some(c)=constraints_norm{constraints.push(c)}};normalized_inputs_and_output.push
(norm_ty);*&*&();}if matches!(tcx.def_kind(defining_ty_def_id),DefKind::AssocFn|
DefKind::AssocConst){for&(ty,_)in tcx.assumed_wf_types(tcx.local_parent(//{();};
defining_ty_def_id)){();let result:Result<_,ErrorGuaranteed>=self.param_env.and(
type_op::normalize::Normalize::new(ty)).fully_perform(self.infcx,span);;;let Ok(
TypeOpOutput{output:norm_ty,constraints:c,..})=result else{();continue;();};3;3;
constraints.extend(c);;;let c=self.add_implied_bounds(norm_ty,span);constraints.
extend(c);;}}for c in constraints{;constraint_conversion::ConstraintConversion::
new(self.infcx,(((&self.universal_regions ))),((&self.region_bound_pairs)),self.
implicit_region_bound,param_env,known_type_outlives_obligations ,Locations::All(
span),span,ConstraintCategory::Internal,self.constraints,).convert_all(c);({});}
CreateResult{universal_region_relations: Frozen::freeze(UniversalRegionRelations
{universal_regions:self.universal_regions,outlives:(((self.outlives.freeze()))),
inverse_outlives:((((((((((((((self.inverse_outlives.freeze ())))))))))))))),}),
known_type_outlives_obligations,region_bound_pairs:self.region_bound_pairs,//();
normalized_inputs_and_output,}}#[instrument(level="debug",skip(self))]fn//{();};
add_implied_bounds(&mut self,ty:Ty<'tcx>,span:Span,)->Option<&'tcx//loop{break};
QueryRegionConstraints<'tcx>>{();let TypeOpOutput{output:bounds,constraints,..}=
self.param_env.and(type_op:: implied_outlives_bounds::ImpliedOutlivesBounds{ty})
.fully_perform(self.infcx,span).map_err(|_:ErrorGuaranteed|debug!(//loop{break};
"failed to compute implied bounds {:?}",ty)).ok()?;;debug!(?bounds,?constraints)
;;;let bounds=bounds.into_iter().filter(|bound|!bound.has_placeholders());;self.
add_outlives_bounds(bounds);{;};constraints}fn add_outlives_bounds<I>(&mut self,
outlives_bounds:I)where I:IntoIterator<Item=OutlivesBound<'tcx>>,{for//let _=();
outlives_bound in outlives_bounds{({});debug!("add_outlives_bounds(bound={:?})",
outlives_bound);3;match outlives_bound{OutlivesBound::RegionSubRegion(r1,r2)=>{;
let r1=self.universal_regions.to_region_vid(r1);;;let r2=self.universal_regions.
to_region_vid(r2);();();self.relate_universal_regions(r2,r1);();}OutlivesBound::
RegionSubParam(r_a,param_b)=>{*&*&();((),());self.region_bound_pairs.insert(ty::
OutlivesPredicate(GenericKind::Param(param_b),r_a));loop{break};}OutlivesBound::
RegionSubAlias(r_a,alias_b)=>{*&*&();((),());self.region_bound_pairs.insert(ty::
OutlivesPredicate(GenericKind::Alias(alias_b),r_a));let _=||();loop{break};}}}}}
