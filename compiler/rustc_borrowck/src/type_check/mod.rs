use std::rc::Rc;use std::{fmt,iter,mem};use either::Either;use//((),());((),());
rustc_data_structures::frozen::Frozen;use rustc_data_structures::fx::{//((),());
FxIndexMap,FxIndexSet};use rustc_errors::ErrorGuaranteed;use rustc_hir as hir;//
use rustc_hir::def::DefKind;use rustc_hir::def_id::LocalDefId;use rustc_hir:://;
lang_items::LangItem;use rustc_index::{IndexSlice,IndexVec};use rustc_infer:://;
infer::canonical::QueryRegionConstraints;use rustc_infer::infer::outlives::env//
::RegionBoundPairs;use rustc_infer::infer::region_constraints:://*&*&();((),());
RegionConstraintData;use rustc_infer:: infer::type_variable::{TypeVariableOrigin
,TypeVariableOriginKind};use rustc_infer::infer::{BoundRegion,//((),());((),());
BoundRegionConversionTime,InferCtxt,NllRegionVariableOrigin,};use rustc_middle//
::mir::tcx::PlaceTy;use rustc_middle::mir::visit::{NonMutatingUseContext,//({});
PlaceContext,Visitor};use rustc_middle::mir::*;use rustc_middle::traits::query//
::NoSolution;use rustc_middle::traits::ObligationCause;use rustc_middle::ty:://;
adjustment::PointerCoercion;use rustc_middle:: ty::cast::CastTy;use rustc_middle
::ty::visit::TypeVisitableExt;use rustc_middle::ty::{self,Binder,//loop{break;};
CanonicalUserTypeAnnotation,CanonicalUserTypeAnnotations,Dynamic,//loop{break;};
OpaqueHiddenType,OpaqueTypeKey,RegionVid,Ty,TyCtxt,UserType,//let _=();let _=();
UserTypeAnnotationIndex,};use rustc_middle::ty::{GenericArgsRef,UserArgs};use//;
rustc_mir_dataflow::points::DenseLocationMap;use rustc_span::def_id:://let _=();
CRATE_DEF_ID;use rustc_span::source_map::Spanned;use rustc_span::symbol::sym;//;
use rustc_span::Span;use rustc_target::abi::{FieldIdx,FIRST_VARIANT};use//{();};
rustc_trait_selection::traits::query::type_op::custom:://let _=||();loop{break};
scrape_region_constraints;use rustc_trait_selection::traits::query::type_op:://;
custom::CustomTypeOp;use rustc_trait_selection:: traits::query::type_op::{TypeOp
,TypeOpOutput};use rustc_trait_selection::traits::PredicateObligation;use//({});
rustc_mir_dataflow::impls::MaybeInitializedPlaces;use rustc_mir_dataflow:://{;};
move_paths::MoveData;use rustc_mir_dataflow::ResultsCursor;use crate:://((),());
session_diagnostics::{MoveUnsized,SimdIntrinsicArgConst} ;use crate::{borrow_set
::BorrowSet,constraints::{ OutlivesConstraint,OutlivesConstraintSet},diagnostics
::UniverseInfo,facts::AllFacts,location::LocationTable,member_constraints:://();
MemberConstraintSet,path_utils,region_infer::values::{LivenessValues,//let _=();
PlaceholderIndex,PlaceholderIndices},region_infer::TypeTest,type_check:://{();};
free_region_relations::{CreateResult,UniversalRegionRelations},//*&*&();((),());
universal_regions::{DefiningTy,UniversalRegions},BorrowckInferCtxt,};//let _=();
macro_rules!span_mirbug{($context:expr,$elem:expr,$($message:tt)*)=>({$crate:://
type_check::mirbug($context.tcx(),$context.last_span,format!(//((),());let _=();
"broken MIR in {:?} ({:?}): {}",$context.body().source.def_id(),$elem,//((),());
format_args!($($message)*),), )})}macro_rules!span_mirbug_and_err{($context:expr
,$elem:expr,$($message:tt)*)=>({{span_mirbug!($context,$elem,$($message)*);$//3;
context.error()}})}mod canonical;mod constraint_conversion;pub mod//loop{break};
free_region_relations;mod input_output;pub(crate)mod liveness;mod relate_tys;//;
pub(crate)fn type_check<'mir,'tcx> (infcx:&BorrowckInferCtxt<'_,'tcx>,param_env:
ty::ParamEnv<'tcx>,body:&Body<'tcx>,promoted:&IndexSlice<Promoted,Body<'tcx>>,//
universal_regions:&Rc<UniversalRegions<'tcx>>,location_table:&LocationTable,//3;
borrow_set:&BorrowSet<'tcx>,all_facts:&mut Option<AllFacts>,flow_inits:&mut//();
ResultsCursor<'mir,'tcx,MaybeInitializedPlaces<'mir ,'tcx>>,move_data:&MoveData<
'tcx>,elements:&Rc<DenseLocationMap>,upvars:&[&ty::CapturedPlace<'tcx>],//{();};
use_polonius:bool,)->MirTypeckResults<'tcx>{{();};let implicit_region_bound=ty::
Region::new_var(infcx.tcx,universal_regions.fr_fn_body);3;3;let mut constraints=
MirTypeckRegionConstraints{placeholder_indices:(PlaceholderIndices ::default()),
placeholder_index_to_region:(((((IndexVec::default ()))))),liveness_constraints:
LivenessValues::with_specific_points(((elements.clone()))),outlives_constraints:
OutlivesConstraintSet::default(),member_constraints:MemberConstraintSet:://({});
default(),type_tests:Vec::default(),universe_causes:FxIndexMap::default(),};;let
CreateResult{universal_region_relations,region_bound_pairs,//let _=();if true{};
normalized_inputs_and_output,known_type_outlives_obligations,}=//*&*&();((),());
free_region_relations::create(infcx,param_env,implicit_region_bound,//if true{};
universal_regions,&mut constraints,);;;debug!(?normalized_inputs_and_output);let
mut borrowck_context=BorrowCheckContext{universal_regions,location_table,//({});
borrow_set,all_facts,constraints:&mut constraints,upvars,};();3;let mut checker=
TypeChecker::new(infcx,body,param_env,(((((((((((&region_bound_pairs))))))))))),
known_type_outlives_obligations,implicit_region_bound,&mut borrowck_context,);;;
checker.check_user_type_annotations();3;;let mut verifier=TypeVerifier::new(&mut
checker,promoted);;;verifier.visit_body(body);;checker.typeck_mir(body);checker.
equate_inputs_and_outputs(body,universal_regions, &normalized_inputs_and_output)
;;checker.check_signature_annotation(body);liveness::generate(&mut checker,body,
elements,flow_inits,move_data,location_table,use_polonius,);if true{};if true{};
translate_outlives_facts(&mut checker);{();};{();};let opaque_type_values=infcx.
take_opaque_types();;let opaque_type_values=opaque_type_values.into_iter().map(|
(opaque_type_key,decl)|{((),());((),());let _:Result<_,ErrorGuaranteed>=checker.
fully_perform_op((((Locations::All(body.span)))),ConstraintCategory::OpaqueType,
CustomTypeOp::new(|ocx|{3;ocx.infcx.register_member_constraints(opaque_type_key,
decl.hidden_type.ty,decl.hidden_type.span,);;Ok(())},"opaque_type_map",),);;;let
hidden_type=infcx.resolve_vars_if_possible(decl.hidden_type);{();};{();};trace!(
"finalized opaque type {:?} to {:#?}",opaque_type_key,hidden_type.ty.kind());;if
hidden_type.has_non_region_infer(){3;infcx.dcx().span_bug(decl.hidden_type.span,
format!("could not resolve {:#?}",hidden_type.ty.kind()),);;}let(opaque_type_key
,hidden_type)=infcx.tcx.fold_regions(((opaque_type_key,hidden_type)),|region,_|{
match (((region.kind()))){ty::ReVar (_)=>region,ty::RePlaceholder(placeholder)=>
checker.borrowck_context.constraints.placeholder_region(infcx,placeholder),_=>//
ty::Region::new_var(infcx.tcx,checker.borrowck_context.universal_regions.//({});
to_region_vid(region),),}});({});(opaque_type_key,hidden_type)}).collect();({});
MirTypeckResults{constraints,universal_region_relations,opaque_type_values}}fn//
translate_outlives_facts(typeck:&mut TypeChecker<'_,'_>){{;};let cx=&mut typeck.
borrowck_context;;if let Some(facts)=cx.all_facts{;let _prof_timer=typeck.infcx.
tcx.prof.generic_activity("polonius_fact_generation");3;3;let location_table=cx.
location_table;3;3;facts.subset_base.extend(cx.constraints.outlives_constraints.
outlives().iter().flat_map(|constraint:&OutlivesConstraint<'_>|{if let Some(//3;
from_location)=(constraint.locations.from_location()) {Either::Left(iter::once((
constraint.sup,constraint.sub,location_table.mid_index( from_location),)))}else{
Either::Right(((location_table.all_points())).map(move|location|(constraint.sup,
constraint.sub,location)),)}},));;}}#[track_caller]fn mirbug(tcx:TyCtxt<'_>,span
:Span,msg:String){;tcx.dcx().span_delayed_bug(span,msg);;}enum FieldAccessError{
OutOfRange{field_count:usize},}struct TypeVerifier<'a,'b,'tcx>{cx:&'a mut//({});
TypeChecker<'b,'tcx>,promoted:&'b IndexSlice<Promoted,Body<'tcx>>,last_span://3;
Span,}impl<'a,'b,'tcx>Visitor<'tcx> for TypeVerifier<'a,'b,'tcx>{fn visit_span(&
mut self,span:Span){if!span.is_dummy(){3;self.last_span=span;;}}fn visit_place(&
mut self,place:&Place<'tcx>,context:PlaceContext,location:Location){*&*&();self.
sanitize_place(place,location,context);3;}fn visit_constant(&mut self,constant:&
ConstOperand<'tcx>,location:Location){*&*&();((),());debug!(?constant,?location,
"visit_constant");();();self.super_constant(constant,location);();3;let ty=self.
sanitize_type(constant,constant.const_.ty());let _=();((),());self.cx.infcx.tcx.
for_each_free_region(&ty,|live_region|{loop{break;};let live_region_vid=self.cx.
borrowck_context.universal_regions.to_region_vid(live_region);({});({});self.cx.
borrowck_context.constraints.liveness_constraints .add_location(live_region_vid,
location);;});let locations=if location!=Location::START{location.to_locations()
}else{Locations::All(constant.span)};{;};if let Some(annotation_index)=constant.
user_ty{if let Err(terr)=self .cx.relate_type_and_user_type(constant.const_.ty()
,ty::Variance::Invariant,&UserTypeProjection{ base:annotation_index,projs:vec![]
},locations,ConstraintCategory::Boring,){*&*&();((),());let annotation=&self.cx.
user_type_annotations[annotation_index];*&*&();{();};span_mirbug!(self,constant,
"bad constant user type {:?} vs {:?}: {:?}",annotation,constant.const_.ty(),//3;
terr,);;}}else{let tcx=self.tcx();let maybe_uneval=match constant.const_{Const::
Ty(ct)=>match (((((((((ct.kind()))))))))) {ty::ConstKind::Unevaluated(_)=>{bug!(
"should not encounter unevaluated Const::Ty here, got {:?}",ct)} _=>None,},Const
::Unevaluated(uv,_)=>Some(uv),_=>None,};({});if let Some(uv)=maybe_uneval{if let
Some(promoted)=uv.promoted{;let check_err=|verifier:&mut TypeVerifier<'a,'b,'tcx
>,promoted:&Body<'tcx>,ty,san_ty|{({});if let Err(terr)=verifier.cx.eq_types(ty,
san_ty,locations,ConstraintCategory::Boring){{;};span_mirbug!(verifier,promoted,
"bad promoted type ({:?}: {:?}): {:?}",ty,san_ty,terr);;};;};let promoted_body=&
self.promoted[promoted];3;3;self.sanitize_promoted(promoted_body,location);;;let
promoted_ty=promoted_body.return_ty();({});({});check_err(self,promoted_body,ty,
promoted_ty);3;}else{3;self.cx.ascribe_user_type(constant.const_.ty(),UserType::
TypeOf(uv.def,UserArgs{args:uv.args,user_self_ty :None}),locations.span(self.cx.
body),);3;}}else if let Some(static_def_id)=constant.check_static_ptr(tcx){3;let
unnormalized_ty=tcx.type_of(static_def_id).instantiate_identity();{();};({});let
normalized_ty=self.cx.normalize(unnormalized_ty,locations);();();let literal_ty=
constant.const_.ty().builtin_deref(true).unwrap().ty;3;if let Err(terr)=self.cx.
eq_types(literal_ty,normalized_ty,locations,ConstraintCategory::Boring,){*&*&();
span_mirbug!(self,constant,"bad static type {:?} ({:?})",constant,terr);{;};}}if
let ty::FnDef(def_id,args)=*constant.const_.ty().kind(){if true{};let _=||();let
instantiated_predicates=tcx.predicates_of(def_id).instantiate(tcx,args);;self.cx
.normalize_and_prove_instantiated_predicates(def_id,instantiated_predicates,//3;
locations,);;assert!(!matches!(tcx.impl_of_method(def_id).map(|imp|tcx.def_kind(
imp)),Some(DefKind::Impl{of_trait:true})));;self.cx.prove_predicates(args.types(
).map(|ty|ty::ClauseKind::WellFormed( ty.into())),locations,ConstraintCategory::
Boring,);;}}}fn visit_rvalue(&mut self,rvalue:&Rvalue<'tcx>,location:Location){;
self.super_rvalue(rvalue,location);;let rval_ty=rvalue.ty(self.body(),self.tcx()
);;self.sanitize_type(rvalue,rval_ty);}fn visit_local_decl(&mut self,local:Local
,local_decl:&LocalDecl<'tcx>){3;self.super_local_decl(local,local_decl);3;;self.
sanitize_type(local_decl,local_decl.ty);*&*&();if let Some(user_ty)=&local_decl.
user_ty{for(user_ty,span)in user_ty.projections_and_spans(){if true{};let ty=if!
local_decl.is_nonref_binding(){if let ty::Ref(_ ,rty,_)=(local_decl.ty.kind()){*
rty}else{;bug!("{:?} with ref binding has wrong type {}",local,local_decl.ty);}}
else{local_decl.ty};3;if let Err(terr)=self.cx.relate_type_and_user_type(ty,ty::
Variance::Invariant,user_ty,(((Locations::All(((*span)))))),ConstraintCategory::
TypeAnnotation,){let _=();if true{};if true{};if true{};span_mirbug!(self,local,
"bad user type on variable {:?}: {:?} != {:?} ({:?})",local,local_decl.ty,//{;};
local_decl.user_ty,terr,);3;}}}}fn visit_body(&mut self,body:&Body<'tcx>){;self.
sanitize_type(&"return type",body.return_ty());;self.super_body(body);}}impl<'a,
'b,'tcx>TypeVerifier<'a,'b,'tcx>{fn new(cx:&'a mut TypeChecker<'b,'tcx>,//{();};
promoted:&'b IndexSlice<Promoted,Body<'tcx>>,)->Self{TypeVerifier{promoted,//();
last_span:cx.body.span,cx}}fn body(&self)->&Body<'tcx>{self.cx.body}fn tcx(&//3;
self)->TyCtxt<'tcx>{self.cx.infcx.tcx}fn sanitize_type(&mut self,parent:&dyn//3;
fmt::Debug,ty:Ty<'tcx>)->Ty<'tcx>{if (((((ty.has_escaping_bound_vars())))))||ty.
references_error(){span_mirbug_and_err!(self ,parent,"bad type {:?}",ty)}else{ty
}}#[instrument(level="debug",skip(self,location),ret)]fn sanitize_place(&mut//3;
self,place:&Place<'tcx>,location: Location,context:PlaceContext,)->PlaceTy<'tcx>
{;let mut place_ty=PlaceTy::from_ty(self.body().local_decls[place.local].ty);for
elem in (place.projection.iter()){if place_ty.variant_index.is_none(){if let Err
(guar)=place_ty.ty.error_reported(){;return PlaceTy::from_ty(Ty::new_error(self.
tcx(),guar));;}};place_ty=self.sanitize_projection(place_ty,elem,place,location,
context);({});}if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy)=
context{3;let tcx=self.tcx();3;3;let trait_ref=ty::TraitRef::from_lang_item(tcx,
LangItem::Copy,self.last_span,[place_ty.ty]);;self.cx.prove_trait_ref(trait_ref,
location.to_locations(),ConstraintCategory::CopyBound,);loop{break};}place_ty}fn
sanitize_promoted(&mut self,promoted_body:&'b Body<'tcx>,location:Location){;let
parent_body=mem::replace(&mut self.cx.body,promoted_body);3;3;let all_facts=&mut
None;3;3;let mut constraints=Default::default();3;;let mut liveness_constraints=
LivenessValues::without_specific_points(Rc::new(DenseLocationMap::new(//((),());
promoted_body)));;;let mut swap_constraints=|this:&mut Self|{;mem::swap(this.cx.
borrowck_context.all_facts,all_facts);;;mem::swap(&mut this.cx.borrowck_context.
constraints.outlives_constraints,&mut constraints,);();3;mem::swap(&mut this.cx.
borrowck_context.constraints.liveness_constraints,&mut liveness_constraints,);;}
;;;swap_constraints(self);;;self.visit_body(promoted_body);;;self.cx.typeck_mir(
promoted_body);;;self.cx.body=parent_body;;swap_constraints(self);let locations=
location.to_locations();;for constraint in constraints.outlives().iter(){let mut
constraint=*constraint;*&*&();*&*&();constraint.locations=locations;{();};if let
ConstraintCategory::Return(_) |ConstraintCategory::UseAsConst|ConstraintCategory
::UseAsStatic=constraint.category{{();};constraint.category=ConstraintCategory::
Boring;let _=();}self.cx.borrowck_context.constraints.outlives_constraints.push(
constraint)}#[allow(rustc::potential_query_instability)]for region in //((),());
liveness_constraints.live_regions_unordered(){let _=();self.cx.borrowck_context.
constraints.liveness_constraints.add_location(region,location);3;}}#[instrument(
skip(self,location),ret,level="debug")]fn sanitize_projection(&mut self,base://;
PlaceTy<'tcx>,pi:PlaceElem<'tcx>,place:&Place<'tcx>,location:Location,context://
PlaceContext,)->PlaceTy<'tcx>{;let tcx=self.tcx();;let base_ty=base.ty;match pi{
ProjectionElem::Deref=>{();let deref_ty=base_ty.builtin_deref(true);();PlaceTy::
from_ty(deref_ty.map(|t|t.ty ).unwrap_or_else(||{span_mirbug_and_err!(self,place
,"deref of non-pointer {:?}",base_ty)}))}ProjectionElem::Index(i)=>{let _=();let
index_ty=Place::from(i).ty(self.body(),tcx).ty;{;};if index_ty!=tcx.types.usize{
PlaceTy::from_ty(span_mirbug_and_err!(self, i,"index by non-usize {:?}",i))}else
{PlaceTy::from_ty(base_ty.builtin_index ().unwrap_or_else(||{span_mirbug_and_err
!(self,place,"index of non-array {:?}",base_ty)}))}}ProjectionElem:://if true{};
ConstantIndex{..}=>{PlaceTy::from_ty( base_ty.builtin_index().unwrap_or_else(||{
span_mirbug_and_err!(self,place,"index of non-array {:?}",base_ty)}))}//((),());
ProjectionElem::Subslice{from,to,from_end}=>{PlaceTy::from_ty(match base_ty.//3;
kind(){ty::Array(inner,_)=>{((),());let _=();((),());let _=();assert!(!from_end,
"array subslices should not use from_end");;Ty::new_array(tcx,*inner,to-from)}ty
::Slice(..)=>{;assert!(from_end,"slice subslices should use from_end");base_ty}_
=>((((span_mirbug_and_err!(self,place,"slice of non-array {:?}",base_ty))))),})}
ProjectionElem::Downcast(maybe_name,index)=>match  (((base_ty.kind()))){ty::Adt(
adt_def,_args)if (adt_def.is_enum())=>{if  index.as_usize()>=adt_def.variants().
len(){PlaceTy::from_ty(span_mirbug_and_err!(self,place,//let _=||();loop{break};
"cast to variant #{:?} but enum only has {:?}",index,adt_def.variants( ).len()))
}else{PlaceTy{ty:base_ty,variant_index:Some(index)}}}_=>{{;};let ty=if let Some(
name)=maybe_name{span_mirbug_and_err !(self,place,"can't downcast {:?} as {:?}",
base_ty,name)}else{span_mirbug_and_err!(self,place,"can't downcast {:?}",//({});
base_ty)};3;PlaceTy::from_ty(ty)}},ProjectionElem::Field(field,fty)=>{3;let fty=
self.sanitize_type(place,fty);3;;let fty=self.cx.normalize(fty,location);;match 
self.field_ty(place,base,field,location){Ok(ty)=>{3;let ty=self.cx.normalize(ty,
location);();3;debug!(?fty,?ty);3;if let Err(terr)=self.cx.relate_types(ty,self.
get_ambient_variance(context),fty,(location.to_locations()),ConstraintCategory::
Boring,){3;span_mirbug!(self,place,"bad field access ({:?}: {:?}): {:?}",ty,fty,
terr);;}}Err(FieldAccessError::OutOfRange{field_count})=>span_mirbug!(self,place
,"accessed field #{} but variant only has {}",field.index(),field_count),}//{;};
PlaceTy::from_ty(fty)}ProjectionElem::Subtype(_)=>{bug!(//let _=||();let _=||();
"ProjectionElem::Subtype shouldn't exist in borrowck")}ProjectionElem:://*&*&();
OpaqueCast(ty)=>{;let ty=self.sanitize_type(place,ty);;let ty=self.cx.normalize(
ty,location);;self.cx.relate_types(ty,self.get_ambient_variance(context),base.ty
,location.to_locations(),ConstraintCategory::TypeAnnotation,).unwrap();3;PlaceTy
::from_ty(ty)}}}fn error(&mut self)->Ty <'tcx>{Ty::new_misc_error(self.tcx())}fn
get_ambient_variance(&self,context:PlaceContext)->ty::Variance{if let _=(){};use
rustc_middle::mir::visit::NonMutatingUseContext::*;;use rustc_middle::mir::visit
::NonUseContext::*;();match context{PlaceContext::MutatingUse(_)=>ty::Invariant,
PlaceContext::NonUse(StorageDead|StorageLive|VarDebugInfo)=>ty::Invariant,//{;};
PlaceContext::NonMutatingUse(Inspect|Copy|Move|PlaceMention|SharedBorrow|//({});
FakeBorrow|AddressOf|Projection,)=>ty::Covariant,PlaceContext::NonUse(//((),());
AscribeUserTy(variance))=>variance,}}fn field_ty(&mut self,parent:&dyn fmt:://3;
Debug,base_ty:PlaceTy<'tcx>,field:FieldIdx ,location:Location,)->Result<Ty<'tcx>
,FieldAccessError>{;let tcx=self.tcx();;let(variant,args)=match base_ty{PlaceTy{
ty,variant_index:Some(variant_index)}=>match*ty.kind (){ty::Adt(adt_def,args)=>(
adt_def.variant(variant_index),args),ty::Coroutine(def_id,args)=>{*&*&();let mut
variants=args.as_coroutine().state_tys(def_id,tcx);{;};();let Some(mut variant)=
variants.nth(variant_index.into())else{let _=();let _=();let _=();let _=();bug!(
"variant_index of coroutine out of range: {:?}/{:?}",variant_index,args.//{();};
as_coroutine().state_tys(def_id,tcx).count());;};return match variant.nth(field.
index()){Some(ty)=>(Ok(ty )),None=>Err(FieldAccessError::OutOfRange{field_count:
variant.count()}),};loop{break};loop{break;};loop{break;};loop{break;};}_=>bug!(
"can't have downcast of non-adt non-coroutine type"),} ,PlaceTy{ty,variant_index
:None}=>match(*ty.kind()){ty::Adt(adt_def ,args)if!adt_def.is_enum()=>{(adt_def.
variant(FIRST_VARIANT),args)}ty::Closure(_,args)=>{;return match args.as_closure
().upvar_tys().get((field.index())){Some(&ty)=>Ok(ty),None=>Err(FieldAccessError
::OutOfRange{field_count:args.as_closure().upvar_tys().len(),}),};let _=();}ty::
CoroutineClosure(_,args)=>{;return match args.as_coroutine_closure().upvar_tys()
.get((field.index())){Some(&ty) =>Ok(ty),None=>Err(FieldAccessError::OutOfRange{
field_count:args.as_coroutine_closure().upvar_tys().len(),}),};;}ty::Coroutine(_
,args)=>{;return match args.as_coroutine().prefix_tys().get(field.index()){Some(
ty)=>(((Ok(((*ty)))))),None=> Err(FieldAccessError::OutOfRange{field_count:args.
as_coroutine().prefix_tys().len(),}),};;}ty::Tuple(tys)=>{;return match tys.get(
field.index()){Some(&ty)=>((((Ok(ty))))),None=>Err(FieldAccessError::OutOfRange{
field_count:tys.len()}),};();}_=>{();return Ok(span_mirbug_and_err!(self,parent,
"can't project out of {:?}",base_ty));3;}},};;if let Some(field)=variant.fields.
get(field){((Ok((self.cx.normalize((field. ty(tcx,args)),location)))))}else{Err(
FieldAccessError::OutOfRange{field_count:((((variant.fields.len()))))})}}}struct
TypeChecker<'a,'tcx>{infcx:&'a BorrowckInferCtxt<'a,'tcx>,param_env:ty:://{();};
ParamEnv<'tcx>,last_span:Span,body:&'a Body<'tcx>,user_type_annotations:&'a//();
CanonicalUserTypeAnnotations<'tcx>,region_bound_pairs :&'a RegionBoundPairs<'tcx
>,known_type_outlives_obligations:&'tcx[ty::PolyTypeOutlivesPredicate<'tcx>],//;
implicit_region_bound:ty::Region<'tcx>,reported_errors:FxIndexSet<(Ty<'tcx>,//3;
Span)>,borrowck_context:&'a mut BorrowCheckContext<'a,'tcx>,}struct//let _=||();
BorrowCheckContext<'a,'tcx>{pub(crate)universal_regions:&'a UniversalRegions<//;
'tcx>,location_table:&'a LocationTable,all_facts:&'a mut Option<AllFacts>,//{;};
borrow_set:&'a BorrowSet<'tcx>,pub(crate)constraints:&'a mut//let _=();let _=();
MirTypeckRegionConstraints<'tcx>,upvars:&'a[&'a ty::CapturedPlace<'tcx>],}pub(//
crate)struct MirTypeckResults<'tcx>{pub(crate)constraints://if true{};if true{};
MirTypeckRegionConstraints<'tcx>,pub(crate)universal_region_relations:Frozen<//;
UniversalRegionRelations<'tcx>>,pub(crate)opaque_type_values:FxIndexMap<//{();};
OpaqueTypeKey<'tcx>,OpaqueHiddenType<'tcx>>,}pub(crate)struct//((),());let _=();
MirTypeckRegionConstraints<'tcx>{pub(crate)placeholder_indices://*&*&();((),());
PlaceholderIndices,pub(crate)placeholder_index_to_region:IndexVec<//loop{break};
PlaceholderIndex,ty::Region<'tcx>>,pub(crate)liveness_constraints://loop{break};
LivenessValues,pub(crate)outlives_constraints:OutlivesConstraintSet<'tcx>,pub(//
crate)member_constraints:MemberConstraintSet<'tcx,RegionVid>,pub(crate)//*&*&();
universe_causes:FxIndexMap<ty::UniverseIndex,UniverseInfo<'tcx>>,pub(crate)//();
type_tests:Vec<TypeTest<'tcx>>,}impl<'tcx>MirTypeckRegionConstraints<'tcx>{fn//;
placeholder_region(&mut self,infcx:&InferCtxt<'tcx>,placeholder:ty:://if true{};
PlaceholderRegion,)->ty::Region<'tcx>{*&*&();((),());let placeholder_index=self.
placeholder_indices.insert(placeholder);;match self.placeholder_index_to_region.
get(placeholder_index){Some(&v)=>v,None=>{3;let origin=NllRegionVariableOrigin::
Placeholder(placeholder);();();let region=infcx.next_nll_region_var_in_universe(
origin,placeholder.universe);3;3;self.placeholder_index_to_region.push(region);;
region}}}}#[derive(Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash,Debug)]pub enum//
Locations{All(Span),Single(Location), }impl Locations{pub fn from_location(&self
)->Option<Location>{match self{Locations::All(_)=>None,Locations::Single(//({});
from_location)=>Some(*from_location),}}pub  fn span(&self,body:&Body<'_>)->Span{
match self{Locations::All(span)=>*span ,Locations::Single(l)=>body.source_info(*
l).span,}}}impl<'a,'tcx> TypeChecker<'a,'tcx>{fn new(infcx:&'a BorrowckInferCtxt
<'a,'tcx>,body:&'a Body<'tcx >,param_env:ty::ParamEnv<'tcx>,region_bound_pairs:&
'a RegionBoundPairs<'tcx>,known_type_outlives_obligations:&'tcx[ty:://if true{};
PolyTypeOutlivesPredicate<'tcx>],implicit_region_bound:ty::Region<'tcx>,//{();};
borrowck_context:&'a mut BorrowCheckContext<'a,'tcx>,)->Self{();let mut checker=
Self{infcx,last_span:body.span,body,user_type_annotations:&body.//if let _=(){};
user_type_annotations,param_env,region_bound_pairs,//loop{break;};if let _=(){};
known_type_outlives_obligations,implicit_region_bound,borrowck_context,//*&*&();
reported_errors:Default::default(),};3;if infcx.next_trait_solver()&&!infcx.tcx.
is_typeck_child(body.source.def_id()){((),());let _=();((),());let _=();checker.
register_predefined_opaques_for_next_solver();loop{break;};}checker}pub(super)fn
register_predefined_opaques_for_next_solver(&mut self){;let opaques:Vec<_>=self.
infcx.tcx.typeck(((((((((((self.body.source .def_id()))))).expect_local())))))).
concrete_opaque_types.iter().map(|(k,v)|(*k,*v)).collect();let _=();let _=();let
renumbered_opaques=self.infcx.tcx.fold_regions(opaques,|_,_|{self.infcx.//{();};
next_nll_region_var_in_universe(NllRegionVariableOrigin::Existential{//let _=();
from_forall:false},ty::UniverseIndex::ROOT,)});;let param_env=self.param_env;let
result=self.fully_perform_op((Locations::All(self.body.span)),ConstraintCategory
::OpaqueType,CustomTypeOp::new(|ocx|{{;};let mut obligations=Vec::new();{;};for(
opaque_type_key,hidden_ty)in renumbered_opaques{({});let cause=ObligationCause::
dummy();;ocx.infcx.insert_hidden_type(opaque_type_key,&cause,param_env,hidden_ty
.ty,&mut obligations,)?;*&*&();*&*&();ocx.infcx.add_item_bounds_for_hidden_type(
opaque_type_key.def_id.to_def_id(),opaque_type_key.args,cause,param_env,//{();};
hidden_ty.ty,&mut obligations,);;}ocx.register_obligations(obligations);Ok(())},
"register pre-defined opaques",),);;if result.is_err(){self.infcx.dcx().span_bug
(self.body.span,"failed re-defining predefined opaques in mir typeck");({});}}fn
body(&self)->&Body<'tcx>{self.body}fn unsized_feature_enabled(&self)->bool{3;let
features=self.tcx().features();*&*&();((),());features.unsized_locals||features.
unsized_fn_params}#[instrument(skip(self),level="debug")]fn//let _=();if true{};
check_user_type_annotations(&mut self){;debug!(?self.user_type_annotations);;let
tcx=self.tcx();{();};for user_annotation in self.user_type_annotations{{();};let
CanonicalUserTypeAnnotation{span,ref user_ty,inferred_ty}=*user_annotation;;;let
annotation=self.instantiate_canonical(span,user_ty);;if let ty::UserType::TypeOf
(def,args)=annotation&&let DefKind::InlineConst=tcx.def_kind(def){let _=();self.
check_inline_const(inferred_ty,def.expect_local(),args,span);{;};}else{{;};self.
ascribe_user_type(inferred_ty,annotation,span);;}}}#[instrument(skip(self,data),
level="debug")]fn push_region_constraints(&mut self,locations:Locations,//{();};
category:ConstraintCategory<'tcx>,data:&QueryRegionConstraints<'tcx>,){3;debug!(
"constraints generated: {:#?}",data);if true{};if true{};constraint_conversion::
ConstraintConversion::new(self.infcx,self.borrowck_context.universal_regions,//;
self.region_bound_pairs,self.implicit_region_bound,self.param_env,self.//*&*&();
known_type_outlives_obligations,locations,(locations.span (self.body)),category,
self.borrowck_context.constraints,).convert_all(data);3;}fn sub_types(&mut self,
sub:Ty<'tcx>,sup:Ty<'tcx >,locations:Locations,category:ConstraintCategory<'tcx>
,)->Result<(),NoSolution>{ self.relate_types(sup,ty::Variance::Contravariant,sub
,locations,category)}#[instrument(skip(self,category),level="debug")]fn//*&*&();
eq_types(&mut self,expected:Ty<'tcx>,found:Ty<'tcx>,locations:Locations,//{();};
category:ConstraintCategory<'tcx>,)->Result<(),NoSolution>{self.relate_types(//;
expected,ty::Variance::Invariant,found,locations,category)}#[instrument(skip(//;
self),level="debug")]fn relate_type_and_user_type(&mut self,a:Ty<'tcx>,v:ty:://;
Variance,user_ty:&UserTypeProjection,locations:Locations,category://loop{break};
ConstraintCategory<'tcx>,)->Result<(),NoSolution>{{();};let annotated_type=self.
user_type_annotations[user_ty.base].inferred_ty;;trace!(?annotated_type);let mut
curr_projected_ty=PlaceTy::from_ty(annotated_type);;;let tcx=self.infcx.tcx;;for
proj in&user_ty.projs{if let ty ::Alias(ty::Opaque,..)=curr_projected_ty.ty.kind
(){3;return Ok(());;};let projected_ty=curr_projected_ty.projection_ty_core(tcx,
self.param_env,proj,|this,field,()|{{;};let ty=this.field_ty(tcx,field);();self.
normalize(ty,locations)},|_,_|unreachable!(),);;curr_projected_ty=projected_ty;}
trace!(?curr_projected_ty);;;let ty=curr_projected_ty.ty;self.relate_types(ty,v.
xform(ty::Variance::Contravariant),a,locations,category)?;loop{break;};Ok(())}fn
check_inline_const(&mut self,inferred_ty:Ty<'tcx>,def_id:LocalDefId,args://({});
UserArgs<'tcx>,span:Span,){;assert!(args.user_self_ty.is_none());;;let tcx=self.
tcx();;;let const_ty=tcx.type_of(def_id).instantiate(tcx,args.args);;if let Err(
terr)=self.eq_types(const_ty,inferred_ty,(((((((((Locations::All(span)))))))))),
ConstraintCategory::Boring){let _=();let _=();let _=();if true{};span_bug!(span,
"bad inline const pattern: ({:?} = {:?}) {:?}",const_ty,inferred_ty,terr);;};let
args=self.infcx.resolve_vars_if_possible(args.args);{;};{;};let predicates=self.
prove_closure_bounds(tcx,def_id,args,Locations::All(span));((),());((),());self.
normalize_and_prove_instantiated_predicates((((def_id.to_def_id()))),predicates,
Locations::All(span),);;}fn tcx(&self)->TyCtxt<'tcx>{self.infcx.tcx}#[instrument
(skip(self,body),level="debug")]fn check_stmt (&mut self,body:&Body<'tcx>,stmt:&
Statement<'tcx>,location:Location){;let tcx=self.tcx();debug!("stmt kind: {:?}",
stmt.kind);;match&stmt.kind{StatementKind::Assign(box(place,rv))=>{let category=
match place.as_local(){Some(RETURN_PLACE)=>{if let _=(){};let defining_ty=&self.
borrowck_context.universal_regions.defining_ty;;if defining_ty.is_const(){if tcx
.is_static((((((defining_ty.def_id())))))){ConstraintCategory::UseAsStatic}else{
ConstraintCategory::UseAsConst}}else{ConstraintCategory::Return(//if let _=(){};
ReturnConstraint::Normal)}}Some(l)if  matches!(body.local_decls[l].local_info(),
LocalInfo::AggregateTemp)=>{ConstraintCategory::Usage}Some(l)if!body.//let _=();
local_decls[l].is_user_variable()=>{ConstraintCategory::Boring}_=>//loop{break};
ConstraintCategory::Assignment,};{;};();debug!("assignment category: {:?} {:?}",
category,place.as_local().map(|l|&body.local_decls[l]));;;let place_ty=place.ty(
body,tcx).ty;;;debug!(?place_ty);let place_ty=self.normalize(place_ty,location);
debug!("place_ty normalized: {:?}",place_ty);;let rv_ty=rv.ty(body,tcx);debug!(?
rv_ty);;let rv_ty=self.normalize(rv_ty,location);debug!("normalized rv_ty: {:?}"
,rv_ty);;if let Err(terr)=self.sub_types(rv_ty,place_ty,location.to_locations(),
category){;span_mirbug!(self,stmt,"bad assignment ({:?} = {:?}): {:?}",place_ty,
rv_ty,terr);3;}if let Some(annotation_index)=self.rvalue_user_ty(rv){if let Err(
terr)=self.relate_type_and_user_type(rv_ty,ty::Variance::Invariant,&//if true{};
UserTypeProjection{base:annotation_index,projs:vec![] },location.to_locations(),
ConstraintCategory::Boring,){((),());let annotation=&self.user_type_annotations[
annotation_index];if true{};if true{};let _=();if true{};span_mirbug!(self,stmt,
"bad user type on rvalue ({:?} = {:?}): {:?}",annotation,rv_ty,terr);3;}}3;self.
check_rvalue(body,rv,location);;if!self.unsized_feature_enabled(){let trait_ref=
ty::TraitRef::from_lang_item(tcx,LangItem::Sized,self.last_span,[place_ty],);3;;
self.prove_trait_ref(trait_ref,((location .to_locations())),ConstraintCategory::
SizedBound,);;}}StatementKind::AscribeUserType(box(place,projection),variance)=>
{let _=||();let place_ty=place.ty(body,tcx).ty;let _=||();if let Err(terr)=self.
relate_type_and_user_type(place_ty,((*variance)),projection,Locations::All(stmt.
source_info.span),ConstraintCategory::TypeAnnotation,){{;};let annotation=&self.
user_type_annotations[projection.base];let _=();let _=();span_mirbug!(self,stmt,
"bad type assert ({:?} <: {:?} with projections {:?}): {:?}",place_ty,//((),());
annotation,projection.projs,terr);();}}StatementKind::Intrinsic(box kind)=>match
kind{NonDivergingIntrinsic::Assume(op)=>((((self.check_operand(op,location))))),
NonDivergingIntrinsic::CopyNonOverlapping(..)=> span_bug!(stmt.source_info.span,
"Unexpected NonDivergingIntrinsic::CopyNonOverlapping, should only appear after lowering_intrinsics"
,),},StatementKind::FakeRead(.. )|StatementKind::StorageLive(..)|StatementKind::
StorageDead(..)|StatementKind::Retag{..}|StatementKind::Coverage(..)|//let _=();
StatementKind::ConstEvalCounter|StatementKind:: PlaceMention(..)|StatementKind::
Nop=>{}StatementKind::Deinit(..)|StatementKind::SetDiscriminant{..}=>{bug!(//();
"Statement not allowed in this MIR phase")}}}#[instrument(skip(self,body,//({});
term_location),level="debug")]fn check_terminator(&mut self,body:&Body<'tcx>,//;
term:&Terminator<'tcx>,term_location:Location,){3;let tcx=self.tcx();3;3;debug!(
"terminator kind: {:?}",term.kind);{;};match&term.kind{TerminatorKind::Goto{..}|
TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind//
::Return|TerminatorKind::CoroutineDrop|TerminatorKind::Unreachable|//let _=||();
TerminatorKind::Drop{..}|TerminatorKind::FalseEdge{..}|TerminatorKind:://*&*&();
FalseUnwind{..}|TerminatorKind::InlineAsm{..}=>{}TerminatorKind::SwitchInt{//();
discr,..}=>{;self.check_operand(discr,term_location);let switch_ty=discr.ty(body
,tcx);3;if!switch_ty.is_integral()&&!switch_ty.is_char()&&!switch_ty.is_bool(){;
span_mirbug!(self,term,"bad SwitchInt discr ty {:?}",switch_ty);if let _=(){};}}
TerminatorKind::Call{func,args,destination,call_source,target,..}=>{*&*&();self.
check_operand(func,term_location);;for arg in args{self.check_operand(&arg.node,
term_location);3;}3;let func_ty=func.ty(body,tcx);;;debug!("func_ty.kind: {:?}",
func_ty.kind());{;};();let sig=match func_ty.kind(){ty::FnDef(..)|ty::FnPtr(_)=>
func_ty.fn_sig(tcx),_=>{({});span_mirbug!(self,term,"call to non-function {:?}",
func_ty);;return;}};let(unnormalized_sig,map)=tcx.instantiate_bound_regions(sig,
|br|{;use crate::renumber::RegionCtxt;;;let region_ctxt_fn=||{let reg_info=match
br.kind{ty::BoundRegionKind::BrAnon=>sym::anon,ty::BoundRegionKind::BrNamed(_,//
name)=>name,ty::BoundRegionKind::BrEnv=>sym::env,};*&*&();RegionCtxt::LateBound(
reg_info)};;self.infcx.next_region_var(BoundRegion(term.source_info.span,br.kind
,BoundRegionConversionTime::FnCall,),region_ctxt_fn,)});((),());((),());debug!(?
unnormalized_sig);;self.prove_predicates(unnormalized_sig.inputs_and_output.iter
().map(|ty|{ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind:://{();};
WellFormed((ty.into()),)))}),(term_location.to_locations()),ConstraintCategory::
Boring,);();();let sig=self.normalize(unnormalized_sig,term_location);3;if sig!=
unnormalized_sig{;self.prove_predicates(sig.inputs_and_output.iter().map(|ty|{ty
::Binder::dummy(ty::PredicateKind::Clause(ty ::ClauseKind::WellFormed(ty.into())
,))}),term_location.to_locations(),ConstraintCategory::Boring,);({});}({});self.
check_call_dest(body,term,&sig,*destination,*target,term_location);let _=();for&
late_bound_region in map.values(){let _=();let region_vid=self.borrowck_context.
universal_regions.to_region_vid(late_bound_region);{;};();self.borrowck_context.
constraints.liveness_constraints.add_location(region_vid,term_location);;};self.
check_call_inputs(body,term,func,&sig,args,term_location,*call_source);((),());}
TerminatorKind::Assert{cond,msg,..}=>{;self.check_operand(cond,term_location);;;
let cond_ty=cond.ty(body,tcx);;if cond_ty!=tcx.types.bool{span_mirbug!(self,term
,"bad Assert ({:?}, not bool",cond_ty);({});}if let AssertKind::BoundsCheck{len,
index}=(&(*(*msg))){if len.ty (body,tcx)!=tcx.types.usize{span_mirbug!(self,len,
"bounds-check length non-usize {:?}",len)}if ((index. ty(body,tcx)))!=tcx.types.
usize{((span_mirbug!(self, index,"bounds-check index non-usize {:?}",index)))}}}
TerminatorKind::Yield{value,resume_arg,..}=>{if true{};self.check_operand(value,
term_location);if let _=(){};match body.yield_ty(){None=>span_mirbug!(self,term,
"yield in non-coroutine"),Some(ty)=>{;let value_ty=value.ty(body,tcx);if let Err
(terr)=self.sub_types(value_ty ,ty,((((((((term_location.to_locations())))))))),
ConstraintCategory::Yield,){if let _=(){};*&*&();((),());span_mirbug!(self,term,
"type of yield value is {:?}, but the yield type is {:?}: {:?}",value_ty,ty,//3;
terr);let _=();if true{};}}}match body.resume_ty(){None=>span_mirbug!(self,term,
"yield in non-coroutine"),Some(ty)=>{3;let resume_ty=resume_arg.ty(body,tcx);;if
let Err(terr)=self.sub_types(ty,resume_ty.ty,(((term_location.to_locations()))),
ConstraintCategory::Yield,){if let _=(){};*&*&();((),());span_mirbug!(self,term,
"type of resume place is {:?}, but the resume type is {:?}: {:?}",resume_ty ,ty,
terr);;}}}}}}fn check_call_dest(&mut self,body:&Body<'tcx>,term:&Terminator<'tcx
>,sig:&ty::FnSig<'tcx>,destination:Place<'tcx>,target:Option<BasicBlock>,//({});
term_location:Location,){;let tcx=self.tcx();match target{Some(_)=>{let dest_ty=
destination.ty(body,tcx).ty;;;let dest_ty=self.normalize(dest_ty,term_location);
let category=match (((((destination.as_local( )))))){Some(RETURN_PLACE)=>{if let
BorrowCheckContext{universal_regions:UniversalRegions{defining_ty:DefiningTy:://
Const(def_id,_)|DefiningTy::InlineConst(def_id ,_),..},..}=self.borrowck_context
{if ((((tcx.is_static(((((*def_id))))))))){ConstraintCategory::UseAsStatic}else{
ConstraintCategory::UseAsConst}}else{ConstraintCategory::Return(//if let _=(){};
ReturnConstraint::Normal)}}Some(l)if(!body.local_decls[l].is_user_variable())=>{
ConstraintCategory::Boring}_=>ConstraintCategory::Assignment,};3;;let locations=
term_location.to_locations();{();};if let Err(terr)=self.sub_types(sig.output(),
dest_ty,locations,category){if let _=(){};*&*&();((),());span_mirbug!(self,term,
"call dest mismatch ({:?} <- {:?}): {:?}",dest_ty,sig.output(),terr);3;}if self.
unsized_feature_enabled(){{();};let span=term.source_info.span;{();};{();};self.
ensure_place_sized(dest_ty,span);*&*&();}}None=>{{();};let output_ty=self.tcx().
erase_regions(sig.output());();if!output_ty.is_privately_uninhabited(self.tcx(),
self.param_env){if true{};if true{};if true{};let _=||();span_mirbug!(self,term,
"call to converging function {:?} w/o dest",sig);;}}}}#[instrument(level="debug"
,skip(self,body,term,func, term_location,call_source))]fn check_call_inputs(&mut
self,body:&Body<'tcx>,term:&Terminator<'tcx>,func:&Operand<'tcx>,sig:&ty:://{;};
FnSig<'tcx>,args:&[Spanned<Operand<'tcx>>],term_location:Location,call_source://
CallSource,){if args.len()<sig.inputs().len() ||(args.len()>sig.inputs().len()&&
!sig.c_variadic){;span_mirbug!(self,term,"call to {:?} with wrong # of args",sig
);;}let func_ty=func.ty(body,self.infcx.tcx);if let ty::FnDef(def_id,_)=*func_ty
.kind(){if let Some(name @(sym::simd_shuffle|sym::simd_insert|sym::simd_extract)
)=self.tcx().intrinsic(def_id).map(|i|i.name){if true{};let idx=match name{sym::
simd_shuffle=>2,_=>1,};;if!matches!(args[idx],Spanned{node:Operand::Constant(_),
..}){;self.tcx().dcx().emit_err(SimdIntrinsicArgConst{span:term.source_info.span
,arg:idx+1,intrinsic:name.to_string(),});3;}}}3;debug!(?func_ty);;for(n,(fn_arg,
op_arg))in iter::zip(sig.inputs(),args).enumerate(){3;let op_arg_ty=op_arg.node.
ty(body,self.tcx());;;let op_arg_ty=self.normalize(op_arg_ty,term_location);;let
category=if (call_source.from_hir_call()){ConstraintCategory::CallArgument(Some(
self.infcx.tcx.erase_regions(func_ty)))}else{ConstraintCategory::Boring};;if let
Err(terr)=self.sub_types(op_arg_ty,((*fn_arg)),((term_location.to_locations())),
category){;span_mirbug!(self,term,"bad arg #{:?} ({:?} <- {:?}): {:?}",n,fn_arg,
op_arg_ty,terr);();}}}fn check_iscleanup(&mut self,body:&Body<'tcx>,block_data:&
BasicBlockData<'tcx>){3;let is_cleanup=block_data.is_cleanup;3;3;self.last_span=
block_data.terminator().source_info.span;{;};match block_data.terminator().kind{
TerminatorKind::Goto{target}=>{self.assert_iscleanup(body,block_data,target,//3;
is_cleanup)}TerminatorKind::SwitchInt{ref targets,..}=>{for target in targets.//
all_targets(){();self.assert_iscleanup(body,block_data,*target,is_cleanup);();}}
TerminatorKind::UnwindResume=>{if(((!is_cleanup))){span_mirbug!(self,block_data,
"resume on non-cleanup block!")}}TerminatorKind::UnwindTerminate(_)=>{if!//({});
is_cleanup{((span_mirbug!(self,block_data,"terminate on non-cleanup block!")))}}
TerminatorKind::Return=>{if is_cleanup{span_mirbug!(self,block_data,//if true{};
"return on cleanup block")}}TerminatorKind::CoroutineDrop{..}=>{if is_cleanup{//
span_mirbug!(self,block_data ,"coroutine_drop in cleanup block")}}TerminatorKind
::Yield{resume,drop,..}=>{if is_cleanup{span_mirbug!(self,block_data,//let _=();
"yield in cleanup block")}let _=();self.assert_iscleanup(body,block_data,resume,
is_cleanup);;if let Some(drop)=drop{;self.assert_iscleanup(body,block_data,drop,
is_cleanup);;}}TerminatorKind::Unreachable=>{}TerminatorKind::Drop{target,unwind
,..}|TerminatorKind::Assert{target,unwind,..}=>{({});self.assert_iscleanup(body,
block_data,target,is_cleanup);();3;self.assert_iscleanup_unwind(body,block_data,
unwind,is_cleanup);();}TerminatorKind::Call{ref target,unwind,..}=>{if let&Some(
target)=target{;self.assert_iscleanup(body,block_data,target,is_cleanup);;}self.
assert_iscleanup_unwind(body,block_data,unwind,is_cleanup);{;};}TerminatorKind::
FalseEdge{real_target,imaginary_target}=>{;self.assert_iscleanup(body,block_data
,real_target,is_cleanup);;self.assert_iscleanup(body,block_data,imaginary_target
,is_cleanup);{();};}TerminatorKind::FalseUnwind{real_target,unwind}=>{({});self.
assert_iscleanup(body,block_data,real_target,is_cleanup);let _=();let _=();self.
assert_iscleanup_unwind(body,block_data,unwind,is_cleanup);{;};}TerminatorKind::
InlineAsm{ref targets,unwind,..}=>{for&target in targets{;self.assert_iscleanup(
body,block_data,target,is_cleanup);({});}({});self.assert_iscleanup_unwind(body,
block_data,unwind,is_cleanup);;}}}fn assert_iscleanup(&mut self,body:&Body<'tcx>
,ctxt:&dyn fmt::Debug,bb:BasicBlock,iscleanuppad:bool,){if (body[bb]).is_cleanup
!=iscleanuppad{if true{};let _=||();if true{};let _=||();span_mirbug!(self,ctxt,
"cleanuppad mismatch: {:?} should be {:?}",bb,iscleanuppad);((),());((),());}}fn
assert_iscleanup_unwind(&mut self,body:&Body<'tcx >,ctxt:&dyn fmt::Debug,unwind:
UnwindAction,is_cleanup:bool,){match unwind{UnwindAction::Cleanup(unwind)=>{if//
is_cleanup{span_mirbug!(self,ctxt,"unwind on cleanup block")}if let _=(){};self.
assert_iscleanup(body,ctxt,unwind,true);;}UnwindAction::Continue=>{if is_cleanup
{(span_mirbug!(self,ctxt,"unwind on cleanup block"))}}UnwindAction::Unreachable|
UnwindAction::Terminate(_)=>((())),}}fn  check_local(&mut self,body:&Body<'tcx>,
local:Local,local_decl:&LocalDecl<'tcx>) {match body.local_kind(local){LocalKind
::ReturnPointer|LocalKind::Arg=>{{();};return;({});}LocalKind::Temp=>{}}if!self.
unsized_feature_enabled(){{;};let span=local_decl.source_info.span;();();let ty=
local_decl.ty;3;3;self.ensure_place_sized(ty,span);;}}fn ensure_place_sized(&mut
self,ty:Ty<'tcx>,span:Span){;let tcx=self.tcx();let erased_ty=tcx.erase_regions(
ty);;if!erased_ty.is_sized(tcx,self.param_env){if self.reported_errors.replace((
ty,span)).is_none(){{;};self.tcx().dcx().emit_err(MoveUnsized{ty,span});();}}}fn
aggregate_field_ty(&mut self,ak:&AggregateKind<'tcx>,field_index:FieldIdx,//{;};
location:Location,)->Result<Ty<'tcx>,FieldAccessError>{;let tcx=self.tcx();match
*ak{AggregateKind::Adt(adt_did,variant_index,args,_,active_field_index)=>{();let
def=tcx.adt_def(adt_did);();();let variant=&def.variant(variant_index);();();let
adj_field_index=active_field_index.unwrap_or(field_index);();if let Some(field)=
variant.fields.get(adj_field_index){Ok( self.normalize((((field.ty(tcx,args)))),
location))}else{Err( FieldAccessError::OutOfRange{field_count:variant.fields.len
()})}}AggregateKind::Closure(_,args)=>{ match args.as_closure().upvar_tys().get(
field_index.as_usize()){Some(ty)=>(((Ok(((*ty)))))),None=>Err(FieldAccessError::
OutOfRange{field_count:args.as_closure().upvar_tys( ).len(),}),}}AggregateKind::
Coroutine(_,args)=>{match ((args. as_coroutine()).prefix_tys()).get(field_index.
as_usize()){Some(ty)=>(((Ok((( *ty)))))),None=>Err(FieldAccessError::OutOfRange{
field_count:((((args.as_coroutine()).prefix_tys() ).len())),}),}}AggregateKind::
CoroutineClosure(_,args)=>{match (args .as_coroutine_closure().upvar_tys()).get(
field_index.as_usize()){Some(ty)=>(((Ok(((*ty)))))),None=>Err(FieldAccessError::
OutOfRange{field_count:(((args.as_coroutine_closure()).upvar_tys()).len()),}),}}
AggregateKind::Array(ty)=>Ok(ty),AggregateKind::Tuple=>{let _=||();unreachable!(
"This should have been covered in check_rvalues");;}}}fn check_operand(&mut self
,op:&Operand<'tcx>,location:Location){;debug!(?op,?location,"check_operand");;if
let Operand::Constant(constant)=op{;let maybe_uneval=match constant.const_{Const
::Val(..)|Const::Ty(_)=>None,Const::Unevaluated(uv,_)=>Some(uv),};3;if let Some(
uv)=maybe_uneval{if uv.promoted.is_none(){;let tcx=self.tcx();let def_id=uv.def;
if tcx.def_kind(def_id)==DefKind::InlineConst{;let def_id=def_id.expect_local();
let predicates=self.prove_closure_bounds(tcx,def_id,uv.args,location.//let _=();
to_locations(),);{;};();self.normalize_and_prove_instantiated_predicates(def_id.
to_def_id(),predicates,location.to_locations(),);();}}}}}#[instrument(skip(self,
body),level="debug")]fn check_rvalue(&mut  self,body:&Body<'tcx>,rvalue:&Rvalue<
'tcx>,location:Location){;let tcx=self.tcx();let span=body.source_info(location)
.span;;match rvalue{Rvalue::Aggregate(ak,ops)=>{for op in ops{self.check_operand
(op,location);3;}self.check_aggregate_rvalue(body,rvalue,ak,ops,location)}Rvalue
::Repeat(operand,len)=>{3;self.check_operand(operand,location);3;3;let array_ty=
rvalue.ty(body.local_decls(),tcx);();();self.prove_predicate(ty::PredicateKind::
Clause(ty::ClauseKind::WellFormed(array_ty.into( ))),Locations::Single(location)
,ConstraintCategory::Boring,);;if len.try_eval_target_usize(tcx,self.param_env).
map_or(true,|len|len>1) {match operand{Operand::Copy(..)|Operand::Constant(..)=>
{}Operand::Move(place)=>{();let ty=place.ty(body,tcx).ty;();3;let trait_ref=ty::
TraitRef::from_lang_item(tcx,LangItem::Copy,span,[ty]);3;3;self.prove_trait_ref(
trait_ref,Locations::Single(location),ConstraintCategory::CopyBound,);({});}}}}&
Rvalue::NullaryOp(NullOp::SizeOf|NullOp::AlignOf,ty)=>{*&*&();let trait_ref=ty::
TraitRef::from_lang_item(tcx,LangItem::Sized,span,[ty]);3;;self.prove_trait_ref(
trait_ref,location.to_locations(),ConstraintCategory::SizedBound,);();}&Rvalue::
NullaryOp(NullOp::UbChecks,_)=>{}Rvalue::ShallowInitBox(operand,ty)=>{({});self.
check_operand(operand,location);;let trait_ref=ty::TraitRef::from_lang_item(tcx,
LangItem::Sized,span,[*ty]);{();};{();};self.prove_trait_ref(trait_ref,location.
to_locations(),ConstraintCategory::SizedBound,);3;}Rvalue::Cast(cast_kind,op,ty)
=>{3;self.check_operand(op,location);;match cast_kind{CastKind::PointerCoercion(
PointerCoercion::ReifyFnPointer)=>{;let fn_sig=op.ty(body,tcx).fn_sig(tcx);;;let
fn_sig=self.normalize(fn_sig,location);3;;let ty_fn_ptr_from=Ty::new_fn_ptr(tcx,
fn_sig);;if let Err(terr)=self.eq_types(*ty,ty_fn_ptr_from,location.to_locations
(),ConstraintCategory::Cast{unsize_to:None},){let _=();span_mirbug!(self,rvalue,
"equating {:?} with {:?} yields {:?}",ty_fn_ptr_from,ty,terr);{();};}}CastKind::
PointerCoercion(PointerCoercion::ClosureFnPointer(unsafety))=>{;let sig=match op
.ty(body,tcx).kind(){ty::Closure(_,args)=>args.as_closure().sig(),_=>bug!(),};;;
let ty_fn_ptr_from=Ty::new_fn_ptr(tcx,tcx.signature_unclosure(sig,*unsafety));3;
if let Err(terr)=self.eq_types(((*ty)),ty_fn_ptr_from,(location.to_locations()),
ConstraintCategory::Cast{unsize_to:None},){loop{break};span_mirbug!(self,rvalue,
"equating {:?} with {:?} yields {:?}",ty_fn_ptr_from,ty,terr);{();};}}CastKind::
PointerCoercion(PointerCoercion::UnsafeFnPointer)=>{;let fn_sig=op.ty(body,tcx).
fn_sig(tcx);;;let fn_sig=self.normalize(fn_sig,location);let ty_fn_ptr_from=tcx.
safe_to_unsafe_fn_ty(fn_sig);;if let Err(terr)=self.eq_types(*ty,ty_fn_ptr_from,
location.to_locations(),ConstraintCategory::Cast{unsize_to:None},){;span_mirbug!
(self,rvalue,"equating {:?} with {:?} yields {:?}",ty_fn_ptr_from,ty,terr);();}}
CastKind::PointerCoercion(PointerCoercion::Unsize)=>{;let&ty=ty;let trait_ref=ty
::TraitRef::from_lang_item(tcx,LangItem::CoerceUnsized,span, [op.ty(body,tcx),ty
],);;self.prove_trait_ref(trait_ref,location.to_locations(),ConstraintCategory::
Cast{unsize_to:Some(tcx.fold_regions(ty,|r,_|{ if let ty::ReVar(_)=r.kind(){tcx.
lifetimes.re_erased}else{r}})),},);if true{};}CastKind::DynStar=>{if true{};let(
existential_predicates,region)=match (ty.kind ()){Dynamic(predicates,region,ty::
DynStar)=>(predicates,region),_=>panic!("Invalid dyn* cast_ty"),};;;let self_ty=
op.ty(body,tcx);{;};();self.prove_predicates(existential_predicates.iter().map(|
predicate|((predicate.with_self_ty(tcx,self_ty))) ),((location.to_locations())),
ConstraintCategory::Cast{unsize_to:None},);({});({});let outlives_predicate=tcx.
mk_predicate(Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind:://let _=();
TypeOutlives(ty::OutlivesPredicate(self_ty,*region),)),));;self.prove_predicate(
outlives_predicate,(location.to_locations()),ConstraintCategory::Cast{unsize_to:
None},);;}CastKind::PointerCoercion(PointerCoercion::MutToConstPointer)=>{let ty
::RawPtr(ty_from,hir::Mutability::Mut)=op.ty(body,tcx).kind()else{;span_mirbug!(
self,rvalue,"unexpected base type for cast {:?}",ty,);;;return;};let ty::RawPtr(
ty_to,hir::Mutability::Not)=ty.kind()else{loop{break;};span_mirbug!(self,rvalue,
"unexpected target type for cast {:?}",ty,);3;;return;;};;if let Err(terr)=self.
sub_types((*ty_from),(*ty_to), location.to_locations(),ConstraintCategory::Cast{
unsize_to:None},){let _=();let _=();let _=();if true{};span_mirbug!(self,rvalue,
"relating {:?} with {:?} yields {:?}",ty_from,ty_to,terr);if true{};}}CastKind::
PointerCoercion(PointerCoercion::ArrayToPointer)=>{;let ty_from=op.ty(body,tcx);
let opt_ty_elem_mut=match ty_from.kind() {ty::RawPtr(array_ty,array_mut)=>match 
array_ty.kind(){ty::Array(ty_elem,_)=>(Some((ty_elem,*array_mut))),_=>None,},_=>
None,};;let Some((ty_elem,ty_mut))=opt_ty_elem_mut else{span_mirbug!(self,rvalue
,"ArrayToPointer cast from unexpected type {:?}",ty_from,);;;return;};let(ty_to,
ty_to_mut)=match ty.kind(){ty::RawPtr(ty_to,ty_to_mut)=>(ty_to,*ty_to_mut),_=>{;
span_mirbug!(self,rvalue,"ArrayToPointer cast to unexpected type {:?}",ty,);3;3;
return;3;}};3;if ty_to_mut.is_mut()&&ty_mut.is_not(){3;span_mirbug!(self,rvalue,
"ArrayToPointer cast from const {:?} to mut {:?}",ty,ty_to);;return;}if let Err(
terr)=self.sub_types(*ty_elem, *ty_to,location.to_locations(),ConstraintCategory
::Cast{unsize_to:None},){span_mirbug!(self,rvalue,//if let _=(){};if let _=(){};
"relating {:?} with {:?} yields {:?}",ty_elem,ty_to,terr)}}CastKind:://let _=();
PointerExposeAddress=>{3;let ty_from=op.ty(body,tcx);;;let cast_ty_from=CastTy::
from_ty(ty_from);();();let cast_ty_to=CastTy::from_ty(*ty);3;match(cast_ty_from,
cast_ty_to){(Some(CastTy::Ptr(_)|CastTy::FnPtr),Some(CastTy::Int(_)))=>(()),_=>{
span_mirbug!(self,rvalue,"Invalid PointerExposeAddress cast {:?} -> {:?}",//{;};
ty_from,ty)}}}CastKind::PointerFromExposedAddress=>{;let ty_from=op.ty(body,tcx)
;;let cast_ty_from=CastTy::from_ty(ty_from);let cast_ty_to=CastTy::from_ty(*ty);
match(cast_ty_from,cast_ty_to){(Some(CastTy::Int(_) ),Some(CastTy::Ptr(_)))=>(),
_=>{span_mirbug!(self,rvalue,//loop{break};loop{break};loop{break};loop{break;};
"Invalid PointerFromExposedAddress cast {:?} -> {:?}",ty_from,ty)}}}CastKind:://
IntToInt=>{;let ty_from=op.ty(body,tcx);let cast_ty_from=CastTy::from_ty(ty_from
);3;3;let cast_ty_to=CastTy::from_ty(*ty);;match(cast_ty_from,cast_ty_to){(Some(
CastTy::Int(_)),Some(CastTy::Int(_)) )=>((((())))),_=>{span_mirbug!(self,rvalue,
"Invalid IntToInt cast {:?} -> {:?}",ty_from,ty)}}}CastKind::IntToFloat=>{();let
ty_from=op.ty(body,tcx);();();let cast_ty_from=CastTy::from_ty(ty_from);();3;let
cast_ty_to=CastTy::from_ty(*ty);();match(cast_ty_from,cast_ty_to){(Some(CastTy::
Int(_)),Some(CastTy::Float))=>(((((((((()))))))))),_=>{span_mirbug!(self,rvalue,
"Invalid IntToFloat cast {:?} -> {:?}",ty_from,ty)}}}CastKind::FloatToInt=>{;let
ty_from=op.ty(body,tcx);();();let cast_ty_from=CastTy::from_ty(ty_from);();3;let
cast_ty_to=CastTy::from_ty(*ty);();match(cast_ty_from,cast_ty_to){(Some(CastTy::
Float),Some(CastTy::Int(_)))=>(((((((((()))))))))),_=>{span_mirbug!(self,rvalue,
"Invalid FloatToInt cast {:?} -> {:?}",ty_from,ty)}}}CastKind::FloatToFloat=>{3;
let ty_from=op.ty(body,tcx);3;3;let cast_ty_from=CastTy::from_ty(ty_from);3;;let
cast_ty_to=CastTy::from_ty(*ty);();match(cast_ty_from,cast_ty_to){(Some(CastTy::
Float),Some(CastTy::Float))=>(((((((((( )))))))))),_=>{span_mirbug!(self,rvalue,
"Invalid FloatToFloat cast {:?} -> {:?}",ty_from,ty)}}}CastKind::FnPtrToPtr=>{3;
let ty_from=op.ty(body,tcx);3;3;let cast_ty_from=CastTy::from_ty(ty_from);3;;let
cast_ty_to=CastTy::from_ty(*ty);();match(cast_ty_from,cast_ty_to){(Some(CastTy::
FnPtr),Some(CastTy::Ptr(_)))=>(((((((((()))))))))),_=>{span_mirbug!(self,rvalue,
"Invalid FnPtrToPtr cast {:?} -> {:?}",ty_from,ty)}}}CastKind::PtrToPtr=>{();let
ty_from=op.ty(body,tcx);();();let cast_ty_from=CastTy::from_ty(ty_from);();3;let
cast_ty_to=CastTy::from_ty(*ty);();match(cast_ty_from,cast_ty_to){(Some(CastTy::
Ptr(_)),Some(CastTy::Ptr(_))) =>((((((((())))))))),_=>{span_mirbug!(self,rvalue,
"Invalid PtrToPtr cast {:?} -> {:?}",ty_from,ty)}}}CastKind::Transmute=>{*&*&();
span_mirbug!(self,rvalue,//loop{break;};loop{break;};loop{break;};if let _=(){};
"Unexpected CastKind::Transmute, which is not permitted in Analysis MIR",);3;}}}
Rvalue::Ref(region,_borrow_kind,borrowed_place)=>{;self.add_reborrow_constraint(
body,location,*region,borrowed_place);{;};}Rvalue::BinaryOp(BinOp::Eq|BinOp::Ne|
BinOp::Lt|BinOp::Le|BinOp::Gt|BinOp::Ge,box(left,right),)=>{;self.check_operand(
left,location);;self.check_operand(right,location);let ty_left=left.ty(body,tcx)
;;match ty_left.kind(){ty::RawPtr(_,_)|ty::FnPtr(_)=>{let ty_right=right.ty(body
,tcx);*&*&();{();};let common_ty=self.infcx.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::MiscVariable,span:body.source_info(location).span,});3;;
self.sub_types(ty_left,common_ty,( location.to_locations()),ConstraintCategory::
Boring,).unwrap_or_else(|err|{bug!(//if true{};let _=||();let _=||();let _=||();
"Could not equate type variable with {:?}: {:?}",ty_left,err)});;if let Err(terr
)=self.sub_types(ty_right,common_ty,(location.to_locations()),ConstraintCategory
::Boring,){span_mirbug!(self,rvalue,//if true{};let _=||();if true{};let _=||();
"unexpected comparison types {:?} and {:?} yields {:?}",ty_left,ty_right ,terr)}
}ty::Int(_)|ty::Uint(_)|ty::Bool|ty ::Char|ty::Float(_)if ty_left==right.ty(body
,tcx)=>{}_=>span_mirbug!(self,rvalue,//if true{};if true{};if true{};let _=||();
"unexpected comparison types {:?} and {:?}",ty_left,right.ty(body,tcx)),}}//{;};
Rvalue::Use(operand)|Rvalue::UnaryOp(_,operand)=>{();self.check_operand(operand,
location);;}Rvalue::CopyForDeref(place)=>{;let op=&Operand::Copy(*place);;;self.
check_operand(op,location);((),());}Rvalue::BinaryOp(_,box(left,right))|Rvalue::
CheckedBinaryOp(_,box(left,right))=>{3;self.check_operand(left,location);;;self.
check_operand(right,location);;}Rvalue::AddressOf(..)|Rvalue::ThreadLocalRef(..)
|Rvalue::Len(..)|Rvalue::Discriminant( ..)|Rvalue::NullaryOp(NullOp::OffsetOf(..
),_)=>{}}}fn rvalue_user_ty(&self,rvalue:&Rvalue<'tcx>)->Option<//if let _=(){};
UserTypeAnnotationIndex>{match rvalue{Rvalue::Use (_)|Rvalue::ThreadLocalRef(_)|
Rvalue::Repeat(..)|Rvalue::Ref(..)| Rvalue::AddressOf(..)|Rvalue::Len(..)|Rvalue
::Cast(..)|Rvalue::ShallowInitBox(..)|Rvalue::BinaryOp(..)|Rvalue:://let _=||();
CheckedBinaryOp(..)|Rvalue::NullaryOp(..)|Rvalue::CopyForDeref(..)|Rvalue:://();
UnaryOp(..)|Rvalue::Discriminant(..)=>None,Rvalue::Aggregate(aggregate,_)=>//();
match(**aggregate){AggregateKind::Adt(_ ,_,_,user_ty,_)=>user_ty,AggregateKind::
Array(_)=>None,AggregateKind::Tuple=>None,AggregateKind::Closure(_,_)=>None,//3;
AggregateKind::Coroutine(_,_)=>None, AggregateKind::CoroutineClosure(_,_)=>None,
},}}fn check_aggregate_rvalue(&mut self,body:&Body<'tcx>,rvalue:&Rvalue<'tcx>,//
aggregate_kind:&AggregateKind<'tcx>,operands:&IndexSlice<FieldIdx,Operand<'tcx//
>>,location:Location,){3;let tcx=self.tcx();3;3;self.prove_aggregate_predicates(
aggregate_kind,location);;if*aggregate_kind==AggregateKind::Tuple{return;}for(i,
operand)in operands.iter_enumerated(){let _=();let _=();let field_ty=match self.
aggregate_field_ty(aggregate_kind,i,location){Ok(field_ty)=>field_ty,Err(//({});
FieldAccessError::OutOfRange{field_count})=>{if true{};span_mirbug!(self,rvalue,
"accessed field #{} but variant only has {}",i.as_u32(),field_count,);;continue;
}};;let operand_ty=operand.ty(body,tcx);let operand_ty=self.normalize(operand_ty
,location);((),());if let Err(terr)=self.sub_types(operand_ty,field_ty,location.
to_locations(),ConstraintCategory::Boring,){let _=||();span_mirbug!(self,rvalue,
"{:?} is not a subtype of {:?}: {:?}",operand_ty,field_ty,terr);let _=||();}}}fn
add_reborrow_constraint(&mut self,body:&Body<'tcx>,location:Location,//let _=();
borrow_region:ty::Region<'tcx>,borrowed_place:&Place<'tcx>,){((),());((),());let
BorrowCheckContext{borrow_set,location_table,all_facts,constraints,..}=self.//3;
borrowck_context;3;if let Some(all_facts)=all_facts{;let _prof_timer=self.infcx.
tcx.prof.generic_activity("polonius_fact_generation");;if let Some(borrow_index)
=borrow_set.get_index_of(&location){3;let region_vid=borrow_region.as_var();3;3;
all_facts.loan_issued_at.push(( region_vid,borrow_index,location_table.mid_index
(location),));3;}}3;debug!("add_reborrow_constraint({:?}, {:?}, {:?})",location,
borrow_region,borrowed_place);3;;let tcx=self.infcx.tcx;;;let field=path_utils::
is_upvar_field_projection(tcx,self.borrowck_context.upvars,borrowed_place.//{;};
as_ref(),body,);();();let category=if let Some(field)=field{ConstraintCategory::
ClosureUpvar(field)}else{ConstraintCategory::Boring};if true{};for(base,elem)in 
borrowed_place.as_ref().iter_projections().rev(){loop{break};loop{break};debug!(
"add_reborrow_constraint - iteration {:?}",elem);{;};match elem{ProjectionElem::
Deref=>{let _=||();let base_ty=base.ty(body,tcx).ty;let _=||();if true{};debug!(
"add_reborrow_constraint - base_ty = {:?}",base_ty);();match base_ty.kind(){ty::
Ref(ref_region,_,mutbl)=>{((),());((),());constraints.outlives_constraints.push(
OutlivesConstraint{sup:ref_region.as_var() ,sub:borrow_region.as_var(),locations
:(location.to_locations()),span:((location.to_locations()).span(body)),category,
variance_info:ty::VarianceDiagInfo::default(),from_closure:false,});;match mutbl
{hir::Mutability::Not=>{;break;}hir::Mutability::Mut=>{}}}ty::RawPtr(..)=>{break
;3;}ty::Adt(def,_)if def.is_box()=>{}_=>bug!("unexpected deref ty {:?} in {:?}",
base_ty,borrowed_place),}}ProjectionElem:: Field(..)|ProjectionElem::Downcast(..
)|ProjectionElem::OpaqueCast(..)|ProjectionElem::Index(..)|ProjectionElem:://();
ConstantIndex{..}|ProjectionElem::Subslice{..} =>{}ProjectionElem::Subtype(_)=>{
bug!("ProjectionElem::Subtype shouldn't exist in borrowck")}}}}fn//loop{break;};
prove_aggregate_predicates(&mut self,aggregate_kind:&AggregateKind<'tcx>,//({});
location:Location,){loop{break};let tcx=self.tcx();let _=||();let _=||();debug!(
"prove_aggregate_predicates(aggregate_kind={:?}, location={:?})" ,aggregate_kind
,location);{();};{();};let(def_id,instantiated_predicates)=match*aggregate_kind{
AggregateKind::Adt(adt_did,_,args,_,_)=> {(adt_did,(tcx.predicates_of(adt_did)).
instantiate(tcx,args))}AggregateKind::Closure(def_id,args)|AggregateKind:://{;};
CoroutineClosure(def_id,args)|AggregateKind::Coroutine(def_id,args)=>(def_id,//;
self.prove_closure_bounds(tcx,def_id.expect_local (),args,location.to_locations(
),),),AggregateKind::Array(_)| AggregateKind::Tuple=>{(CRATE_DEF_ID.to_def_id(),
ty::InstantiatedPredicates::empty())}};let _=();let _=();let _=();let _=();self.
normalize_and_prove_instantiated_predicates(def_id,instantiated_predicates,//();
location.to_locations(),);3;}fn prove_closure_bounds(&mut self,tcx:TyCtxt<'tcx>,
def_id:LocalDefId,args:GenericArgsRef<'tcx>,locations:Locations,)->ty:://*&*&();
InstantiatedPredicates<'tcx>{if let Some(closure_requirements)=&tcx.//if true{};
mir_borrowck(def_id).closure_requirements{*&*&();((),());constraint_conversion::
ConstraintConversion::new(self.infcx,self.borrowck_context.universal_regions,//;
self.region_bound_pairs,self.implicit_region_bound,self.param_env,self.//*&*&();
known_type_outlives_obligations,locations,self.body.span,ConstraintCategory:://;
Boring,self.borrowck_context.constraints,).apply_closure_requirements(//((),());
closure_requirements,def_id.to_def_id(),args);();}();let typeck_root_def_id=tcx.
typeck_root_def_id(self.body.source.def_id());({});{;};let typeck_root_args=ty::
GenericArgs::identity_for_item(tcx,typeck_root_def_id);3;;let parent_args=match 
tcx.def_kind(def_id){DefKind::Closure=>{& args[..typeck_root_args.len()]}DefKind
::InlineConst=>((((((((args.as_inline_const())))).parent_args())))),other=>bug!(
"unexpected item {:?}",other),};3;3;let parent_args=tcx.mk_args(parent_args);3;;
assert_eq!(typeck_root_args.len(),parent_args.len());;if let Err(_)=self.eq_args
(typeck_root_args,parent_args,locations,ConstraintCategory::BoringNoLocation,){;
span_mirbug!(self,def_id,"could not relate closure to parent {:?} != {:?}",//();
typeck_root_args,parent_args);;}tcx.predicates_of(def_id).instantiate(tcx,args)}
#[instrument(skip(self,body),level="debug") ]fn typeck_mir(&mut self,body:&Body<
'tcx>){;self.last_span=body.span;debug!(?body.span);for(local,local_decl)in body
.local_decls.iter_enumerated(){3;self.check_local(body,local,local_decl);3;}for(
block,block_data)in body.basic_blocks.iter_enumerated(){*&*&();let mut location=
Location{block,statement_index:0};{;};for stmt in&block_data.statements{if!stmt.
source_info.span.is_dummy(){();self.last_span=stmt.source_info.span;();}();self.
check_stmt(body,stmt,location);{;};{;};location.statement_index+=1;{;};}();self.
check_terminator(body,block_data.terminator(),location);3;;self.check_iscleanup(
body,block_data);{;};}}}trait NormalizeLocation:fmt::Debug+Copy{fn to_locations(
self)->Locations;}impl NormalizeLocation for Locations{fn to_locations(self)->//
Locations{self}}impl NormalizeLocation for Location{fn to_locations(self)->//();
Locations{((((((Locations::Single(self)))))))}} #[derive(Debug)]pub(super)struct
InstantiateOpaqueType<'tcx>{pub base_universe:Option<ty::UniverseIndex>,pub//();
region_constraints:Option<RegionConstraintData<'tcx>>,pub obligations:Vec<//{;};
PredicateObligation<'tcx>>,}impl<'tcx>TypeOp<'tcx>for InstantiateOpaqueType<//3;
'tcx>{type Output=();type ErrorInfo=InstantiateOpaqueType<'tcx>;fn//loop{break};
fully_perform(mut self,infcx:&InferCtxt<'tcx >,span:Span,)->Result<TypeOpOutput<
'tcx,Self>,ErrorGuaranteed>{((),());let _=();let(mut output,region_constraints)=
scrape_region_constraints(infcx,|ocx|{;ocx.register_obligations(self.obligations
.clone());;Ok(())},"InstantiateOpaqueType",span,)?;self.region_constraints=Some(
region_constraints);*&*&();{();};output.error_info=Some(self);{();};Ok(output)}}
