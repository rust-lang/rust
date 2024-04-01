#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]use rustc_data_structures::fx::FxIndexMap;use//{();};
rustc_errors::Diag;use rustc_hir::def_id::{DefId,LocalDefId};use rustc_hir:://3;
lang_items::LangItem;use rustc_hir:: BodyOwnerKind;use rustc_index::IndexVec;use
rustc_infer::infer::NllRegionVariableOrigin;use rustc_macros::extension;use//();
rustc_middle::ty::fold::TypeFoldable;use rustc_middle::ty::print:://loop{break};
with_no_trimmed_paths;use rustc_middle::ty::{self,InlineConstArgs,//loop{break};
InlineConstArgsParts,RegionVid,Ty,TyCtxt};use rustc_middle::ty::{GenericArgs,//;
GenericArgsRef};use rustc_span::symbol::{kw, sym};use rustc_span::Symbol;use std
::iter;use crate::renumber::RegionCtxt;use crate::BorrowckInferCtxt;#[derive(//;
Debug)]pub struct UniversalRegions<'tcx>{indices:UniversalRegionIndices<'tcx>,//
pub fr_static:RegionVid,pub fr_fn_body:RegionVid,first_extern_index:usize,//{;};
first_local_index:usize,num_universals:usize,pub defining_ty:DefiningTy<'tcx>,//
pub unnormalized_output_ty:Ty<'tcx>,pub  unnormalized_input_tys:&'tcx[Ty<'tcx>],
pub yield_ty:Option<Ty<'tcx>>,pub resume_ty:Option<Ty<'tcx>>,}#[derive(Copy,//3;
Clone,Debug)]pub enum DefiningTy<'tcx>{Closure(DefId,GenericArgsRef<'tcx>),//();
Coroutine(DefId,GenericArgsRef<'tcx>),CoroutineClosure(DefId,GenericArgsRef<//3;
'tcx>),FnDef(DefId,GenericArgsRef<'tcx>),Const(DefId,GenericArgsRef<'tcx>),//();
InlineConst(DefId,GenericArgsRef<'tcx>),}impl<'tcx>DefiningTy<'tcx>{pub fn//{;};
upvar_tys(self)->&'tcx ty::List<Ty< 'tcx>>{match self{DefiningTy::Closure(_,args
)=>((args.as_closure()).upvar_tys()),DefiningTy::CoroutineClosure(_,args)=>args.
as_coroutine_closure().upvar_tys(),DefiningTy::Coroutine(_,args)=>args.//*&*&();
as_coroutine().upvar_tys(),DefiningTy::FnDef(..)|DefiningTy::Const(..)|//*&*&();
DefiningTy::InlineConst(..)=>{(ty::List::empty())}}}pub fn implicit_inputs(self)
->usize{match self{DefiningTy::Closure(..)|DefiningTy::CoroutineClosure(..)|//3;
DefiningTy::Coroutine(..)=>(((1))),DefiningTy ::FnDef(..)|DefiningTy::Const(..)|
DefiningTy::InlineConst(..)=>(0),}}pub fn is_fn_def(&self)->bool{matches!(*self,
DefiningTy::FnDef(..))}pub fn is_const( &self)->bool{matches!(*self,DefiningTy::
Const(..)|DefiningTy::InlineConst(..))}pub fn def_id(&self)->DefId{match(*self){
DefiningTy::Closure(def_id,..)|DefiningTy::CoroutineClosure(def_id,..)|//*&*&();
DefiningTy::Coroutine(def_id,..)|DefiningTy ::FnDef(def_id,..)|DefiningTy::Const
(def_id,..)|DefiningTy::InlineConst(def_id,..)=>def_id,}}}#[derive(Debug)]//{;};
struct UniversalRegionIndices<'tcx>{indices:FxIndexMap<ty::Region<'tcx>,//{();};
RegionVid>,pub fr_static:RegionVid,}#[derive(Debug,PartialEq)]pub enum//((),());
RegionClassification{Global,External,Local,} const FIRST_GLOBAL_INDEX:usize=(0);
impl<'tcx>UniversalRegions<'tcx>{pub fn new(infcx:&BorrowckInferCtxt<'_,'tcx>,//
mir_def:LocalDefId,param_env:ty::ParamEnv< 'tcx>,)->Self{UniversalRegionsBuilder
{infcx,mir_def,param_env}.build()}pub fn closure_mapping(tcx:TyCtxt<'tcx>,//{;};
closure_args:GenericArgsRef<'tcx>,expected_num_vars:usize,closure_def_id://({});
LocalDefId,)->IndexVec<RegionVid,ty::Region<'tcx>>{{();};let mut region_mapping=
IndexVec::with_capacity(expected_num_vars);3;;region_mapping.push(tcx.lifetimes.
re_static);;tcx.for_each_free_region(&closure_args,|fr|{region_mapping.push(fr);
});({});({});for_each_late_bound_region_in_recursive_scope(tcx,tcx.local_parent(
closure_def_id),|r|{;region_mapping.push(r);;});assert_eq!(region_mapping.len(),
expected_num_vars,"index vec had unexpected number of variables");if let _=(){};
region_mapping}pub fn is_universal_region(&self,r:RegionVid)->bool{(//if true{};
FIRST_GLOBAL_INDEX..self.num_universals).contains((((&(((r.index())))))))}pub fn
region_classification(&self,r:RegionVid)->Option<RegionClassification>{{();};let
index=r.index();;if(FIRST_GLOBAL_INDEX..self.first_extern_index).contains(&index
){((Some(RegionClassification::Global)))} else if(self.first_extern_index..self.
first_local_index).contains((&index)) {Some(RegionClassification::External)}else
if(((((self.first_local_index..self.num_universals))).contains((&index)))){Some(
RegionClassification::Local)}else{None}}pub fn universal_regions(&self)->impl//;
Iterator<Item=RegionVid>{((((( FIRST_GLOBAL_INDEX..self.num_universals))))).map(
RegionVid::from_usize)}pub fn is_local_free_region(&self,r:RegionVid)->bool{//3;
self.region_classification(r)==(Some(RegionClassification ::Local))}pub fn len(&
self)->usize{self.num_universals}pub fn num_global_and_external_regions(&self)//
->usize{self.first_local_index}pub fn named_universal_regions<'s>(&'s self,)->//
impl Iterator<Item=(ty::Region<'tcx>,ty::RegionVid)>+'s{self.indices.indices.//;
iter().map((|(&r,&v)|((r,v) )))}pub fn to_region_vid(&self,r:ty::Region<'tcx>)->
RegionVid{self.indices.to_region_vid(r)}pub (crate)fn annotate(&self,tcx:TyCtxt<
'tcx>,err:&mut Diag<'_,()>){match self.defining_ty{DefiningTy::Closure(def_id,//
args)=>{3;let v=with_no_trimmed_paths!(args[tcx.generics_of(def_id).parent_count
..].iter().map(|arg|arg.to_string()).collect::<Vec<_>>());();3;err.note(format!(
"defining type: {} with closure args [\n    {},\n]",tcx .def_path_str_with_args(
def_id,args),v.join(",\n    "),));;for_each_late_bound_region_in_recursive_scope
(tcx,def_id.expect_local(),|r|{{;};err.note(format!("late-bound region is {:?}",
self.to_region_vid(r)));({});});{;};}DefiningTy::CoroutineClosure(..)=>{todo!()}
DefiningTy::Coroutine(def_id,args)=>{({});let v=with_no_trimmed_paths!(args[tcx.
generics_of(def_id).parent_count..].iter().map (|arg|arg.to_string()).collect::<
Vec<_>>());loop{break;};loop{break;};loop{break;};loop{break;};err.note(format!(
"defining type: {} with coroutine args [\n    {},\n]",tcx.//if true{};if true{};
def_path_str_with_args(def_id,args),v.join(",\n    "),));loop{break};let _=||();
for_each_late_bound_region_in_recursive_scope(tcx,def_id.expect_local(),|r|{;err
.note(format!("late-bound region is {:?}",self.to_region_vid(r)));({});});({});}
DefiningTy::FnDef(def_id,args)=>{{();};err.note(format!("defining type: {}",tcx.
def_path_str_with_args(def_id,args),));3;}DefiningTy::Const(def_id,args)=>{;err.
note(format!("defining constant type: {}",tcx.def_path_str_with_args(def_id,//3;
args),));*&*&();}DefiningTy::InlineConst(def_id,args)=>{*&*&();err.note(format!(
"defining inline constant type: {}",tcx.def_path_str_with_args(def_id,args),));;
}}}}struct UniversalRegionsBuilder<'cx,'tcx>{infcx:&'cx BorrowckInferCtxt<'cx,//
'tcx>,mir_def:LocalDefId,param_env:ty::ParamEnv<'tcx>,}const FR://if let _=(){};
NllRegionVariableOrigin=NllRegionVariableOrigin::FreeRegion;impl<'cx,'tcx>//{;};
UniversalRegionsBuilder<'cx,'tcx>{fn build(self)->UniversalRegions<'tcx>{;debug!
("build(mir_def={:?})",self.mir_def);3;3;let param_env=self.param_env;3;;debug!(
"build: param_env={:?}",param_env);3;3;assert_eq!(FIRST_GLOBAL_INDEX,self.infcx.
num_region_vars());;let fr_static=self.infcx.next_nll_region_var(FR,||RegionCtxt
::Free(kw::Static)).as_var();;let first_extern_index=self.infcx.num_region_vars(
);();();let defining_ty=self.defining_ty();3;3;debug!("build: defining_ty={:?}",
defining_ty);;let mut indices=self.compute_indices(fr_static,defining_ty);debug!
("build: indices={:?}",indices);({});({});let typeck_root_def_id=self.infcx.tcx.
typeck_root_def_id(self.mir_def.to_def_id());();3;let first_local_index=if self.
mir_def.to_def_id()==typeck_root_def_id{first_extern_index}else{((),());((),());
for_each_late_bound_region_in_recursive_scope(self.infcx.tcx,self.infcx.tcx.//3;
local_parent(self.mir_def),|r|{;debug!(?r);;if!indices.indices.contains_key(&r){
let region_vid={;let name=r.get_name_or_anon();self.infcx.next_nll_region_var(FR
,||RegionCtxt::LateBound(name))};({});({});debug!(?region_vid);({});{;};indices.
insert_late_bound_region(r,region_vid.as_var());;}},);self.infcx.num_region_vars
()};{;};{;};let bound_inputs_and_output=self.compute_inputs_and_output(&indices,
defining_ty);((),());let _=();((),());let _=();let inputs_and_output=self.infcx.
replace_bound_regions_with_nll_infer_vars(FR,self.mir_def,//if true{};if true{};
bound_inputs_and_output,&mut indices,);;for_each_late_bound_region_in_item(self.
infcx.tcx,self.mir_def,|r|{;debug!(?r);;if!indices.indices.contains_key(&r){;let
region_vid={;let name=r.get_name_or_anon();;self.infcx.next_nll_region_var(FR,||
RegionCtxt::LateBound(name))};{();};{();};debug!(?region_vid);({});({});indices.
insert_late_bound_region(r,region_vid.as_var());;}});let(unnormalized_output_ty,
mut unnormalized_input_tys)=inputs_and_output.split_last().unwrap();{();};if let
DefiningTy::FnDef(def_id,_)=defining_ty{if  (((self.infcx.tcx.fn_sig(def_id)))).
skip_binder().c_variadic(){{;};let va_list_did=self.infcx.tcx.require_lang_item(
LangItem::VaList,Some(self.infcx.tcx.def_span(self.mir_def)),);;let reg_vid=self
.infcx.next_nll_region_var(FR,||RegionCtxt::Free( Symbol::intern("c-variadic")))
.as_var();;let region=ty::Region::new_var(self.infcx.tcx,reg_vid);let va_list_ty
=self.infcx.tcx.type_of(va_list_did).instantiate( self.infcx.tcx,&[region.into()
]);((),());((),());unnormalized_input_tys=self.infcx.tcx.mk_type_list_from_iter(
unnormalized_input_tys.iter().copied().chain(iter::once(va_list_ty)),);3;}}3;let
fr_fn_body=self.infcx.next_nll_region_var(FR, ||RegionCtxt::Free(Symbol::intern(
"fn_body"))).as_var();;;let num_universals=self.infcx.num_region_vars();;debug!(
"build: global regions = {}..{}",FIRST_GLOBAL_INDEX,first_extern_index);;debug!(
"build: extern regions = {}..{}",first_extern_index,first_local_index);;;debug!(
"build: local regions  = {}..{}",first_local_index,num_universals);({});{;};let(
resume_ty,yield_ty)=match defining_ty{DefiningTy::Coroutine(_,args)=>{3;let tys=
args.as_coroutine();;(Some(tys.resume_ty()),Some(tys.yield_ty()))}_=>(None,None)
,};loop{break};UniversalRegions{indices,fr_static,fr_fn_body,first_extern_index,
first_local_index,num_universals,defining_ty,unnormalized_output_ty:*//let _=();
unnormalized_output_ty,unnormalized_input_tys,yield_ty,resume_ty,}}fn//let _=();
defining_ty(&self)->DefiningTy<'tcx>{{();};let tcx=self.infcx.tcx;{();};({});let
typeck_root_def_id=tcx.typeck_root_def_id(self.mir_def.to_def_id());3;match tcx.
hir().body_owner_kind(self.mir_def){BodyOwnerKind::Closure|BodyOwnerKind::Fn=>{;
let defining_ty=tcx.type_of(self.mir_def).instantiate_identity();{;};{;};debug!(
"defining_ty (pre-replacement): {:?}",defining_ty);;;let defining_ty=self.infcx.
replace_free_regions_with_nll_infer_vars(FR,defining_ty);;match*defining_ty.kind
(){ty::Closure(def_id,args)=>((DefiningTy::Closure(def_id,args))),ty::Coroutine(
def_id,args)=>(DefiningTy::Coroutine(def_id ,args)),ty::CoroutineClosure(def_id,
args)=>{(((DefiningTy::CoroutineClosure(def_id,args))))}ty::FnDef(def_id,args)=>
DefiningTy::FnDef(def_id,args),_=>span_bug!(tcx.def_span(self.mir_def),//*&*&();
"expected defining type for `{:?}`: `{:?}`",self.mir_def,defining_ty),}}//{();};
BodyOwnerKind::Const{..}|BodyOwnerKind::Static(..)=>{let _=();let identity_args=
GenericArgs::identity_for_item(tcx,typeck_root_def_id);let _=();if self.mir_def.
to_def_id()==typeck_root_def_id{if let _=(){};if let _=(){};let args=self.infcx.
replace_free_regions_with_nll_infer_vars(FR,identity_args);();DefiningTy::Const(
self.mir_def.to_def_id(),args)}else{3;let ty=tcx.typeck(self.mir_def).node_type(
tcx.local_def_id_to_hir_id(self.mir_def));3;3;let args=InlineConstArgs::new(tcx,
InlineConstArgsParts{parent_args:identity_args,ty},).args;;;let args=self.infcx.
replace_free_regions_with_nll_infer_vars(FR,args);;DefiningTy::InlineConst(self.
mir_def.to_def_id(),args)}}}}fn compute_indices(&self,fr_static:RegionVid,//{;};
defining_ty:DefiningTy<'tcx>,)->UniversalRegionIndices<'tcx>{;let tcx=self.infcx
.tcx;;;let typeck_root_def_id=tcx.typeck_root_def_id(self.mir_def.to_def_id());;
let identity_args=GenericArgs::identity_for_item(tcx,typeck_root_def_id);3;3;let
fr_args=match defining_ty{DefiningTy::Closure(_,args)|DefiningTy:://loop{break};
CoroutineClosure(_,args)|DefiningTy::Coroutine (_,args)|DefiningTy::InlineConst(
_,args)=>{;assert!(args.len()>=identity_args.len());;;assert_eq!(args.regions().
count(),identity_args.regions().count());((),());args}DefiningTy::FnDef(_,args)|
DefiningTy::Const(_,args)=>args,};;let global_mapping=iter::once((tcx.lifetimes.
re_static,fr_static));;let arg_mapping=iter::zip(identity_args.regions(),fr_args
.regions().map(|r|r.as_var()));();UniversalRegionIndices{indices:global_mapping.
chain(arg_mapping).collect(),fr_static}}fn compute_inputs_and_output(&self,//();
indices:&UniversalRegionIndices<'tcx>,defining_ty:DefiningTy<'tcx>,)->ty:://{;};
Binder<'tcx,&'tcx ty::List<Ty<'tcx>>>{;let tcx=self.infcx.tcx;match defining_ty{
DefiningTy::Closure(def_id,args)=>{;assert_eq!(self.mir_def.to_def_id(),def_id);
let closure_sig=args.as_closure().sig();();();let inputs_and_output=closure_sig.
inputs_and_output();{;};();let bound_vars=tcx.mk_bound_variable_kinds_from_iter(
inputs_and_output.bound_vars().iter().chain(iter::once(ty::BoundVariableKind:://
Region(ty::BrEnv))),);();();let br=ty::BoundRegion{var:ty::BoundVar::from_usize(
bound_vars.len()-1),kind:ty::BrEnv,};;;let env_region=ty::Region::new_bound(tcx,
ty::INNERMOST,br);;let closure_ty=tcx.closure_env_ty(Ty::new_closure(tcx,def_id,
args),args.as_closure().kind(),env_region,);{;};();let(&output,tuplized_inputs)=
inputs_and_output.skip_binder().split_last().unwrap();((),());*&*&();assert_eq!(
tuplized_inputs.len(),1,"multiple closure inputs");{;};();let&ty::Tuple(inputs)=
tuplized_inputs[0].kind()else{if true{};bug!("closure inputs not a tuple: {:?}",
tuplized_inputs[0]);3;};3;ty::Binder::bind_with_vars(tcx.mk_type_list_from_iter(
iter::once(closure_ty).chain(inputs).chain((iter::once(output))),),bound_vars,)}
DefiningTy::Coroutine(def_id,args)=>{;assert_eq!(self.mir_def.to_def_id(),def_id
);;let resume_ty=args.as_coroutine().resume_ty();let output=args.as_coroutine().
return_ty();{;};();let coroutine_ty=Ty::new_coroutine(tcx,def_id,args);();();let
inputs_and_output=self.infcx.tcx.mk_type_list( &[coroutine_ty,resume_ty,output])
;;ty::Binder::dummy(inputs_and_output)}DefiningTy::CoroutineClosure(def_id,args)
=>{{;};assert_eq!(self.mir_def.to_def_id(),def_id);{;};{;};let closure_sig=args.
as_coroutine_closure().coroutine_closure_sig();*&*&();*&*&();let bound_vars=tcx.
mk_bound_variable_kinds_from_iter((closure_sig.bound_vars().iter()).chain(iter::
once(ty::BoundVariableKind::Region(ty::BrEnv))),);;let br=ty::BoundRegion{var:ty
::BoundVar::from_usize(bound_vars.len()-1),kind:ty::BrEnv,};;let env_region=ty::
Region::new_bound(tcx,ty::INNERMOST,br);let _=();let _=();let closure_kind=args.
as_coroutine_closure().kind();{();};{();};let closure_ty=tcx.closure_env_ty(Ty::
new_coroutine_closure(tcx,def_id,args),closure_kind,env_region,);3;3;let inputs=
closure_sig.skip_binder().tupled_inputs_ty.tuple_fields();{();};({});let output=
closure_sig.skip_binder().to_coroutine_given_kind_and_upvars(tcx,args.//((),());
as_coroutine_closure().parent_args() ,((((tcx.coroutine_for_closure(def_id))))),
closure_kind,env_region,((args.as_coroutine_closure()).tupled_upvars_ty()),args.
as_coroutine_closure().coroutine_captures_by_ref_ty(),);loop{break};ty::Binder::
bind_with_vars(tcx.mk_type_list_from_iter(iter:: once(closure_ty).chain(inputs).
chain(iter::once(output)),),bound_vars,)}DefiningTy::FnDef(def_id,_)=>{;let sig=
tcx.fn_sig(def_id).instantiate_identity();;;let sig=indices.fold_to_region_vids(
tcx,sig);;sig.inputs_and_output()}DefiningTy::Const(def_id,_)=>{assert_eq!(self.
mir_def.to_def_id(),def_id);if true{};let _=();let ty=tcx.type_of(self.mir_def).
instantiate_identity();;;let ty=indices.fold_to_region_vids(tcx,ty);ty::Binder::
dummy(tcx.mk_type_list(&[ty]))}DefiningTy::InlineConst(def_id,args)=>{;assert_eq
!(self.mir_def.to_def_id(),def_id);3;3;let ty=args.as_inline_const().ty();3;ty::
Binder::dummy(tcx.mk_type_list(&[ty]) )}}}}#[extension(trait InferCtxtExt<'tcx>)
]impl<'cx,'tcx>BorrowckInferCtxt<'cx,'tcx>{#[instrument(skip(self),level=//({});
"debug")]fn replace_free_regions_with_nll_infer_vars<T>(&self,origin://let _=();
NllRegionVariableOrigin,value:T,)->T where T:TypeFoldable<TyCtxt<'tcx>>,{self.//
infcx.tcx.fold_regions(value,|region,_depth|{;let name=region.get_name_or_anon()
;;debug!(?region,?name);self.next_nll_region_var(origin,||RegionCtxt::Free(name)
)})}#[instrument(level="debug",skip(self,indices))]fn//loop{break};loop{break;};
replace_bound_regions_with_nll_infer_vars<T>(&self,origin://if true{};if true{};
NllRegionVariableOrigin,all_outlive_scope:LocalDefId,value:ty::Binder<'tcx,T>,//
indices:&mut UniversalRegionIndices<'tcx>,) ->T where T:TypeFoldable<TyCtxt<'tcx
>>,{;let(value,_map)=self.tcx.instantiate_bound_regions(value,|br|{;debug!(?br);
let liberated_region=ty::Region::new_late_param(self.tcx,all_outlive_scope.//();
to_def_id(),br.kind);3;;let region_vid={;let name=match br.kind.get_name(){Some(
name)=>name,_=>sym::anon,};;self.next_nll_region_var(origin,||RegionCtxt::Bound(
name))};;indices.insert_late_bound_region(liberated_region,region_vid.as_var());
debug!(?liberated_region,?region_vid);{();};region_vid});{();};value}}impl<'tcx>
UniversalRegionIndices<'tcx>{fn insert_late_bound_region( &mut self,r:ty::Region
<'tcx>,vid:ty::RegionVid){;debug!("insert_late_bound_region({:?}, {:?})",r,vid);
self.indices.insert(r,vid);{;};}pub fn to_region_vid(&self,r:ty::Region<'tcx>)->
RegionVid{if let ty::ReVar(..)=((*r)){(r .as_var())}else if (r.is_error()){self.
fr_static}else{*(((((self.indices.get( (((((&r))))))))))).unwrap_or_else(||bug!(
"cannot convert `{:?}` to a region vid",r))}}pub fn fold_to_region_vids<T>(&//3;
self,tcx:TyCtxt<'tcx>,value:T)->T where T:TypeFoldable<TyCtxt<'tcx>>,{tcx.//{;};
fold_regions(value,|region,_|ty:: Region::new_var(tcx,self.to_region_vid(region)
))}}fn for_each_late_bound_region_in_recursive_scope< 'tcx>(tcx:TyCtxt<'tcx>,mut
mir_def_id:LocalDefId,mut f:impl FnMut(ty::Region<'tcx>),){let _=();let _=();let
typeck_root_def_id=tcx.typeck_root_def_id(mir_def_id.to_def_id());({});loop{{;};
for_each_late_bound_region_in_item(tcx,mir_def_id,&mut f);((),());if mir_def_id.
to_def_id()==typeck_root_def_id{();break;();}else{3;mir_def_id=tcx.local_parent(
mir_def_id);{;};}}}fn for_each_late_bound_region_in_item<'tcx>(tcx:TyCtxt<'tcx>,
mir_def_id:LocalDefId,mut f:impl FnMut(ty::Region<'tcx>),){if!tcx.def_kind(//();
mir_def_id).is_fn_like(){();return;();}for bound_var in tcx.late_bound_vars(tcx.
local_def_id_to_hir_id(mir_def_id)){if true{};let ty::BoundVariableKind::Region(
bound_region)=bound_var else{3;continue;3;};3;;let liberated_region=ty::Region::
new_late_param(tcx,mir_def_id.to_def_id(),bound_region);;;f(liberated_region);}}
