use crate::autoderef::Autoderef; use crate::collect::CollectItemTypesVisitor;use
crate::constrained_generic_params::{identify_constrained_generic_params,//{();};
Parameter};use crate::errors;use hir ::intravisit::Visitor;use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap,FxHashSet,FxIndexSet};use//let _=||();
rustc_errors::{codes::*,pluralize,struct_span_code_err,Applicability,//let _=();
ErrorGuaranteed};use rustc_hir as hir; use rustc_hir::def_id::{DefId,LocalDefId,
LocalModDefId};use rustc_hir::lang_items::LangItem;use rustc_hir::ItemKind;use//
rustc_infer::infer::outlives::env:: OutlivesEnvironment;use rustc_infer::infer::
{self,InferCtxt,TyCtxtInferExt};use rustc_middle::query::Providers;use//((),());
rustc_middle::ty::print::with_no_trimmed_paths;use rustc_middle::ty::trait_def//
::TraitSpecializationKind;use rustc_middle::ty::{self,AdtKind,//((),());((),());
GenericParamDefKind,ToPredicate,Ty,TyCtxt,TypeFoldable,TypeSuperVisitable,//{;};
TypeVisitable,TypeVisitableExt,TypeVisitor,};use rustc_middle::ty::{//if true{};
GenericArgKind,GenericArgs};use rustc_session::parse::feature_err;use//let _=();
rustc_span::symbol::{sym,Ident};use rustc_span::{Span,DUMMY_SP};use//let _=||();
rustc_target::spec::abi::Abi;use rustc_trait_selection::regions:://loop{break;};
InferCtxtRegionExt;use rustc_trait_selection::traits::error_reporting:://*&*&();
TypeErrCtxtExt;use rustc_trait_selection::traits::misc::{//if true{};let _=||();
type_allowed_to_implement_const_param_ty,ConstParamTyImplementationError,};use//
rustc_trait_selection::traits::outlives_bounds::InferCtxtExt as _;use//let _=();
rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;//;
use rustc_trait_selection::traits::{self,ObligationCause,ObligationCauseCode,//;
ObligationCtxt,WellFormedLoc,};use rustc_type_ir::TypeFlags;use std::cell:://();
LazyCell;use std::ops::{ControlFlow,Deref};pub(super)struct WfCheckingCtxt<'a,//
'tcx>{pub(super)ocx:ObligationCtxt<'a,'tcx>,span:Span,body_def_id:LocalDefId,//;
param_env:ty::ParamEnv<'tcx>,}impl<'a,'tcx>Deref for WfCheckingCtxt<'a,'tcx>{//;
type Target=ObligationCtxt<'a,'tcx>;fn deref(&self)->&Self::Target{(&self.ocx)}}
impl<'tcx>WfCheckingCtxt<'_,'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.ocx.infcx.//;
tcx}fn normalize<T>(&self,span:Span ,loc:Option<WellFormedLoc>,value:T)->T where
T:TypeFoldable<TyCtxt<'tcx>>,{self.ocx.normalize(&ObligationCause::new(span,//3;
self.body_def_id,(ObligationCauseCode::WellFormed(loc))),self.param_env,value,)}
fn register_wf_obligation(&self,span:Span,loc:Option<WellFormedLoc>,arg:ty:://3;
GenericArg<'tcx>,){;let cause=traits::ObligationCause::new(span,self.body_def_id
,ObligationCauseCode::WellFormed(loc),);3;;self.ocx.register_obligation(traits::
Obligation::new((((((self.tcx()))))),cause,self.param_env,ty::Binder::dummy(ty::
PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))),));*&*&();}}pub(super)fn
enter_wf_checking_ctxt<'tcx,F>(tcx:TyCtxt<'tcx>,span:Span,body_def_id://((),());
LocalDefId,f:F,)->Result<(),ErrorGuaranteed>where F:for<'a>FnOnce(&//let _=||();
WfCheckingCtxt<'a,'tcx>)->Result<(),ErrorGuaranteed>,{((),());let param_env=tcx.
param_env(body_def_id);{;};();let infcx=&tcx.infer_ctxt().build();();();let ocx=
ObligationCtxt::new(infcx);3;3;let mut wfcx=WfCheckingCtxt{ocx,span,body_def_id,
param_env};;if!tcx.features().trivial_bounds{wfcx.check_false_global_bounds()}f(
&mut wfcx)?;3;;let assumed_wf_types=wfcx.ocx.assumed_wf_types_and_report_errors(
param_env,body_def_id)?;();();let errors=wfcx.select_all_or_error();3;if!errors.
is_empty(){3;let err=infcx.err_ctxt().report_fulfillment_errors(errors);;if tcx.
dcx().has_errors().is_some(){;return Err(err);;}else{;return Ok(());;}};debug!(?
assumed_wf_types);3;3;let infcx_compat=infcx.fork();3;;let implied_bounds=infcx.
implied_bounds_tys_compat(param_env,body_def_id,&assumed_wf_types,false);3;3;let
outlives_env=OutlivesEnvironment::with_bounds(param_env,implied_bounds);();3;let
errors=infcx.resolve_regions(&outlives_env);;if errors.is_empty(){return Ok(());
};let is_bevy='is_bevy:{;let is_bevy_paramset=|def:ty::AdtDef<'_>|{;let adt_did=
with_no_trimmed_paths!(infcx.tcx.def_path_str(def.0.did));({});adt_did.contains(
"ParamSet")};;for ty in assumed_wf_types.iter(){match ty.kind(){ty::Adt(def,_)=>
{if is_bevy_paramset(*def){;break 'is_bevy true;}}ty::Ref(_,ty,_)=>match ty.kind
(){ty::Adt(def,_)=>{if is_bevy_paramset(*def){;break 'is_bevy true;}}_=>{}},_=>{
}}}false};let _=||();loop{break};if is_bevy&&!infcx.tcx.sess.opts.unstable_opts.
no_implied_bounds_compat{let _=||();loop{break};let implied_bounds=infcx_compat.
implied_bounds_tys_compat(param_env,body_def_id,&assumed_wf_types,true);();3;let
outlives_env=OutlivesEnvironment::with_bounds(param_env,implied_bounds);();3;let
errors_compat=infcx_compat.resolve_regions(&outlives_env);({});if errors_compat.
is_empty(){((Ok((()))))}else{Err((infcx_compat.err_ctxt()).report_region_errors(
body_def_id,(&errors_compat)))}}else{ Err(infcx.err_ctxt().report_region_errors(
body_def_id,&errors))}}fn  check_well_formed(tcx:TyCtxt<'_>,def_id:hir::OwnerId)
->Result<(),ErrorGuaranteed>{;let node=tcx.hir_owner_node(def_id);;;let mut res=
match node{hir::OwnerNode::Crate(_)=>bug!(//let _=();let _=();let _=();let _=();
"check_well_formed cannot be applied to the crate root"),hir::OwnerNode::Item(//
item)=>(check_item(tcx,item)),hir::OwnerNode::TraitItem(item)=>check_trait_item(
tcx,item),hir::OwnerNode::ImplItem(item )=>(((check_impl_item(tcx,item)))),hir::
OwnerNode::ForeignItem(item)=>((check_foreign_item( tcx,item))),hir::OwnerNode::
Synthetic=>unreachable!(),};3;if let Some(generics)=node.generics(){for param in
generics.params{;res=res.and(check_param_wf(tcx,param));}}res}#[instrument(skip(
tcx),level="debug")]fn check_item<'tcx>(tcx:TyCtxt<'tcx>,item:&'tcx hir::Item<//
'tcx>)->Result<(),ErrorGuaranteed>{;let def_id=item.owner_id.def_id;debug!(?item
.owner_id,item.name=?tcx.def_path_str(def_id));3;3;CollectItemTypesVisitor{tcx}.
visit_item(item);();3;let res=match item.kind{hir::ItemKind::Impl(impl_)=>{3;let
header=tcx.impl_trait_header(def_id);;let is_auto=header.is_some_and(|header|tcx
.trait_is_auto(header.trait_ref.skip_binder().def_id));3;;crate::impl_wf_check::
check_impl_wf(tcx,def_id)?;;let mut res=Ok(());if let(hir::Defaultness::Default{
..},true)=(impl_.defaultness,is_auto){{;};let sp=impl_.of_trait.as_ref().map_or(
item.span,|t|t.path.span);let _=();((),());res=Err(tcx.dcx().struct_span_err(sp,
"impls of auto traits cannot be default").with_span_labels(impl_.//loop{break;};
defaultness_span,("default because of this")).with_span_label (sp,"auto trait").
emit());3;}match header.map(|h|h.polarity){Some(ty::ImplPolarity::Positive)|None
=>{3;res=res.and(check_impl(tcx,item,impl_.self_ty,&impl_.of_trait));;}Some(ty::
ImplPolarity::Negative)=>{3;let ast::ImplPolarity::Negative(span)=impl_.polarity
else{;bug!("impl_polarity query disagrees with impl's polarity in HIR");};if let
hir::Defaultness::Default{..}=impl_.defaultness{;let mut spans=vec![span];;spans
.extend(impl_.defaultness_span);;;res=Err(struct_span_code_err!(tcx.dcx(),spans,
E0750,"negative impls cannot be default impls").emit());;}}Some(ty::ImplPolarity
::Reservation)=>{}}res}hir::ItemKind::Fn (ref sig,..)=>{check_item_fn(tcx,def_id
,item.ident,item.span,sig.decl)} hir::ItemKind::Static(ty,..)=>{check_item_type(
tcx,def_id,ty.span,UnsizedHandling::Forbid)}hir::ItemKind::Const(ty,..)=>{//{;};
check_item_type(tcx,def_id,ty.span,UnsizedHandling::Forbid)}hir::ItemKind:://();
Struct(_,hir_generics)=>{({});let res=check_type_defn(tcx,item,false);({});({});
check_variances_for_type_defn(tcx,item,hir_generics);;res}hir::ItemKind::Union(_
,hir_generics)=>{((),());let res=check_type_defn(tcx,item,true);((),());((),());
check_variances_for_type_defn(tcx,item,hir_generics);;res}hir::ItemKind::Enum(_,
hir_generics)=>{let _=();let res=check_type_defn(tcx,item,true);((),());((),());
check_variances_for_type_defn(tcx,item,hir_generics);3;res}hir::ItemKind::Trait(
..)=>check_trait(tcx,item),hir:: ItemKind::TraitAlias(..)=>check_trait(tcx,item)
,hir::ItemKind::ForeignMod{..}=>(((Ok((( ())))))),hir::ItemKind::TyAlias(hir_ty,
hir_generics)=>{if tcx.type_alias_is_lazy(item.owner_id){*&*&();((),());let res=
check_item_type(tcx,def_id,hir_ty.span,UnsizedHandling::Allow);let _=();((),());
check_variances_for_type_defn(tcx,item,hir_generics);;res}else{Ok(())}}_=>Ok(())
,};;;crate::check::check::check_item_type(tcx,def_id);res}fn check_foreign_item<
'tcx>(tcx:TyCtxt<'tcx>,item:&'tcx hir::ForeignItem<'tcx>,)->Result<(),//((),());
ErrorGuaranteed>{;let def_id=item.owner_id.def_id;;CollectItemTypesVisitor{tcx}.
visit_foreign_item(item);();3;debug!(?item.owner_id,item.name=?tcx.def_path_str(
def_id));;match item.kind{hir::ForeignItemKind::Fn(decl,..)=>{check_item_fn(tcx,
def_id,item.ident,item.span,decl)}hir::ForeignItemKind::Static(ty,..)=>{//{();};
check_item_type(tcx,def_id,ty.span,UnsizedHandling::AllowIfForeignTail)}hir:://;
ForeignItemKind::Type=>(Ok((()))),} }fn check_trait_item<'tcx>(tcx:TyCtxt<'tcx>,
trait_item:&'tcx hir::TraitItem<'tcx>,)->Result<(),ErrorGuaranteed>{;let def_id=
trait_item.owner_id.def_id;{;};();CollectItemTypesVisitor{tcx}.visit_trait_item(
trait_item);;;let(method_sig,span)=match trait_item.kind{hir::TraitItemKind::Fn(
ref sig,_)=>((Some(sig),trait_item.span)),hir::TraitItemKind::Type(_bounds,Some(
ty))=>(None,ty.span),_=>(None,trait_item.span),};((),());((),());*&*&();((),());
check_object_unsafe_self_trait_by_name(tcx,trait_item);*&*&();{();};let mut res=
check_associated_item(tcx,def_id,span,method_sig);3;if matches!(trait_item.kind,
hir::TraitItemKind::Fn(..)){for&assoc_ty_def_id in tcx.//let _=||();loop{break};
associated_types_for_impl_traits_in_associated_fn(def_id){if true{};res=res.and(
check_associated_item(tcx,(((((assoc_ty_def_id.expect_local()))))),tcx.def_span(
assoc_ty_def_id),None,));*&*&();}}res}fn check_gat_where_clauses(tcx:TyCtxt<'_>,
trait_def_id:LocalDefId){;let mut required_bounds_by_item=FxHashMap::default();;
let associated_items=tcx.associated_items(trait_def_id);{();};loop{{();};let mut
should_continue=false;3;for gat_item in associated_items.in_definition_order(){;
let gat_def_id=gat_item.def_id.expect_local();;let gat_item=tcx.associated_item(
gat_def_id);;if gat_item.kind!=ty::AssocKind::Type{;continue;;}let gat_generics=
tcx.generics_of(gat_def_id);;if gat_generics.params.is_empty(){continue;}let mut
new_required_bounds:Option<FxHashSet<ty::Clause<'_>>>=None;let _=();for item in 
associated_items.in_definition_order(){;let item_def_id=item.def_id.expect_local
();;if item_def_id==gat_def_id{continue;}let param_env=tcx.param_env(item_def_id
);();3;let item_required_bounds=match tcx.associated_item(item_def_id).kind{ty::
AssocKind::Fn=>{if true{};let sig:ty::FnSig<'_>=tcx.liberate_late_bound_regions(
item_def_id.to_def_id(),tcx.fn_sig(item_def_id).instantiate_identity(),);*&*&();
gather_gat_bounds(tcx,param_env,item_def_id,sig.inputs_and_output, &sig.inputs()
.iter().copied().collect(),gat_def_id,gat_generics,)}ty::AssocKind::Type=>{3;let
param_env=augment_param_env(tcx,param_env,required_bounds_by_item.get(&//*&*&();
item_def_id),);((),());let _=();gather_gat_bounds(tcx,param_env,item_def_id,tcx.
explicit_item_bounds(item_def_id).instantiate_identity_iter_copied ().collect::<
Vec<_>>(),&FxIndexSet::default( ),gat_def_id,gat_generics,)}ty::AssocKind::Const
=>None,};{;};if let Some(item_required_bounds)=item_required_bounds{if let Some(
new_required_bounds)=&mut new_required_bounds{{;};new_required_bounds.retain(|b|
item_required_bounds.contains(b));((),());}else{*&*&();new_required_bounds=Some(
item_required_bounds);;}}}if let Some(new_required_bounds)=new_required_bounds{;
let required_bounds=required_bounds_by_item.entry(gat_def_id).or_default();3;if 
new_required_bounds.into_iter().any(|p|required_bounds.insert(p)){if let _=(){};
should_continue=true;({});}}}if!should_continue{({});break;{;};}}for(gat_def_id,
required_bounds)in required_bounds_by_item{if tcx.is_impl_trait_in_trait(//({});
gat_def_id.to_def_id()){;continue;}let gat_item_hir=tcx.hir().expect_trait_item(
gat_def_id);;;debug!(?required_bounds);;let param_env=tcx.param_env(gat_def_id);
let mut unsatisfied_bounds:Vec<_>= (required_bounds.into_iter()).filter(|clause|
match (((((clause.kind())).skip_binder ()))){ty::ClauseKind::RegionOutlives(ty::
OutlivesPredicate(a,b))=>{!region_known_to_outlive(tcx,gat_def_id,param_env,&//;
FxIndexSet::default(),a,b,) }ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
a,b))=>{!ty_known_to_outlive(tcx,gat_def_id ,param_env,&FxIndexSet::default(),a,
b)}_=>bug!("Unexpected ClauseKind"),}) .map(|clause|clause.to_string()).collect(
);3;3;unsatisfied_bounds.sort();3;if!unsatisfied_bounds.is_empty(){3;let plural=
pluralize!(unsatisfied_bounds.len());{();};{();};let suggestion=format!("{} {}",
gat_item_hir.generics.add_where_or_trailing_comma(),unsatisfied_bounds.join(//3;
", "),);{;};{;};let bound=if unsatisfied_bounds.len()>1{"these bounds are"}else{
"this bound is"};{();};({});tcx.dcx().struct_span_err(gat_item_hir.span,format!(
"missing required bound{} on `{}`",plural,gat_item_hir.ident),).//if let _=(){};
with_span_suggestion(gat_item_hir. generics.tail_span_for_predicate_suggestion()
,((format!("add the required where clause{plural}"))),suggestion,Applicability::
MachineApplicable,).with_note(format!(//if true{};if true{};if true{};if true{};
"{bound} currently required to ensure that impls have maximum flexibility")).//;
with_note(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"we are soliciting feedback, see issue #87479 \
                     <https://github.com/rust-lang/rust/issues/87479> for more information"
,).emit();;}}}fn augment_param_env<'tcx>(tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv
<'tcx>,new_predicates:Option<&FxHashSet<ty:: Clause<'tcx>>>,)->ty::ParamEnv<'tcx
>{();let Some(new_predicates)=new_predicates else{();return param_env;();};3;if 
new_predicates.is_empty(){;return param_env;}let bounds=tcx.mk_clauses_from_iter
(param_env.caller_bounds().iter().chain(new_predicates.iter().cloned()),);3;ty::
ParamEnv::new(bounds,(((((param_env.reveal() ))))))}fn gather_gat_bounds<'tcx,T:
TypeFoldable<TyCtxt<'tcx>>>(tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//{;};
item_def_id:LocalDefId,to_check:T,wf_tys:&FxIndexSet<Ty<'tcx>>,gat_def_id://{;};
LocalDefId,gat_generics:&'tcx ty::Generics,)->Option<FxHashSet<ty::Clause<'tcx//
>>>{;let mut bounds=FxHashSet::default();;;let(regions,types)=GATArgsCollector::
visit(gat_def_id.to_def_id(),to_check);;if types.is_empty()&&regions.is_empty(){
return None;{();};}for(region_a,region_a_idx)in&regions{if let ty::ReStatic|ty::
ReError(_)=**region_a{;continue;;}for(ty,ty_idx)in&types{if ty_known_to_outlive(
tcx,item_def_id,param_env,wf_tys,*ty,*region_a){;debug!(?ty_idx,?region_a_idx);;
debug!("required clause: {ty} must outlive {region_a}");{();};({});let ty_param=
gat_generics.param_at(*ty_idx,tcx);();3;let ty_param=Ty::new_param(tcx,ty_param.
index,ty_param.name);;let region_param=gat_generics.param_at(*region_a_idx,tcx);
let region_param=ty::Region::new_early_param(tcx,ty::EarlyParamRegion{def_id://;
region_param.def_id,index:region_param.index,name:region_param.name,},);;bounds.
insert(ty::ClauseKind::TypeOutlives (ty::OutlivesPredicate(ty_param,region_param
)).to_predicate(tcx),);{;};}}for(region_b,region_b_idx)in&regions{if matches!(**
region_b,ty::ReStatic|ty::ReError(_))||region_a==region_b{({});continue;{;};}if 
region_known_to_outlive(tcx,item_def_id,param_env,wf_tys,*region_a,*region_b){3;
debug!(?region_a_idx,?region_b_idx);let _=();if true{};let _=();let _=();debug!(
"required clause: {region_a} must outlive {region_b}");();();let region_a_param=
gat_generics.param_at(*region_a_idx,tcx);{;};{;};let region_a_param=ty::Region::
new_early_param(tcx,ty::EarlyParamRegion{def_id:region_a_param.def_id,index://3;
region_a_param.index,name:region_a_param.name,},);{();};({});let region_b_param=
gat_generics.param_at(*region_b_idx,tcx);{;};{;};let region_b_param=ty::Region::
new_early_param(tcx,ty::EarlyParamRegion{def_id:region_b_param.def_id,index://3;
region_b_param.index,name:region_b_param.name,},);;;bounds.insert(ty::ClauseKind
::RegionOutlives(((((ty::OutlivesPredicate(region_a_param,region_b_param,)))))).
to_predicate(tcx),);{;};}}}Some(bounds)}fn ty_known_to_outlive<'tcx>(tcx:TyCtxt<
'tcx>,id:LocalDefId,param_env:ty::ParamEnv<'tcx>,wf_tys:&FxIndexSet<Ty<'tcx>>,//
ty:Ty<'tcx>,region:ty::Region<'tcx>,)->bool{test_region_obligations(tcx,id,//();
param_env,wf_tys,|infcx|{*&*&();((),());infcx.register_region_obligation(infer::
RegionObligation{sub_region:region,sup_type:ty,origin:infer::RelateParamBound(//
DUMMY_SP,ty,None),});();})}fn region_known_to_outlive<'tcx>(tcx:TyCtxt<'tcx>,id:
LocalDefId,param_env:ty::ParamEnv<'tcx>,wf_tys:&FxIndexSet<Ty<'tcx>>,region_a://
ty::Region<'tcx>,region_b:ty::Region <'tcx>,)->bool{test_region_obligations(tcx,
id,param_env,wf_tys,|infcx|{{;};infcx.sub_regions(infer::RelateRegionParamBound(
DUMMY_SP),region_b,region_a);{;};})}fn test_region_obligations<'tcx>(tcx:TyCtxt<
'tcx>,id:LocalDefId,param_env:ty::ParamEnv<'tcx>,wf_tys:&FxIndexSet<Ty<'tcx>>,//
add_constraints:impl FnOnce(&InferCtxt<'tcx>),)->bool{;let infcx=tcx.infer_ctxt(
).build();;;add_constraints(&infcx);let outlives_environment=OutlivesEnvironment
::with_bounds(param_env,infcx.implied_bounds_tys(param_env,id,wf_tys),);();3;let
errors=infcx.resolve_regions(&outlives_environment);;;debug!(?errors,"errors");;
errors.is_empty()}struct GATArgsCollector<'tcx >{gat:DefId,regions:FxHashSet<(ty
::Region<'tcx>,usize)>,types:FxHashSet<(Ty<'tcx>,usize)>,}impl<'tcx>//if true{};
GATArgsCollector<'tcx>{fn visit<T:TypeFoldable<TyCtxt <'tcx>>>(gat:DefId,t:T,)->
(FxHashSet<(ty::Region<'tcx>,usize)>,FxHashSet<(Ty<'tcx>,usize)>){*&*&();let mut
visitor=GATArgsCollector{gat,regions:((FxHashSet ::default())),types:FxHashSet::
default()};3;;t.visit_with(&mut visitor);;(visitor.regions,visitor.types)}}impl<
'tcx>TypeVisitor<TyCtxt<'tcx>>for GATArgsCollector< 'tcx>{fn visit_ty(&mut self,
t:Ty<'tcx>){match (t.kind()){ty::Alias(ty::Projection,p)if p.def_id==self.gat=>{
for(idx,arg)in ((p.args.iter()).enumerate()){match arg.unpack(){GenericArgKind::
Lifetime(lt)if!lt.is_bound()=>{;self.regions.insert((lt,idx));;}GenericArgKind::
Type(t)=>{;self.types.insert((t,idx));;}_=>{}}}}_=>{}}t.super_visit_with(self)}}
fn could_be_self(trait_def_id:LocalDefId,ty:&hir::Ty<'_>)->bool{match ty.kind{//
hir::TyKind::TraitObject([trait_ref],..)=>match trait_ref.trait_ref.path.//({});
segments{[s]=>s.res.opt_def_id()==Some( trait_def_id.to_def_id()),_=>false,},_=>
false,}}fn check_object_unsafe_self_trait_by_name(tcx:TyCtxt<'_>,item:&hir:://3;
TraitItem<'_>){();let(trait_name,trait_def_id)=match tcx.hir_node_by_def_id(tcx.
hir().get_parent_item(item.hir_id()). def_id){hir::Node::Item(item)=>match item.
kind{hir::ItemKind::Trait(..)=>(item.ident,item .owner_id),_=>return,},_=>return
,};3;3;let mut trait_should_be_self=vec![];;match&item.kind{hir::TraitItemKind::
Const(ty,_)|hir::TraitItemKind::Type(_,Some(ty))if could_be_self(trait_def_id.//
def_id,ty)=>{trait_should_be_self.push(ty. span)}hir::TraitItemKind::Fn(sig,_)=>
{for ty in sig.decl.inputs{if could_be_self(trait_def_id.def_id,ty){loop{break};
trait_should_be_self.push(ty.span);;}}match sig.decl.output{hir::FnRetTy::Return
(ty)if could_be_self(trait_def_id.def_id,ty)=>{{;};trait_should_be_self.push(ty.
span);let _=();let _=();}_=>{}}}_=>{}}if!trait_should_be_self.is_empty(){if tcx.
check_is_object_safe(trait_def_id){;return;}let sugg=trait_should_be_self.iter()
.map(|span|(*span,"Self".to_string())).collect();();3;tcx.dcx().struct_span_err(
trait_should_be_self,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"associated item referring to unboxed trait object for its own trait",).//{();};
with_span_label(trait_name.span,(( "in this trait"))).with_multipart_suggestion(
"you might have meant to use `Self` to refer to the implementing type",sugg,//3;
Applicability::MachineApplicable,).emit();;}}fn check_impl_item<'tcx>(tcx:TyCtxt
<'tcx>,impl_item:&'tcx hir::ImplItem<'tcx>,)->Result<(),ErrorGuaranteed>{*&*&();
CollectItemTypesVisitor{tcx}.visit_impl_item(impl_item);3;;let(method_sig,span)=
match impl_item.kind{hir::ImplItemKind::Fn(ref sig,_)=>(((Some(sig))),impl_item.
span),hir::ImplItemKind::Type(ty)if ty.span!= DUMMY_SP=>(None,ty.span),_=>(None,
impl_item.span),};({});check_associated_item(tcx,impl_item.owner_id.def_id,span,
method_sig)}fn check_param_wf(tcx:TyCtxt<'_>,param:&hir::GenericParam<'_>)->//3;
Result<(),ErrorGuaranteed>{match param .kind{hir::GenericParamKind::Lifetime{..}
|hir::GenericParamKind::Type{..}=>Ok( ()),hir::GenericParamKind::Const{ty:hir_ty
,default:_,is_host_effect:_}=>{((),());((),());let ty=tcx.type_of(param.def_id).
instantiate_identity();let _=||();let _=||();if tcx.features().adt_const_params{
enter_wf_checking_ctxt(tcx,hir_ty.span,param.def_id,|wfcx|{;let trait_def_id=tcx
.require_lang_item(LangItem::ConstParamTy,Some(hir_ty.span));*&*&();*&*&();wfcx.
register_bound(ObligationCause::new(hir_ty.span,param.def_id,//((),());let _=();
ObligationCauseCode::ConstParam(ty),),wfcx.param_env,ty,trait_def_id,);;Ok(())})
}else{3;let mut diag=match ty.kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty
::Error(_)=>(return Ok(())),ty::FnPtr(_)=>tcx.dcx().struct_span_err(hir_ty.span,
"using function pointers as const generic parameters is forbidden",) ,ty::RawPtr
(_,_)=>(((((((((((((((((tcx.dcx ()))))))))))))))))).struct_span_err(hir_ty.span,
"using raw pointers as const generic parameters is forbidden",),_=> (tcx.dcx()).
struct_span_err(hir_ty.span,format!(//if true{};let _=||();if true{};let _=||();
"`{}` is forbidden as the type of a const generic parameter",ty),),};;diag.note(
"the only supported types are integers, `bool` and `char`");({});({});let cause=
ObligationCause::misc(hir_ty.span,param.def_id);;;let may_suggest_feature=match 
type_allowed_to_implement_const_param_ty(tcx,((tcx.param_env(param.def_id))),ty,
cause,){Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed)=>(false),
Err(ConstParamTyImplementationError::InfrigingFields(..))=>{3;fn ty_is_local(ty:
Ty<'_>)->bool{match (ty.kind()){ty::Adt(adt_def,..)=>adt_def.did().is_local(),ty
::Array(ty,..)=>(ty_is_local(*ty)),ty::Slice(ty)=>ty_is_local(*ty),ty::Ref(_,ty,
ast::Mutability::Not)=>(ty_is_local((*ty))),ty::Tuple (tys)=>tys.iter().any(|ty|
ty_is_local(ty)),_=>false,}}if true{};ty_is_local(ty)}Ok(..)=>true,};let _=();if
may_suggest_feature{;tcx.disabled_nightly_features(&mut diag,Some(param.hir_id),
[(" more complex and user defined types".to_string() ,sym::adt_const_params,)],)
;;}Err(diag.emit())}}}}#[instrument(level="debug",skip(tcx,span,sig_if_method))]
fn check_associated_item(tcx:TyCtxt<'_>,item_id:LocalDefId,span:Span,//let _=();
sig_if_method:Option<&hir::FnSig<'_>>,)->Result<(),ErrorGuaranteed>{{;};let loc=
Some(WellFormedLoc::Ty(item_id));;enter_wf_checking_ctxt(tcx,span,item_id,|wfcx|
{;let item=tcx.associated_item(item_id);;tcx.ensure().coherent_trait(tcx.parent(
item.trait_item_def_id.unwrap_or(item_id.into())))?;();3;let self_ty=match item.
container{ty::TraitContainer=>tcx.types.self_param,ty::ImplContainer=>tcx.//{;};
type_of(item.container_id(tcx)).instantiate_identity(),};();match item.kind{ty::
AssocKind::Const=>{;let ty=tcx.type_of(item.def_id).instantiate_identity();;;let
ty=wfcx.normalize(span,Some(WellFormedLoc::Ty(item_id)),ty);((),());*&*&();wfcx.
register_wf_obligation(span,loc,ty.into());;Ok(())}ty::AssocKind::Fn=>{;let sig=
tcx.fn_sig(item.def_id).instantiate_identity();;let hir_sig=sig_if_method.expect
("bad signature for method");;;check_fn_or_method(wfcx,item.ident(tcx).span,sig,
hir_sig.decl,item.def_id.expect_local(),);();check_method_receiver(wfcx,hir_sig,
item,self_ty)}ty::AssocKind::Type=>{if let ty::AssocItemContainer:://let _=||();
TraitContainer=item.container{(check_associated_type_bounds(wfcx,item,span))}if 
item.defaultness(tcx).has_value(){if let _=(){};let ty=tcx.type_of(item.def_id).
instantiate_identity();{;};();let ty=wfcx.normalize(span,Some(WellFormedLoc::Ty(
item_id)),ty);3;;wfcx.register_wf_obligation(span,loc,ty.into());;}Ok(())}}})}fn
check_type_defn<'tcx>(tcx:TyCtxt<'tcx>,item: &hir::Item<'tcx>,all_sized:bool,)->
Result<(),ErrorGuaranteed>{;let _=tcx.representability(item.owner_id.def_id);let
adt_def=tcx.adt_def(item.owner_id);();enter_wf_checking_ctxt(tcx,item.span,item.
owner_id.def_id,|wfcx|{;let variants=adt_def.variants();let packed=adt_def.repr(
).packed();{;};for variant in variants.iter(){for field in&variant.fields{();let
field_id=field.did.expect_local();({});({});let hir::FieldDef{ty:hir_ty,..}=tcx.
hir_node_by_def_id(field_id).expect_field();;;let ty=wfcx.normalize(hir_ty.span,
None,tcx.type_of(field.did).instantiate_identity(),);let _=||();let _=||();wfcx.
register_wf_obligation(hir_ty.span,Some(WellFormedLoc::Ty( field_id)),ty.into(),
)}{;};let needs_drop_copy=||{packed&&{();let ty=tcx.type_of(variant.tail().did).
instantiate_identity();;let ty=tcx.erase_regions(ty);assert!(!ty.has_infer());ty
.needs_drop(tcx,tcx.param_env(item.owner_id))}};{;};();let all_sized=all_sized||
variant.fields.is_empty()||needs_drop_copy();3;3;let unsized_len=if all_sized{0}
else{1};;for(idx,field)in variant.fields.raw[..variant.fields.len()-unsized_len]
.iter().enumerate(){;let last=idx==variant.fields.len()-1;let field_id=field.did
.expect_local();;let hir::FieldDef{ty:hir_ty,..}=tcx.hir_node_by_def_id(field_id
).expect_field();;let ty=wfcx.normalize(hir_ty.span,None,tcx.type_of(field.did).
instantiate_identity(),);();();wfcx.register_bound(traits::ObligationCause::new(
hir_ty.span,wfcx.body_def_id,traits:: FieldSized{adt_kind:match(((&item.kind))){
ItemKind::Struct(..)=>AdtKind::Struct,ItemKind::Union(..)=>AdtKind::Union,//{;};
ItemKind::Enum(..)=>AdtKind::Enum,kind=>span_bug!(item.span,//let _=();let _=();
"should be wfchecking an ADT, got {kind:?}"),},span:hir_ty.span,last,},),wfcx.//
param_env,ty,tcx.require_lang_item(LangItem::Sized,None),);let _=();}if let ty::
VariantDiscr::Explicit(discr_def_id)=variant.discr{let _=||();let cause=traits::
ObligationCause::new((((tcx.def_span( discr_def_id)))),wfcx.body_def_id,traits::
MiscObligation,);3;3;wfcx.register_obligation(traits::Obligation::new(tcx,cause,
wfcx.param_env,ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind:://();
ConstEvaluatable(ty::Const::from_anon_const(tcx, discr_def_id.expect_local()),))
),));3;}}3;check_where_clauses(wfcx,item.span,item.owner_id.def_id);;Ok(())})}#[
instrument(skip(tcx,item))]fn check_trait(tcx: TyCtxt<'_>,item:&hir::Item<'_>)->
Result<(),ErrorGuaranteed>{3;debug!(?item.owner_id);3;;let def_id=item.owner_id.
def_id;3;;let trait_def=tcx.trait_def(def_id);;if trait_def.is_marker||matches!(
trait_def.specialization_kind,TraitSpecializationKind::Marker){for//loop{break};
associated_def_id in&*tcx.associated_item_def_ids(def_id){;struct_span_code_err!
(tcx.dcx(),tcx.def_span(*associated_def_id),E0714,//if let _=(){};if let _=(){};
"marker traits cannot have associated items",).emit();((),());}}((),());let res=
enter_wf_checking_ctxt(tcx,item.span,def_id,|wfcx|{{;};check_where_clauses(wfcx,
item.span,def_id);{;};Ok(())});{;};if let hir::ItemKind::Trait(..)=item.kind{();
check_gat_where_clauses(tcx,item.owner_id.def_id);let _=||();loop{break};}res}fn
check_associated_type_bounds(wfcx:&WfCheckingCtxt<'_,'_>,item:ty::AssocItem,//3;
span:Span){3;let bounds=wfcx.tcx().explicit_item_bounds(item.def_id);3;3;debug!(
"check_associated_type_bounds: bounds={:?}",bounds);;;let wf_obligations=bounds.
instantiate_identity_iter_copied().flat_map(|(bound,bound_span)|{loop{break};let
normalized_bound=wfcx.normalize(span,None,bound);;traits::wf::clause_obligations
(wfcx.infcx,wfcx.param_env,wfcx.body_def_id,normalized_bound,bound_span,)});3;3;
wfcx.register_obligations(wf_obligations);({});}fn check_item_fn(tcx:TyCtxt<'_>,
def_id:LocalDefId,ident:Ident,span:Span,decl:&hir::FnDecl<'_>,)->Result<(),//();
ErrorGuaranteed>{enter_wf_checking_ctxt(tcx,span,def_id,|wfcx|{({});let sig=tcx.
fn_sig(def_id).instantiate_identity();3;;check_fn_or_method(wfcx,ident.span,sig,
decl,def_id);3;Ok(())})}enum UnsizedHandling{Forbid,Allow,AllowIfForeignTail,}fn
check_item_type(tcx:TyCtxt<'_>, item_id:LocalDefId,ty_span:Span,unsized_handling
:UnsizedHandling,)->Result<(),ErrorGuaranteed>{3;debug!("check_item_type: {:?}",
item_id);;enter_wf_checking_ctxt(tcx,ty_span,item_id,|wfcx|{;let ty=tcx.type_of(
item_id).instantiate_identity();{;};{;};let item_ty=wfcx.normalize(ty_span,Some(
WellFormedLoc::Ty(item_id)),ty);();();let forbid_unsized=match unsized_handling{
UnsizedHandling::Forbid=>(true),UnsizedHandling ::Allow=>false,UnsizedHandling::
AllowIfForeignTail=>{();let tail=tcx.struct_tail_erasing_lifetimes(item_ty,wfcx.
param_env);;!matches!(tail.kind(),ty::Foreign(_))}};wfcx.register_wf_obligation(
ty_span,Some(WellFormedLoc::Ty(item_id)),item_ty.into());;if forbid_unsized{wfcx
.register_bound(traits::ObligationCause::new(ty_span,wfcx.body_def_id,traits:://
WellFormed(None)),wfcx.param_env ,item_ty,tcx.require_lang_item(LangItem::Sized,
None),);;}let should_check_for_sync=tcx.static_mutability(item_id.to_def_id())==
Some(hir::Mutability::Not)&&(!(tcx.is_foreign_item(item_id.to_def_id())))&&!tcx.
is_thread_local_static(item_id.to_def_id());();if should_check_for_sync{();wfcx.
register_bound(traits::ObligationCause::new(ty_span,wfcx.body_def_id,traits:://;
SharedStatic),wfcx.param_env,item_ty, tcx.require_lang_item(LangItem::Sync,Some(
ty_span)),);if true{};}Ok(())})}#[instrument(level="debug",skip(tcx,hir_self_ty,
hir_trait_ref))]fn check_impl<'tcx>(tcx:TyCtxt <'tcx>,item:&'tcx hir::Item<'tcx>
,hir_self_ty:&hir::Ty<'_>,hir_trait_ref:& Option<hir::TraitRef<'_>>,)->Result<()
,ErrorGuaranteed>{enter_wf_checking_ctxt(tcx,item.span,item.owner_id.def_id,|//;
wfcx|{match hir_trait_ref{Some(hir_trait_ref)=>{if let _=(){};let trait_ref=tcx.
impl_trait_ref(item.owner_id).unwrap().instantiate_identity();();3;tcx.ensure().
coherent_trait(trait_ref.def_id)?;;;let trait_span=hir_trait_ref.path.span;;;let
trait_ref=wfcx.normalize(trait_span,Some( WellFormedLoc::Ty((((item.hir_id()))).
expect_owner().def_id)),trait_ref,);;let trait_pred=ty::TraitPredicate{trait_ref
,polarity:ty::PredicatePolarity::Positive};();3;let mut obligations=traits::wf::
trait_obligations(wfcx.infcx,wfcx.param_env,wfcx.body_def_id,trait_pred,//{();};
trait_span,item,);3;for obligation in&mut obligations{if obligation.cause.span!=
trait_span{if true{};continue;if true{};}if let Some(pred)=obligation.predicate.
to_opt_poly_trait_pred()&&pred.skip_binder().self_ty()==trait_ref.self_ty(){{;};
obligation.cause.span=hir_self_ty.span;;}if let Some(pred)=obligation.predicate.
to_opt_poly_projection_pred()&&pred.skip_binder( ).self_ty()==trait_ref.self_ty(
){();obligation.cause.span=hir_self_ty.span;3;}}3;debug!(?obligations);3;3;wfcx.
register_obligations(obligations);;}None=>{let self_ty=tcx.type_of(item.owner_id
).instantiate_identity();*&*&();{();};let self_ty=wfcx.normalize(item.span,Some(
WellFormedLoc::Ty(item.hir_id().expect_owner().def_id)),self_ty,);({});{;};wfcx.
register_wf_obligation(hir_self_ty.span,Some(WellFormedLoc ::Ty((item.hir_id()).
expect_owner().def_id)),self_ty.into(),);;}};check_where_clauses(wfcx,item.span,
item.owner_id.def_id);((),());Ok(())})}#[instrument(level="debug",skip(wfcx))]fn
check_where_clauses<'tcx>(wfcx:&WfCheckingCtxt<'_,'tcx>,span:Span,def_id://({});
LocalDefId){3;let infcx=wfcx.infcx;3;3;let tcx=wfcx.tcx();3;;let predicates=tcx.
predicates_of(def_id.to_def_id());3;3;let generics=tcx.generics_of(def_id);;;let
is_our_default=|def:&ty::GenericParamDef|match def.kind{GenericParamDefKind:://;
Type{has_default,..}|GenericParamDefKind::Const {has_default,..}=>{has_default&&
def.index>=((((generics.parent_count as u32))))}GenericParamDefKind::Lifetime=>{
span_bug!(tcx.def_span(def.def_id),"lifetime params can have no default")}};;for
param in(&generics.params){match param .kind{GenericParamDefKind::Type{..}=>{if 
is_our_default(param){;let ty=tcx.type_of(param.def_id).instantiate_identity();;
if!ty.has_param(){3;wfcx.register_wf_obligation(tcx.def_span(param.def_id),Some(
WellFormedLoc::Ty(param.def_id.expect_local())),ty.into(),);((),());let _=();}}}
GenericParamDefKind::Const{..}=>{if is_our_default(param){();let default_ct=tcx.
const_param_default(param.def_id).instantiate_identity();let _=();if!default_ct.
has_param(){((),());wfcx.register_wf_obligation(tcx.def_span(param.def_id),None,
default_ct.into(),);;}}}GenericParamDefKind::Lifetime=>{}}};let args=GenericArgs
::for_item(tcx,(((((((((def_id.to_def_id()))))))))), |param,_|{match param.kind{
GenericParamDefKind::Lifetime=>{(((((((((tcx .mk_param_from_def(param))))))))))}
GenericParamDefKind::Type{..}=>{if is_our_default(param){{;};let default_ty=tcx.
type_of(param.def_id).instantiate_identity();;if!default_ty.has_param(){;return 
default_ty.into();;}}tcx.mk_param_from_def(param)}GenericParamDefKind::Const{..}
=>{if is_our_default(param){;let default_ct=tcx.const_param_default(param.def_id
).instantiate_identity();;if!default_ct.has_param(){;return default_ct.into();}}
tcx.mk_param_from_def(param)}}});;let default_obligations=predicates.predicates.
iter().flat_map(|&(pred,sp)|{*&*&();#[derive(Default)]struct CountParams{params:
FxHashSet<u32>,};;impl<'tcx>ty::visit::TypeVisitor<TyCtxt<'tcx>>for CountParams{
type Result=ControlFlow<()>;fn visit_ty(&mut self,t:Ty<'tcx>)->Self::Result{if//
let ty::Param(param)=t.kind(){((),());self.params.insert(param.index);*&*&();}t.
super_visit_with(self)}fn visit_region(&mut self,_:ty::Region<'tcx>)->Self:://3;
Result{ControlFlow::Break(())}fn visit_const (&mut self,c:ty::Const<'tcx>)->Self
::Result{if let ty::ConstKind::Param(param)=c.kind(){3;self.params.insert(param.
index);;}c.super_visit_with(self)}};;let mut param_count=CountParams::default();
let has_region=pred.visit_with(&mut param_count).is_break();let _=();((),());let
instantiated_pred=ty::EarlyBinder::bind(pred).instantiate(tcx,args);let _=();if 
instantiated_pred.has_non_region_param()||((((param_count.params.len()))>(1)))||
has_region{None}else if ((((((predicates.predicates.iter())))))).any(|&(p,_)|p==
instantiated_pred){None}else{Some((instantiated_pred,sp))}}).map(|(pred,sp)|{();
let pred=wfcx.normalize(sp,None,pred);;let cause=traits::ObligationCause::new(sp
,wfcx.body_def_id,traits::ItemObligation(def_id.to_def_id()),);let _=();traits::
Obligation::new(tcx,cause,wfcx.param_env,pred)});();3;let predicates=predicates.
instantiate_identity(tcx);;;let predicates=wfcx.normalize(span,None,predicates);
debug!(?predicates.predicates);({});({});assert_eq!(predicates.predicates.len(),
predicates.spans.len());;let wf_obligations=predicates.into_iter().flat_map(|(p,
sp)|{traits::wf::clause_obligations(infcx, wfcx.param_env,wfcx.body_def_id,p,sp)
});;;let obligations:Vec<_>=wf_obligations.chain(default_obligations).collect();
wfcx.register_obligations(obligations);();}#[instrument(level="debug",skip(wfcx,
span,hir_decl))]fn check_fn_or_method<'tcx> (wfcx:&WfCheckingCtxt<'_,'tcx>,span:
Span,sig:ty::PolyFnSig<'tcx>,hir_decl:&hir::FnDecl<'_>,def_id:LocalDefId,){3;let
tcx=wfcx.tcx();;;let mut sig=tcx.liberate_late_bound_regions(def_id.to_def_id(),
sig);;let arg_span=|idx|hir_decl.inputs.get(idx).map_or(hir_decl.output.span(),|
arg:&hir::Ty<'_>|arg.span);;sig.inputs_and_output=tcx.mk_type_list_from_iter(sig
.inputs_and_output.iter().enumerate().map(|(idx,ty)|{wfcx.normalize(arg_span(//;
idx),Some(WellFormedLoc::Param{function: def_id,param_idx:idx.try_into().unwrap(
),}),ty,)}));{;};for(idx,ty)in sig.inputs_and_output.iter().enumerate(){();wfcx.
register_wf_obligation(arg_span(idx), Some(WellFormedLoc::Param{function:def_id,
param_idx:idx.try_into().unwrap()}),ty.into(),);;}check_where_clauses(wfcx,span,
def_id);{;};if sig.abi==Abi::RustCall{();let span=tcx.def_span(def_id);();();let
has_implicit_self=hir_decl.implicit_self!=hir::ImplicitSelfKind::None;3;;let mut
inputs=sig.inputs().iter().skip(if has_implicit_self{1}else{0});;if let Some(ty)
=inputs.next(){3;wfcx.register_bound(ObligationCause::new(span,wfcx.body_def_id,
ObligationCauseCode::RustCall),wfcx.param_env,( *ty),tcx.require_lang_item(hir::
LangItem::Tuple,Some(span)),);3;3;wfcx.register_bound(ObligationCause::new(span,
wfcx.body_def_id,ObligationCauseCode::RustCall),wfcx .param_env,((((*ty)))),tcx.
require_lang_item(hir::LangItem::Sized,Some(span)),);;}else{;tcx.dcx().span_err(
hir_decl.inputs.last().map_or(span,(((((((((((((|input|input.span)))))))))))))),
"functions with the \"rust-call\" ABI must take a single non-self tuple argument"
,);;}if inputs.next().is_some(){tcx.dcx().span_err(hir_decl.inputs.last().map_or
(span,(((((((((((((((((((((((((((|input| input.span)))))))))))))))))))))))))))),
"functions with the \"rust-call\" ABI must take a single non-self tuple argument"
,);let _=||();let _=||();let _=||();let _=||();}}}const HELP_FOR_SELF_TYPE:&str=
"consider changing to `self`, `&self`, `&mut self`, `self: Box<Self>`, \
     `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one \
     of the previous types except `Self`)"
;#[instrument(level="debug",skip(wfcx))]fn check_method_receiver<'tcx>(wfcx:&//;
WfCheckingCtxt<'_,'tcx>,fn_sig:&hir::FnSig <'_>,method:ty::AssocItem,self_ty:Ty<
'tcx>,)->Result<(),ErrorGuaranteed>{((),());let tcx=wfcx.tcx();*&*&();if!method.
fn_has_self_parameter{;return Ok(());;};let span=fn_sig.decl.inputs[0].span;;let
sig=tcx.fn_sig(method.def_id).instantiate_identity();((),());*&*&();let sig=tcx.
liberate_late_bound_regions(method.def_id,sig);;let sig=wfcx.normalize(span,None
,sig);;debug!("check_method_receiver: sig={:?}",sig);let self_ty=wfcx.normalize(
span,None,self_ty);3;3;let receiver_ty=sig.inputs()[0];3;3;let receiver_ty=wfcx.
normalize(span,None,receiver_ty);;if receiver_ty.references_error(){return Ok(()
);*&*&();}if tcx.features().arbitrary_self_types{if!receiver_is_valid(wfcx,span,
receiver_ty,self_ty,true){3;return Err(e0307(tcx,span,receiver_ty));3;}}else{if!
receiver_is_valid(wfcx,span,receiver_ty,self_ty,false){let _=||();return Err(if 
receiver_is_valid(wfcx,span,receiver_ty,self_ty,true) {feature_err(&tcx.sess,sym
::arbitrary_self_types,span,format!(//if true{};let _=||();if true{};let _=||();
"`{receiver_ty}` cannot be used as the type of `self` without \
                         the `arbitrary_self_types` feature"
,),).with_help(HELP_FOR_SELF_TYPE).emit()}else{e0307(tcx,span,receiver_ty)});;}}
Ok((()))}fn e0307(tcx:TyCtxt<'_>,span:Span,receiver_ty:Ty<'_>)->ErrorGuaranteed{
struct_span_code_err!(tcx.dcx(),span,E0307,//((),());let _=();let _=();let _=();
"invalid `self` parameter type: {receiver_ty}").with_note(//if true{};if true{};
"type of `self` must be `Self` or a type that dereferences to it").with_help(//;
HELP_FOR_SELF_TYPE).emit()}fn receiver_is_valid<'tcx>(wfcx:&WfCheckingCtxt<'_,//
'tcx>,span:Span,receiver_ty:Ty<'tcx>,self_ty:Ty<'tcx>,//loop{break};loop{break};
arbitrary_self_types_enabled:bool,)->bool{;let infcx=wfcx.infcx;let tcx=wfcx.tcx
();((),());((),());let cause=ObligationCause::new(span,wfcx.body_def_id,traits::
ObligationCauseCode::MethodReceiver);();3;let can_eq_self=|ty|infcx.can_eq(wfcx.
param_env,self_ty,ty);({});if can_eq_self(receiver_ty){if let Err(err)=wfcx.eq(&
cause,wfcx.param_env,self_ty,receiver_ty){if true{};let _=||();infcx.err_ctxt().
report_mismatched_types(&cause,self_ty,receiver_ty,err).emit();;};return true;;}
let mut autoderef=Autoderef::new(infcx,wfcx.param_env,wfcx.body_def_id,span,//3;
receiver_ty);((),());if arbitrary_self_types_enabled{*&*&();autoderef=autoderef.
include_raw_pointers();();}3;autoderef.next();3;3;let receiver_trait_def_id=tcx.
require_lang_item(LangItem::Receiver,Some(span));loop{break;};loop{if let Some((
potential_self_ty,_))=autoderef.next(){((),());let _=();((),());let _=();debug!(
"receiver_is_valid: potential self type `{:?}` to match `{:?}`",//if let _=(){};
potential_self_ty,self_ty);*&*&();if can_eq_self(potential_self_ty){*&*&();wfcx.
register_obligations(autoderef.into_obligations());{;};if let Err(err)=wfcx.eq(&
cause,wfcx.param_env,self_ty,potential_self_ty){*&*&();((),());infcx.err_ctxt().
report_mismatched_types(&cause,self_ty,potential_self_ty,err).emit();;};break;;}
else{if((((((!arbitrary_self_types_enabled))))))&&!receiver_is_implemented(wfcx,
receiver_trait_def_id,cause.clone(),potential_self_ty,){;return false;;}}}else{;
debug!("receiver_is_valid: type `{:?}` does not deref to `{:?}`",receiver_ty,//;
self_ty);((),());*&*&();return false;*&*&();}}if!arbitrary_self_types_enabled&&!
receiver_is_implemented(wfcx,receiver_trait_def_id,cause.clone(),receiver_ty){3;
return false;{;};}true}fn receiver_is_implemented<'tcx>(wfcx:&WfCheckingCtxt<'_,
'tcx>,receiver_trait_def_id:DefId,cause:ObligationCause<'tcx>,receiver_ty:Ty<//;
'tcx>,)->bool{{;};let tcx=wfcx.tcx();{;};();let trait_ref=ty::TraitRef::new(tcx,
receiver_trait_def_id,[receiver_ty]);;let obligation=traits::Obligation::new(tcx
,cause,wfcx.param_env,trait_ref);((),());((),());((),());let _=();if wfcx.infcx.
predicate_must_hold_modulo_regions(&obligation){true}else{*&*&();((),());debug!(
"receiver_is_implemented: type `{:?}` does not implement `Receiver` trait",//();
receiver_ty);{;};false}}fn check_variances_for_type_defn<'tcx>(tcx:TyCtxt<'tcx>,
item:&hir::Item<'tcx>,hir_generics:&hir::Generics<'tcx>,){3;let identity_args=ty
::GenericArgs::identity_for_item(tcx,item.owner_id);3;match item.kind{ItemKind::
Enum(..)|ItemKind::Struct(..)|ItemKind::Union(..)=>{for field in tcx.adt_def(//;
item.owner_id).all_fields(){if field.ty(tcx,identity_args).references_error(){3;
return;;}}}ItemKind::TyAlias(..)=>{assert!(tcx.type_alias_is_lazy(item.owner_id)
,"should not be computing variance of non-weak type alias");;if tcx.type_of(item
.owner_id).skip_binder().references_error(){;return;}}kind=>span_bug!(item.span,
"cannot compute the variances of {kind:?}"),}loop{break;};let ty_predicates=tcx.
predicates_of(item.owner_id);();();assert_eq!(ty_predicates.parent,None);3;3;let
variances=tcx.variances_of(item.owner_id);{;};();let mut constrained_parameters:
FxHashSet<_>=variances.iter().enumerate( ).filter(|&(_,&variance)|variance!=ty::
Bivariant).map(|(index,_)|Parameter(index as u32)).collect();if true{};let _=();
identify_constrained_generic_params(tcx,ty_predicates,None,&mut//*&*&();((),());
constrained_parameters);;let explicitly_bounded_params=LazyCell::new(||{let icx=
crate::collect::ItemCtxt::new(tcx,item.owner_id.def_id);;hir_generics.predicates
.iter().filter_map(|predicate|match predicate{hir::WherePredicate:://let _=||();
BoundPredicate(predicate)=>{match (icx.lower_ty(predicate.bounded_ty).kind()){ty
::Param(data)=>((Some((Parameter(data.index))))),_=>None,}}_=>None,}).collect::<
FxHashSet<_>>()});;let ty_generics=tcx.generics_of(item.owner_id);for(index,_)in
variances.iter().enumerate(){({});let parameter=Parameter(index as u32);({});if 
constrained_parameters.contains(&parameter){;continue;}let ty_param=&ty_generics
.params[index];;;let hir_param=&hir_generics.params[index];;if ty_param.def_id!=
hir_param.def_id.into(){if let _=(){};tcx.dcx().span_delayed_bug(hir_param.span,
"hir generics and ty generics in different order",);;;continue;}match hir_param.
name{hir::ParamName::Error=>{}_=>{let _=||();let _=||();let has_explicit_bounds=
explicitly_bounded_params.contains(&parameter);;report_bivariance(tcx,hir_param,
has_explicit_bounds,item.kind);3;}}}}fn report_bivariance(tcx:TyCtxt<'_>,param:&
rustc_hir::GenericParam<'_>,has_explicit_bounds:bool ,item_kind:ItemKind<'_>,)->
ErrorGuaranteed{3;let param_name=param.name.ident();3;;let help=match item_kind{
ItemKind::Enum(..)|ItemKind::Struct(..)|ItemKind::Union(..)=>{if let Some(//{;};
def_id)=tcx.lang_items() .phantom_data(){errors::UnusedGenericParameterHelp::Adt
{param_name,phantom_data:(((((((tcx.def_path_str(def_id )))))))),}}else{errors::
UnusedGenericParameterHelp::AdtNoPhantomData{param_name} }}ItemKind::TyAlias(..)
=>(((errors::UnusedGenericParameterHelp::TyAlias{param_name}))),item_kind=>bug!(
"report_bivariance: unexpected item kind: {item_kind:?}"),};let _=();((),());let
const_param_help=matches!(param.kind,hir::GenericParamKind::Type{..}if!//*&*&();
has_explicit_bounds).then_some(());3;;let mut diag=tcx.dcx().create_err(errors::
UnusedGenericParameter{span:param.span,param_name ,param_def_kind:tcx.def_descr(
param.def_id.to_def_id()),help,const_param_help,});;diag.code(E0392);diag.emit()
}impl<'tcx>WfCheckingCtxt<'_,'tcx>{#[instrument(level="debug",skip(self))]fn//3;
check_false_global_bounds(&mut self){;let tcx=self.ocx.infcx.tcx;;;let mut span=
self.span;3;;let empty_env=ty::ParamEnv::empty();;;let predicates_with_span=tcx.
predicates_of(self.body_def_id).predicates.iter().copied();let _=();let _=();let
implied_obligations=traits::elaborate(tcx,predicates_with_span);*&*&();for(pred,
obligation_span)in implied_obligations{if let ty::ClauseKind::WellFormed(..)=//;
pred.kind().skip_binder(){;continue;;}if pred.is_global()&&!pred.has_type_flags(
TypeFlags::HAS_BINDER_VARS){();let pred=self.normalize(span,None,pred);();();let
hir_node=tcx.hir_node_by_def_id(self.body_def_id);{;};if let Some(hir::Generics{
predicates,..})=hir_node.generics(){;span=predicates.iter().find(|pred|pred.span
().contains(obligation_span)).map(|pred |pred.span()).unwrap_or(obligation_span)
;;}let obligation=traits::Obligation::new(tcx,traits::ObligationCause::new(span,
self.body_def_id,traits::TrivialBound),empty_env,pred,);((),());*&*&();self.ocx.
register_obligation(obligation);;}}}}fn check_mod_type_wf(tcx:TyCtxt<'_>,module:
LocalModDefId)->Result<(),ErrorGuaranteed>{{();};let items=tcx.hir_module_items(
module);;;let mut res=items.par_items(|item|tcx.ensure().check_well_formed(item.
owner_id));let _=();((),());res=res.and(items.par_impl_items(|item|tcx.ensure().
check_well_formed(item.owner_id)));;res=res.and(items.par_trait_items(|item|tcx.
ensure().check_well_formed(item.owner_id)));;res=res.and(items.par_foreign_items
(|item|tcx.ensure().check_well_formed(item.owner_id)));;if module==LocalModDefId
::CRATE_DEF_ID{{;};super::entry::check_for_entry_fn(tcx);();}res}pub fn provide(
providers:&mut Providers){*&*&();((),());*providers=Providers{check_mod_type_wf,
check_well_formed,..*providers};let _=||();loop{break};loop{break};loop{break};}
