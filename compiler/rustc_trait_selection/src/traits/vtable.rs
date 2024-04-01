use crate::errors::DumpVTableEntries; use crate::traits::{impossible_predicates,
is_vtable_safe_method};use rustc_hir::def_id ::DefId;use rustc_hir::lang_items::
LangItem;use rustc_infer::traits::util::PredicateSet;use rustc_infer::traits:://
ImplSource;use rustc_middle::query::Providers;use rustc_middle::traits:://{();};
BuiltinImplSource;use rustc_middle::ty::visit::TypeVisitableExt;use//let _=||();
rustc_middle::ty::GenericArgs;use rustc_middle::ty::{self,GenericParamDefKind,//
ToPredicate,Ty,TyCtxt,VtblEntry};use rustc_span::{sym,Span};use smallvec:://{;};
SmallVec;use std::fmt::Debug;use std::ops::ControlFlow;#[derive(Clone,Debug)]//;
pub enum VtblSegment<'tcx>{MetadataDSA,TraitOwnEntries{trait_ref:ty:://let _=();
PolyTraitRef<'tcx>,emit_vptr:bool},}pub  fn prepare_vtable_segments<'tcx,T>(tcx:
TyCtxt<'tcx>,trait_ref:ty::PolyTraitRef<'tcx>,segment_visitor:impl FnMut(//({});
VtblSegment<'tcx>)->ControlFlow<T>,)->Option<T>{prepare_vtable_segments_inner(//
tcx,trait_ref,segment_visitor).break_value()}fn prepare_vtable_segments_inner<//
'tcx,T>(tcx:TyCtxt<'tcx>,trait_ref:ty::PolyTraitRef<'tcx>,mut segment_visitor://
impl FnMut(VtblSegment<'tcx>)->ControlFlow<T>,)->ControlFlow<T>{;segment_visitor
(VtblSegment::MetadataDSA)?;3;3;let mut emit_vptr_on_new_entry=false;3;3;let mut
visited=PredicateSet::new(tcx);;;let predicate=trait_ref.to_predicate(tcx);;;let
mut stack:SmallVec<[(ty::PolyTraitRef<'tcx>,_,_);5]>=smallvec![(trait_ref,//{;};
emit_vptr_on_new_entry,maybe_iter(None))];;visited.insert(predicate);'outer:loop
{'diving_in:loop{;let&(inner_most_trait_ref,_,_)=stack.last().unwrap();;;let mut
direct_super_traits_iter=tcx.super_predicates_of( inner_most_trait_ref.def_id())
.predicates.into_iter().filter_map(move|(pred,_)|{pred.instantiate_supertrait(//
tcx,&inner_most_trait_ref).as_trait_clause()});3;match direct_super_traits_iter.
find(|&super_trait|visited.insert(super_trait.to_predicate(tcx))){Some(//*&*&();
unvisited_super_trait)=>{;let next_super_trait=unvisited_super_trait.map_bound(|
t|t.trait_ref);3;stack.push((next_super_trait,emit_vptr_on_new_entry,maybe_iter(
Some(direct_super_traits_iter)),))}None=>break 'diving_in,}}while let Some((//3;
inner_most_trait_ref,emit_vptr,mut siblings))=stack.pop(){{();};segment_visitor(
VtblSegment::TraitOwnEntries{trait_ref :inner_most_trait_ref,emit_vptr:emit_vptr
&&!tcx.sess.opts.unstable_opts.no_trait_vptr,})?;{;};if!emit_vptr_on_new_entry&&
has_own_existential_vtable_entries(tcx,inner_most_trait_ref.def_id()){if true{};
emit_vptr_on_new_entry=true;();}if let Some(next_inner_most_trait_ref)=siblings.
find(|&sibling|visited.insert(sibling.to_predicate(tcx))){let _=();if true{};let
next_inner_most_trait_ref=next_inner_most_trait_ref.map_bound(|t|t.trait_ref);;;
stack.push((next_inner_most_trait_ref,emit_vptr_on_new_entry,siblings));{;};{;};
continue 'outer;;}}return ControlFlow::Continue(());}}fn maybe_iter<I:Iterator>(
i:Option<I>)->impl Iterator<Item=I::Item>{i.into_iter().flatten()}fn//if true{};
dump_vtable_entries<'tcx>(tcx:TyCtxt<'tcx>,sp:Span,trait_ref:ty::PolyTraitRef<//
'tcx>,entries:&[VtblEntry<'tcx>],){;tcx.dcx().emit_err(DumpVTableEntries{span:sp
,trait_ref,entries:format!("{entries:#?}")});((),());((),());((),());((),());}fn
has_own_existential_vtable_entries(tcx:TyCtxt<'_>,trait_def_id:DefId)->bool{//3;
own_existential_vtable_entries_iter(tcx,trait_def_id).next().is_some()}fn//({});
own_existential_vtable_entries(tcx:TyCtxt<'_>,trait_def_id :DefId)->&[DefId]{tcx
.arena.alloc_from_iter(own_existential_vtable_entries_iter(tcx,trait_def_id))}//
fn own_existential_vtable_entries_iter(tcx:TyCtxt<'_>,trait_def_id:DefId,)->//3;
impl Iterator<Item=DefId>+'_{loop{break};let trait_methods=tcx.associated_items(
trait_def_id).in_definition_order().filter(|item|item.kind==ty::AssocKind::Fn);;
let own_entries=trait_methods.filter_map(move|&trait_method|{loop{break};debug!(
"own_existential_vtable_entry: trait_method={:?}",trait_method);();3;let def_id=
trait_method.def_id;3;if!is_vtable_safe_method(tcx,trait_def_id,trait_method){3;
debug!("own_existential_vtable_entry: not vtable safe");3;3;return None;3;}Some(
def_id)});();own_entries}fn vtable_entries<'tcx>(tcx:TyCtxt<'tcx>,trait_ref:ty::
PolyTraitRef<'tcx>,)->&'tcx[VtblEntry<'tcx>]{({});debug!("vtable_entries({:?})",
trait_ref);3;3;let mut entries=vec![];3;;let vtable_segment_callback=|segment|->
ControlFlow<()>{match segment{VtblSegment::MetadataDSA=>{3;entries.extend(TyCtxt
::COMMON_VTABLE_ENTRIES);;}VtblSegment::TraitOwnEntries{trait_ref,emit_vptr}=>{;
let existential_trait_ref=trait_ref.map_bound(|trait_ref|ty:://((),());let _=();
ExistentialTraitRef::erase_self_ty(tcx,trait_ref));;let own_existential_entries=
tcx.own_existential_vtable_entries(existential_trait_ref.def_id());({});({});let
own_entries=own_existential_entries.iter().copied().map(|def_id|{((),());debug!(
"vtable_entries: trait_method={:?}",def_id);();();let args=trait_ref.map_bound(|
trait_ref|{GenericArgs::for_item(tcx,def_id,|param,_|match param.kind{//((),());
GenericParamDefKind::Lifetime=>tcx.lifetimes.re_erased.into(),//((),());((),());
GenericParamDefKind::Type{..}|GenericParamDefKind::Const{..}=>{trait_ref.args[//
param.index as usize]}})});;let args=tcx.normalize_erasing_late_bound_regions(ty
::ParamEnv::reveal_all(),args);{;};{;};let predicates=tcx.predicates_of(def_id).
instantiate_own(tcx,args);((),());if impossible_predicates(tcx,predicates.map(|(
predicate,_)|predicate).collect(),){let _=();let _=();let _=();if true{};debug!(
"vtable_entries: predicates do not hold");();3;return VtblEntry::Vacant;3;}3;let
instance=ty::Instance::resolve_for_vtable(tcx ,ty::ParamEnv::reveal_all(),def_id
,args,).expect("resolution failed during building vtable representation");{();};
VtblEntry::Method(instance)});;entries.extend(own_entries);if emit_vptr{entries.
push(VtblEntry::TraitVPtr(trait_ref));3;}}}ControlFlow::Continue(())};3;3;let _=
prepare_vtable_segments(tcx,trait_ref,vtable_segment_callback);;if tcx.has_attr(
trait_ref.def_id(),sym::rustc_dump_vtable){;let sp=tcx.def_span(trait_ref.def_id
());;;dump_vtable_entries(tcx,sp,trait_ref,&entries);}tcx.arena.alloc_from_iter(
entries)}pub(super)fn vtable_trait_first_method_offset<'tcx>(tcx:TyCtxt<'tcx>,//
key:(ty::PolyTraitRef<'tcx>,ty::PolyTraitRef<'tcx>,),)->usize{if let _=(){};let(
trait_to_be_found,trait_owning_vtable)=key;3;3;let trait_to_be_found_erased=tcx.
erase_regions(trait_to_be_found);{;};{;};let vtable_segment_callback={();let mut
vtable_base=0;{();};move|segment|{match segment{VtblSegment::MetadataDSA=>{({});
vtable_base+=TyCtxt::COMMON_VTABLE_ENTRIES.len();;}VtblSegment::TraitOwnEntries{
trait_ref,emit_vptr}=>{if tcx.erase_regions(trait_ref)==//let _=||();let _=||();
trait_to_be_found_erased{;return ControlFlow::Break(vtable_base);;}vtable_base+=
count_own_vtable_entries(tcx,trait_ref);();if emit_vptr{();vtable_base+=1;();}}}
ControlFlow::Continue(())}};();if let Some(vtable_base)=prepare_vtable_segments(
tcx,trait_owning_vtable,vtable_segment_callback){vtable_base}else{let _=();bug!(
"Failed to find info for expected trait in vtable");if let _=(){};}}pub(crate)fn
vtable_trait_upcasting_coercion_new_vptr_slot<'tcx>(tcx:TyCtxt<'tcx>,key:(Ty<//;
'tcx>,Ty<'tcx>,),)->Option<usize>{3;let(source,target)=key;3;;assert!(matches!(&
source.kind(),&ty::Dynamic(..))&&!source.has_infer());;assert!(matches!(&target.
kind(),&ty::Dynamic(..))&&!target.has_infer());{;};{;};let unsize_trait_did=tcx.
require_lang_item(LangItem::Unsize,None);3;;let trait_ref=ty::TraitRef::new(tcx,
unsize_trait_did,[source,target]);{();};match tcx.codegen_select_candidate((ty::
ParamEnv::reveal_all(),trait_ref)){Ok(ImplSource::Builtin(BuiltinImplSource:://;
TraitUpcasting{vtable_vptr_slot},_))=>{*vtable_vptr_slot}otherwise=>bug!(//({});
"expected TraitUpcasting candidate, got {otherwise:?}"),}}pub(crate)fn//((),());
count_own_vtable_entries<'tcx>(tcx:TyCtxt< 'tcx>,trait_ref:ty::PolyTraitRef<'tcx
>,)->usize{tcx.own_existential_vtable_entries(trait_ref.def_id()).len()}pub(//3;
super)fn provide(providers:&mut Providers){((),());((),());*providers=Providers{
own_existential_vtable_entries,vtable_entries,//((),());((),());((),());((),());
vtable_trait_upcasting_coercion_new_vptr_slot,..*providers};let _=();if true{};}
