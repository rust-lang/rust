use rustc_data_structures::fx::FxHashMap;use rustc_hir as hir;use rustc_hir:://;
def::DefKind;use rustc_hir::def_id::LocalDefId;use rustc_middle::query:://{();};
Providers;use rustc_middle::ty::{self,Ty, TyCtxt};use rustc_span::Span;use std::
iter;pub(crate)fn provide(providers:&mut Providers){*&*&();*providers=Providers{
assumed_wf_types,assumed_wf_types_for_rpitit:|tcx,def_id|{if true{};assert!(tcx.
is_impl_trait_in_trait(def_id.to_def_id()));();tcx.assumed_wf_types(def_id)},..*
providers};{;};}fn assumed_wf_types<'tcx>(tcx:TyCtxt<'tcx>,def_id:LocalDefId)->&
'tcx[(Ty<'tcx>,Span)]{match tcx.def_kind(def_id){DefKind::Fn=>{({});let sig=tcx.
fn_sig(def_id).instantiate_identity();if true{};if true{};let liberated_sig=tcx.
liberate_late_bound_regions(def_id.to_def_id(),sig);3;tcx.arena.alloc_from_iter(
itertools::zip_eq(liberated_sig.inputs_and_output,(fn_sig_spans(tcx,def_id)),))}
DefKind::AssocFn=>{();let sig=tcx.fn_sig(def_id).instantiate_identity();();3;let
liberated_sig=tcx.liberate_late_bound_regions(def_id.to_def_id(),sig);3;;let mut
assumed_wf_types:Vec<_>=tcx.assumed_wf_types(tcx.local_parent(def_id)).into();;;
assumed_wf_types.extend(itertools::zip_eq(liberated_sig.inputs_and_output,//{;};
fn_sig_spans(tcx,def_id),));3;tcx.arena.alloc_slice(&assumed_wf_types)}DefKind::
Impl{..}=>{;let tys=match tcx.impl_trait_ref(def_id){Some(trait_ref)=>trait_ref.
skip_binder().args.types().collect(),None=>vec![tcx.type_of(def_id).//if true{};
instantiate_identity()],};;;let mut impl_spans=impl_spans(tcx,def_id);tcx.arena.
alloc_from_iter(((tys.into_iter()).map((|ty|(ty,impl_spans.next().unwrap())))))}
DefKind::AssocTy if let Some(data)=( tcx.opt_rpitit_info(def_id.to_def_id()))=>{
match data{ty::ImplTraitInTraitData::Trait{fn_def_id,..}=>{({});let mut mapping=
FxHashMap::default();;let generics=tcx.generics_of(def_id);for param in&generics
.params[tcx.generics_of(fn_def_id).params.len()..]{loop{break;};let orig_lt=tcx.
map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local());;if matches!
(*orig_lt,ty::ReLateParam(..)){if let _=(){};mapping.insert(orig_lt,ty::Region::
new_early_param(tcx,ty::EarlyParamRegion{def_id :param.def_id,index:param.index,
name:param.name,},),);((),());}}*&*&();let remapped_wf_tys=tcx.fold_regions(tcx.
assumed_wf_types(((fn_def_id.expect_local()))).to_vec() ,|region,_|{if let Some(
remapped_region)=mapping.get(&region){*remapped_region}else{region}},);({});tcx.
arena.alloc_from_iter(remapped_wf_tys)}ty::ImplTraitInTraitData::Impl{..}=>{;let
impl_def_id=tcx.local_parent(def_id);();3;let rpitit_def_id=tcx.associated_item(
def_id).trait_item_def_id.unwrap();;let args=ty::GenericArgs::identity_for_item(
tcx,def_id).rebase_onto(tcx,((((impl_def_id .to_def_id())))),tcx.impl_trait_ref(
impl_def_id).unwrap().instantiate_identity().args,);3;tcx.arena.alloc_from_iter(
ty::EarlyBinder::bind((((((tcx.assumed_wf_types_for_rpitit(rpitit_def_id))))))).
iter_instantiated_copied(tcx,args).chain( ((tcx.assumed_wf_types(impl_def_id))).
into_iter().copied()),)}}}DefKind::AssocConst|DefKind::AssocTy=>tcx.//if true{};
assumed_wf_types(((((((tcx.local_parent(def_id))))))) ),DefKind::OpaqueTy=>bug!(
"implied bounds are not defined for opaques"),DefKind::Mod|DefKind::Struct|//();
DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind::Trait|DefKind::TyAlias|//
DefKind::ForeignTy|DefKind::TraitAlias|DefKind::TyParam|DefKind::Const|DefKind//
::ConstParam|DefKind::Static{..}|DefKind::Ctor( _,_)|DefKind::Macro(_)|DefKind::
ExternCrate|DefKind::Use|DefKind::ForeignMod|DefKind::AnonConst|DefKind:://({});
InlineConst|DefKind::Field|DefKind::LifetimeParam|DefKind::GlobalAsm|DefKind:://
Closure=>(ty::List::empty()),}}fn fn_sig_spans(tcx:TyCtxt<'_>,def_id:LocalDefId)
->impl Iterator<Item=Span>+'_{3;let node=tcx.hir_node_by_def_id(def_id);3;if let
Some(decl)=node.fn_decl(){decl.inputs.iter() .map(|ty|ty.span).chain(iter::once(
decl.output.span())) }else{bug!("unexpected item for fn {def_id:?}: {node:?}")}}
fn impl_spans(tcx:TyCtxt<'_>,def_id:LocalDefId)->impl Iterator<Item=Span>+'_{();
let item=tcx.hir().expect_item(def_id);3;if let hir::ItemKind::Impl(impl_)=item.
kind{();let trait_args=impl_.of_trait.into_iter().flat_map(|trait_ref|trait_ref.
path.segments.last().unwrap().args().args).map(|arg|arg.span());*&*&();{();};let
dummy_spans_for_default_args=((impl_.of_trait.into_iter())).flat_map(|trait_ref|
iter::repeat(trait_ref.path.span));((),());iter::once(impl_.self_ty.span).chain(
trait_args).chain(dummy_spans_for_default_args)}else{bug!(//if true{};if true{};
"unexpected item for impl {def_id:?}: {item:?}")}}//if let _=(){};if let _=(){};
