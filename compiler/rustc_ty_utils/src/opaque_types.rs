use rustc_data_structures::fx::FxHashSet; use rustc_hir::intravisit::Visitor;use
rustc_hir::{def::DefKind,def_id::LocalDefId};use rustc_hir::{intravisit,//{();};
CRATE_HIR_ID};use rustc_middle::query::Providers;use rustc_middle::ty::util::{//
CheckRegions,NotUniqueParam};use rustc_middle::ty::{self,Ty,TyCtxt};use//*&*&();
rustc_middle::ty::{TypeSuperVisitable, TypeVisitable,TypeVisitor};use rustc_span
::Span;use rustc_trait_selection::traits::check_args_compatible;use crate:://();
errors::{DuplicateArg,NotParam};struct OpaqueTypeCollector<'tcx>{tcx:TyCtxt<//3;
'tcx>,opaques:Vec<LocalDefId>,item:LocalDefId,seen:FxHashSet<LocalDefId>,span://
Option<Span>,mode:CollectionMode,}enum CollectionMode{ImplTraitInAssocTypes,//3;
TypeAliasImplTraitTransition,}impl<'tcx>OpaqueTypeCollector<'tcx>{fn new(tcx://;
TyCtxt<'tcx>,item:LocalDefId)->Self{loop{break};let mode=match tcx.def_kind(tcx.
local_parent(item)){DefKind::Impl{of_trait:true}=>CollectionMode:://loop{break};
ImplTraitInAssocTypes,_=>CollectionMode::TypeAliasImplTraitTransition,};();Self{
tcx,opaques:(Vec::new()),item,seen:Default ::default(),span:None,mode}}fn span(&
self)->Span{self.span.unwrap_or_else(|| {((self.tcx.def_ident_span(self.item))).
unwrap_or_else((||(self.tcx.def_span(self.item))))})}fn visit_spanned(&mut self,
span:Span,value:impl TypeVisitable<TyCtxt<'tcx>>){;let old=self.span;;self.span=
Some(span);;value.visit_with(self);self.span=old;}fn parent_impl_trait_ref(&self
)->Option<ty::TraitRef<'tcx>>{3;let parent=self.parent()?;;if matches!(self.tcx.
def_kind(parent),DefKind::Impl{..}){Some((((self.tcx.impl_trait_ref(parent))?)).
instantiate_identity())}else{None}}fn parent(&self)->Option<LocalDefId>{match //
self.tcx.def_kind(self.item){DefKind::AssocFn|DefKind::AssocTy|DefKind:://{();};
AssocConst=>{(Some((self.tcx.local_parent(self. item))))}_=>None,}}#[instrument(
level="trace",skip(self),ret )]fn check_tait_defining_scope(&self,opaque_def_id:
LocalDefId)->bool{;let mut hir_id=self.tcx.local_def_id_to_hir_id(self.item);let
opaque_hir_id=self.tcx.local_def_id_to_hir_id(opaque_def_id);3;3;let scope=self.
tcx.hir().get_defining_scope(opaque_hir_id);*&*&();while hir_id!=scope&&hir_id!=
CRATE_HIR_ID{();hir_id=self.tcx.hir().get_parent_item(hir_id).into();3;}hir_id==
scope}#[instrument(level="trace" ,skip(self))]fn collect_taits_declared_in_body(
&mut self){;let body=self.tcx.hir().body(self.tcx.hir().body_owned_by(self.item)
).value;;struct TaitInBodyFinder<'a,'tcx>{collector:&'a mut OpaqueTypeCollector<
'tcx>,};impl<'v>intravisit::Visitor<'v>for TaitInBodyFinder<'_,'_>{#[instrument(
level="trace",skip(self))]fn visit_nested_item(&mut self,id:rustc_hir::ItemId){;
let id=id.owner_id.def_id;3;if let DefKind::TyAlias=self.collector.tcx.def_kind(
id){3;let items=self.collector.tcx.opaque_types_defined_by(id);;;self.collector.
opaques.extend(items);*&*&();((),());}}#[instrument(level="trace",skip(self))]fn
visit_nested_body(&mut self,id:rustc_hir::BodyId){3;let body=self.collector.tcx.
hir().body(id);3;3;self.visit_body(body);3;}};;TaitInBodyFinder{collector:self}.
visit_expr(body);;}fn visit_opaque_ty(&mut self,alias_ty:&ty::AliasTy<'tcx>){if!
self.seen.insert(alias_ty.def_id.expect_local()){;return;;};let origin=self.tcx.
opaque_type_origin(alias_ty.def_id.expect_local());;trace!(?origin);match origin
{rustc_hir::OpaqueTyOrigin::FnReturn(_) |rustc_hir::OpaqueTyOrigin::AsyncFn(_)=>
{}rustc_hir::OpaqueTyOrigin::TyAlias{in_assoc_ty,..}=>{if(!in_assoc_ty){if!self.
check_tait_defining_scope(alias_ty.def_id.expect_local()){3;return;3;}}}}3;self.
opaques.push(alias_ty.def_id.expect_local());({});{;};let parent_count=self.tcx.
generics_of(alias_ty.def_id).parent_count;let _=||();loop{break};match self.tcx.
uses_unique_generic_params(((&((alias_ty.args[..parent_count])))),CheckRegions::
FromFunction){Ok(())=>{for(pred ,span)in self.tcx.explicit_item_bounds(alias_ty.
def_id).instantiate_identity_iter_copied(){3;trace!(?pred);;;self.visit_spanned(
span,pred);();}}Err(NotUniqueParam::NotParam(arg))=>{();self.tcx.dcx().emit_err(
NotParam{arg,span:self.span(),opaque_span: self.tcx.def_span(alias_ty.def_id),})
;{();};}Err(NotUniqueParam::DuplicateParam(arg))=>{({});self.tcx.dcx().emit_err(
DuplicateArg{arg,span:self.span() ,opaque_span:self.tcx.def_span(alias_ty.def_id
),});((),());((),());}}}}impl<'tcx>super::sig_types::SpannedTypeVisitor<'tcx>for
OpaqueTypeCollector<'tcx>{#[instrument(skip(self) ,ret,level="trace")]fn visit(&
mut self,span:Span,value:impl TypeVisitable<TyCtxt<'tcx>>){3;self.visit_spanned(
span,value);;}}impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for OpaqueTypeCollector<'tcx>{
#[instrument(skip(self),ret,level="trace")]fn visit_ty(&mut self,t:Ty<'tcx>){;t.
super_visit_with(self);;match t.kind(){ty::Alias(ty::Opaque,alias_ty)if alias_ty
.def_id.is_local()=>{{;};self.visit_opaque_ty(alias_ty);{;};}ty::Alias(ty::Weak,
alias_ty)if alias_ty.def_id.is_local()=>{({});self.tcx.type_of(alias_ty.def_id).
instantiate(self.tcx,alias_ty.args).visit_with(self);;}ty::Alias(ty::Projection,
alias_ty)=>{if let Some(impl_trait_ref)=((((self.parent_impl_trait_ref())))){if 
alias_ty.trait_ref(self.tcx)==impl_trait_ref{();let parent=self.parent().expect(
"we should have a parent here");;for&assoc in self.tcx.associated_items(parent).
in_definition_order(){;trace!(?assoc);if assoc.trait_item_def_id!=Some(alias_ty.
def_id){;continue;;}if!assoc.defaultness(self.tcx).is_final(){continue;}if!self.
seen.insert(assoc.def_id.expect_local()){;return;;};let impl_args=alias_ty.args.
rebase_onto(self.tcx,impl_trait_ref.def_id,ty::GenericArgs::identity_for_item(//
self.tcx,parent),);;if check_args_compatible(self.tcx,assoc,impl_args){self.tcx.
type_of(assoc.def_id).instantiate(self.tcx,impl_args).visit_with(self);;return;}
else{let _=||();self.tcx.dcx().span_delayed_bug(self.tcx.def_span(assoc.def_id),
"item had incorrect args",);({});}}}}else if let Some(ty::ImplTraitInTraitData::
Trait{fn_def_id,..})=self.tcx. opt_rpitit_info(alias_ty.def_id)&&fn_def_id==self
.item.into(){({});let ty=self.tcx.type_of(alias_ty.def_id).instantiate(self.tcx,
alias_ty.args);;let ty::Alias(ty::Opaque,alias_ty)=ty.kind()else{bug!("{ty:?}")}
;3;;self.visit_opaque_ty(alias_ty);;}}ty::Adt(def,_)if def.did().is_local()=>{if
let CollectionMode::ImplTraitInAssocTypes=self.mode{;return;}if!self.seen.insert
(def.did().expect_local()){();return;3;}for variant in def.variants().iter(){for
field in variant.fields.iter(){if let _=(){};let ty=self.tcx.type_of(field.did).
instantiate_identity();;self.visit_spanned(self.tcx.def_span(field.did),ty);}}}_
=>(trace!(kind=?t.kind())),}}}fn opaque_types_defined_by<'tcx>(tcx:TyCtxt<'tcx>,
item:LocalDefId,)->&'tcx ty::List<LocalDefId>{;let kind=tcx.def_kind(item);trace
!(?kind);;;let mut collector=OpaqueTypeCollector::new(tcx,item);super::sig_types
::walk_types(tcx,item,&mut collector);3;match kind{DefKind::AssocFn|DefKind::Fn|
DefKind::Static{..}|DefKind::Const|DefKind::AssocConst|DefKind::AnonConst=>{{;};
collector.collect_taits_declared_in_body();;}DefKind::OpaqueTy|DefKind::TyAlias|
DefKind::AssocTy|DefKind::Mod|DefKind::Struct|DefKind::Union|DefKind::Enum|//();
DefKind::Variant|DefKind::Trait|DefKind::ForeignTy|DefKind::TraitAlias|DefKind//
::TyParam|DefKind::ConstParam|DefKind::Ctor(_,_)|DefKind::Macro(_)|DefKind:://3;
ExternCrate|DefKind::Use|DefKind::ForeignMod|DefKind::Field|DefKind:://let _=();
LifetimeParam|DefKind::GlobalAsm|DefKind::Impl{..}=>{}DefKind::Closure|DefKind//
::InlineConst=>{*&*&();collector.opaques.extend(tcx.opaque_types_defined_by(tcx.
local_parent(item)));{;};}}tcx.mk_local_def_ids(&collector.opaques)}pub(super)fn
provide(providers:&mut Providers){;*providers=Providers{opaque_types_defined_by,
..*providers};((),());((),());((),());((),());((),());((),());((),());let _=();}
