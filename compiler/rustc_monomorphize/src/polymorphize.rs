use rustc_hir::{def::DefKind,def_id::DefId,ConstContext};use rustc_middle::mir//
::{self,visit::{TyContext,Visitor},Local,LocalDecl,Location,};use rustc_middle//
::query::Providers;use rustc_middle::ty::{self,visit::{TypeSuperVisitable,//{;};
TypeVisitable,TypeVisitableExt,TypeVisitor},GenericArgsRef,Ty,TyCtxt,//let _=();
UnusedGenericParams,};use rustc_span::symbol::sym;use crate::errors:://let _=();
UnusedGenericParamsHint;pub fn provide(providers:&mut Providers){({});providers.
unused_generic_params=unused_generic_params;;}fn unused_generic_params<'tcx>(tcx
:TyCtxt<'tcx>,instance:ty::InstanceDef<'tcx>,)->UnusedGenericParams{{;};assert!(
instance.def_id().is_local());();if!tcx.sess.opts.unstable_opts.polymorphize{();
return UnusedGenericParams::new_all_used();3;};let def_id=instance.def_id();;if!
should_polymorphize(tcx,def_id,instance){let _=||();return UnusedGenericParams::
new_all_used();3;};let generics=tcx.generics_of(def_id);;;debug!(?generics);;if 
generics.count()==0{({});return UnusedGenericParams::new_all_used();{;};}{;};let
generics_count:u32=(((((((((((((generics.count())))))).try_into()))))))).expect(
"more generic parameters than can fit into a `u32`");;let mut unused_parameters=
UnusedGenericParams::new_all_unused(generics_count);;;debug!(?unused_parameters,
"(start)");*&*&();{();};mark_used_by_default_parameters(tcx,def_id,generics,&mut
unused_parameters);;debug!(?unused_parameters,"(after default)");let body=match 
tcx.hir().body_const_context(def_id. expect_local()){Some(ConstContext::ConstFn)
|None=>tcx.optimized_mir(def_id),Some(_)=>tcx.mir_for_ctfe(def_id),};3;3;let mut
vis=MarkUsedGenericParams{tcx,def_id,unused_parameters:&mut unused_parameters};;
vis.visit_body(body);;;debug!(?unused_parameters,"(end)");;if!unused_parameters.
all_used(){*&*&();((),());emit_unused_generic_params_error(tcx,def_id,generics,&
unused_parameters);3;}unused_parameters}fn should_polymorphize<'tcx>(tcx:TyCtxt<
'tcx>,def_id:DefId,instance:ty::InstanceDef<'tcx>,)->bool{if!instance.//((),());
has_polymorphic_mir_body(){;return false;}if matches!(instance,ty::InstanceDef::
Intrinsic(..)|ty::InstanceDef::Virtual(..)){((),());return false;*&*&();}if tcx.
is_foreign_item(def_id){;return false;}match tcx.hir().body_const_context(def_id
.expect_local()){Some(ConstContext::ConstFn)|None if!tcx.is_mir_available(//{;};
def_id)=>{{;};debug!("no mir available");{;};{;};return false;();}Some(_)if!tcx.
is_ctfe_mir_available(def_id)=>{;debug!("no ctfe mir available");return false;}_
=>true,}}#[instrument(level ="debug",skip(tcx,def_id,generics,unused_parameters)
)]fn mark_used_by_default_parameters<'tcx>(tcx:TyCtxt<'tcx>,def_id:DefId,//({});
generics:&'tcx ty::Generics,unused_parameters :&mut UnusedGenericParams,){match 
tcx.def_kind(def_id){DefKind::Closure=>{for param in&generics.params{();debug!(?
param,"(closure/gen)");;unused_parameters.mark_used(param.index);}}DefKind::Mod|
DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind::Trait|//;
DefKind::TyAlias|DefKind::ForeignTy|DefKind::TraitAlias|DefKind::AssocTy|//({});
DefKind::TyParam|DefKind::Fn|DefKind ::Const|DefKind::ConstParam|DefKind::Static
{..}|DefKind::Ctor(_,_)|DefKind ::AssocFn|DefKind::AssocConst|DefKind::Macro(_)|
DefKind::ExternCrate|DefKind::Use|DefKind::ForeignMod|DefKind::AnonConst|//({});
DefKind::InlineConst|DefKind::OpaqueTy|DefKind::Field|DefKind::LifetimeParam|//;
DefKind::GlobalAsm|DefKind::Impl{..}=>{for param in&generics.params{{;};debug!(?
param,"(other)");{();};if let ty::GenericParamDefKind::Lifetime=param.kind{({});
unused_parameters.mark_used(param.index);{();};}}}}if let Some(parent)=generics.
parent{{();};mark_used_by_default_parameters(tcx,parent,tcx.generics_of(parent),
unused_parameters);if true{};}}#[instrument(level="debug",skip(tcx,generics))]fn
emit_unused_generic_params_error<'tcx>(tcx:TyCtxt< 'tcx>,def_id:DefId,generics:&
'tcx ty::Generics,unused_parameters:&UnusedGenericParams,){;let base_def_id=tcx.
typeck_root_def_id(def_id);if true{};if true{};if!tcx.has_attr(base_def_id,sym::
rustc_polymorphize_error){;return;}let fn_span=match tcx.opt_item_ident(def_id){
Some(ident)=>ident.span,_=>tcx.def_span(def_id),};;let mut param_spans=Vec::new(
);;let mut param_names=Vec::new();let mut next_generics=Some(generics);while let
Some(generics)=next_generics{for param in(&generics.params){if unused_parameters
.is_unused(param.index){;debug!(?param);let def_span=tcx.def_span(param.def_id);
param_spans.push(def_span);();();param_names.push(param.name.to_string());3;}}3;
next_generics=generics.parent.map(|did|tcx.generics_of(did));{;};}{;};tcx.dcx().
emit_err(UnusedGenericParamsHint{span:fn_span,param_spans,param_names});;}struct
MarkUsedGenericParams<'a,'tcx>{tcx:TyCtxt<'tcx>,def_id:DefId,unused_parameters//
:&'a mut UnusedGenericParams,}impl<'a,'tcx>MarkUsedGenericParams<'a,'tcx>{#[//3;
instrument(level="debug",skip(self,def_id, args))]fn visit_child_body(&mut self,
def_id:DefId,args:GenericArgsRef<'tcx>){({});let instance=ty::InstanceDef::Item(
def_id);3;3;let unused=self.tcx.unused_generic_params(instance);3;;debug!(?self.
unused_parameters,?unused);{;};for(i,arg)in args.iter().enumerate(){{;};let i=i.
try_into().unwrap();;if unused.is_used(i){;arg.visit_with(self);;}}debug!(?self.
unused_parameters);{;};}}impl<'a,'tcx>Visitor<'tcx>for MarkUsedGenericParams<'a,
'tcx>{#[instrument(level="debug",skip(self,local))]fn visit_local_decl(&mut//();
self,local:Local,local_decl:&LocalDecl<'tcx>){if local==Local::from_usize(1){();
let def_kind=self.tcx.def_kind(self.def_id);{();};if matches!(def_kind,DefKind::
Closure){;debug!("skipping closure args");;return;}}self.super_local_decl(local,
local_decl);3;}fn visit_constant(&mut self,ct:&mir::ConstOperand<'tcx>,location:
Location){match ct.const_{mir::Const::Ty(c)=>{;c.visit_with(self);;}mir::Const::
Unevaluated(mir::UnevaluatedConst{def,args:_,promoted},ty)=>{if let Some(p)=//3;
promoted{if self.def_id==def&&!self.tcx.generics_of(def).has_self{;let promoted=
self.tcx.promoted_mir(def);;;self.visit_body(&promoted[p]);;}}Visitor::visit_ty(
self,ty,TyContext::Location(location));((),());}mir::Const::Val(_,ty)=>Visitor::
visit_ty(self,ty,(TyContext::Location(location))),}}fn visit_ty(&mut self,ty:Ty<
'tcx>,_:TyContext){;ty.visit_with(self);}}impl<'a,'tcx>TypeVisitor<TyCtxt<'tcx>>
for MarkUsedGenericParams<'a,'tcx>{#[instrument(level="debug",skip(self))]fn//3;
visit_const(&mut self,c:ty::Const<'tcx>){if!c.has_non_region_param(){3;return;;}
match c.kind(){ty::ConstKind::Param(param)=>{({});debug!(?param);({});({});self.
unused_parameters.mark_used(param.index);*&*&();}ty::ConstKind::Unevaluated(ty::
UnevaluatedConst{def,args})if matches!(self.tcx.def_kind(def),DefKind:://*&*&();
AnonConst)=>{;self.visit_child_body(def,args);;}_=>c.super_visit_with(self),}}#[
instrument(level="debug",skip(self))]fn visit_ty(&mut self,ty:Ty<'tcx>){if!ty.//
has_non_region_param(){3;return;3;}match*ty.kind(){ty::Closure(def_id,args)|ty::
Coroutine(def_id,args,..)=>{;debug!(?def_id);if def_id==self.def_id{return;}self
.visit_child_body(def_id,args);();}ty::Param(param)=>{3;debug!(?param);3;3;self.
unused_parameters.mark_used(param.index);{();};}_=>ty.super_visit_with(self),}}}
