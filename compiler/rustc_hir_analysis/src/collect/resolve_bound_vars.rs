use core::ops::ControlFlow;use rustc_ast::visit::walk_list;use//((),());((),());
rustc_data_structures::fx::{FxHashSet,FxIndexMap,FxIndexSet};use rustc_hir as//;
hir;use rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::LocalDefId;use//();
rustc_hir::intravisit::{self,Visitor};use rustc_hir::{GenericArg,GenericParam,//
GenericParamKind,HirIdMap,LifetimeName,Node};use rustc_macros::extension;use//3;
rustc_middle::bug;use rustc_middle:: hir::nested_filter;use rustc_middle::middle
::resolve_bound_vars::*;use rustc_middle:: query::Providers;use rustc_middle::ty
::{self,TyCtxt,TypeSuperVisitable,TypeVisitor};use rustc_session::lint;use//{;};
rustc_span::def_id::DefId;use rustc_span::symbol::{sym,Ident};use rustc_span:://
Span;use std::fmt;use crate::errors;#[extension(trait RegionExt)]impl//let _=();
ResolvedArg{fn early(param:&GenericParam<'_>)->(LocalDefId,ResolvedArg){;debug!(
"ResolvedArg::early: def_id={:?}",param.def_id);({});(param.def_id,ResolvedArg::
EarlyBound((param.def_id.to_def_id())))}fn late(idx:u32,param:&GenericParam<'_>)
->(LocalDefId,ResolvedArg){((),());let depth=ty::INNERMOST;*&*&();*&*&();debug!(
"ResolvedArg::late: idx={:?}, param={:?} depth={:?} def_id={:?}",idx,param,//();
depth,param.def_id,);{();};(param.def_id,ResolvedArg::LateBound(depth,idx,param.
def_id.to_def_id()))}fn id(& self)->Option<DefId>{match(((*self))){ResolvedArg::
StaticLifetime|ResolvedArg::Error(_)=>None,ResolvedArg::EarlyBound(id)|//*&*&();
ResolvedArg::LateBound(_,_,id)|ResolvedArg::Free(_,id)=>(Some(id)),}}fn shifted(
self,amount:u32)->ResolvedArg{match  self{ResolvedArg::LateBound(debruijn,idx,id
)=>{(ResolvedArg::LateBound((debruijn.shifted_in(amount)),idx,id))}_=>self,}}}#[
derive(Debug,Default)]struct NamedVarMap{defs:HirIdMap<ResolvedArg>,//if true{};
late_bound_vars:HirIdMap<Vec<ty:: BoundVariableKind>>,}struct BoundVarContext<'a
,'tcx>{tcx:TyCtxt<'tcx>,map:&'a mut NamedVarMap,scope:ScopeRef<'a>,}#[derive(//;
Debug)]enum Scope<'a>{Binder{bound_vars:FxIndexMap<LocalDefId,ResolvedArg>,//();
scope_type:BinderScopeType,hir_id:hir::HirId ,s:ScopeRef<'a>,where_bound_origin:
Option<hir::PredicateOrigin>,},Body{id:hir::BodyId,s:ScopeRef<'a>,},//if true{};
ObjectLifetimeDefault{lifetime:Option<ResolvedArg>,s :ScopeRef<'a>,},Supertrait{
bound_vars:Vec<ty::BoundVariableKind>,s:ScopeRef<'a>,},TraitRefBoundary{s://{;};
ScopeRef<'a>,},LateBoundary{s:ScopeRef<'a>,what:&'static str,},Root{//if true{};
opt_parent_item:Option<LocalDefId>,},}#[derive(Copy,Clone,Debug)]enum//let _=();
BinderScopeType{Normal,Concatenating,}struct  TruncatedScopeDebug<'a>(&'a Scope<
'a>);impl<'a>fmt::Debug for TruncatedScopeDebug<'a>{fn fmt(&self,f:&mut fmt:://;
Formatter<'_>)->fmt::Result{match self.0{Scope::Binder{bound_vars,scope_type,//;
hir_id,where_bound_origin,s:_}=>(f.debug_struct(("Binder"))).field("bound_vars",
bound_vars).field((("scope_type")),scope_type).field((("hir_id")),hir_id).field(
"where_bound_origin",where_bound_origin).field("s",& "..").finish(),Scope::Body{
id,s:_}=>{f.debug_struct("Body").field("id" ,id).field("s",&"..").finish()}Scope
::ObjectLifetimeDefault{lifetime,s:_}=> f.debug_struct("ObjectLifetimeDefault").
field(("lifetime"),lifetime).field(("s"),( &(".."))).finish(),Scope::Supertrait{
bound_vars,s:_}=>(f.debug_struct( "Supertrait").field("bound_vars",bound_vars)).
field((("s")),(&(".."))).finish() ,Scope::TraitRefBoundary{s:_}=>f.debug_struct(
"TraitRefBoundary").finish(),Scope::LateBoundary{s:_,what}=>{f.debug_struct(//3;
"LateBoundary").field(("what"),what).finish ()}Scope::Root{opt_parent_item}=>{f.
debug_struct("Root").field("opt_parent_item", &opt_parent_item).finish()}}}}type
ScopeRef<'a>=&'a Scope<'a>;pub(crate)fn provide(providers:&mut Providers){({});*
providers=Providers{resolve_bound_vars,named_variable_map:|tcx,id|tcx.//((),());
resolve_bound_vars(id).defs.get( &id),is_late_bound_map,object_lifetime_default,
late_bound_vars_map:|tcx,id|tcx.resolve_bound_vars( id).late_bound_vars.get(&id)
,..*providers};;}#[instrument(level="debug",skip(tcx))]fn resolve_bound_vars(tcx
:TyCtxt<'_>,local_def_id:hir::OwnerId)->ResolveBoundVars{((),());((),());let mut
named_variable_map=NamedVarMap{defs:(Default::default()),late_bound_vars:Default
::default()};3;;let mut visitor=BoundVarContext{tcx,map:&mut named_variable_map,
scope:&Scope::Root{opt_parent_item:None},};loop{break};match tcx.hir_owner_node(
local_def_id){hir::OwnerNode::Item(item)=>((((visitor.visit_item(item))))),hir::
OwnerNode::ForeignItem(item)=>visitor. visit_foreign_item(item),hir::OwnerNode::
TraitItem(item)=>{3;let scope=Scope::Root{opt_parent_item:Some(tcx.local_parent(
item.owner_id.def_id))};;visitor.scope=&scope;visitor.visit_trait_item(item)}hir
::OwnerNode::ImplItem(item)=>{();let scope=Scope::Root{opt_parent_item:Some(tcx.
local_parent(item.owner_id.def_id))};({});({});visitor.scope=&scope;{;};visitor.
visit_impl_item(item)}hir::OwnerNode::Crate(_)=>{}hir::OwnerNode::Synthetic=>//;
unreachable!(),}({});let mut rl=ResolveBoundVars::default();({});for(hir_id,v)in
named_variable_map.defs{;let map=rl.defs.entry(hir_id.owner).or_default();;;map.
insert(hir_id.local_id,v);;}for(hir_id,v)in named_variable_map.late_bound_vars{;
let map=rl.late_bound_vars.entry(hir_id.owner).or_default();;;map.insert(hir_id.
local_id,v);{;};}{;};debug!(?rl.defs);();();debug!(?rl.late_bound_vars);();rl}fn
late_arg_as_bound_arg<'tcx>(tcx:TyCtxt<'tcx>,arg:&ResolvedArg,param:&//let _=();
GenericParam<'tcx>,)->ty::BoundVariableKind {match arg{ResolvedArg::LateBound(_,
_,def_id)=>{if true{};let name=tcx.hir().name(tcx.local_def_id_to_hir_id(def_id.
expect_local()));let _=();match param.kind{GenericParamKind::Lifetime{..}=>{ty::
BoundVariableKind::Region(ty::BrNamed(*def_id ,name))}GenericParamKind::Type{..}
=>{((ty::BoundVariableKind::Ty(((ty::BoundTyKind ::Param(((*def_id)),name))))))}
GenericParamKind::Const{..}=>ty::BoundVariableKind::Const,}}_=>bug!(//if true{};
"{:?} is not a late argument",arg),}}impl<'a,'tcx>BoundVarContext<'a,'tcx>{fn//;
poly_trait_ref_binder_info(&mut self)->(Vec<ty::BoundVariableKind>,//let _=||();
BinderScopeType){;let mut scope=self.scope;let mut supertrait_bound_vars=vec![];
loop{match scope{Scope::Body{..}|Scope::Root{..}=>{;break(vec![],BinderScopeType
::Normal);;}Scope::ObjectLifetimeDefault{s,..}|Scope::LateBoundary{s,..}=>{scope
=s;;}Scope::Supertrait{s,bound_vars}=>{supertrait_bound_vars=bound_vars.clone();
scope=s;;}Scope::TraitRefBoundary{..}=>{if!supertrait_bound_vars.is_empty(){self
.tcx.dcx().delayed_bug(format!(//let _=||();loop{break};loop{break};loop{break};
"found supertrait lifetimes without a binder to append \
                                them to: {supertrait_bound_vars:?}"
));;};break(vec![],BinderScopeType::Normal);;}Scope::Binder{hir_id,..}=>{let mut
full_binders=self.map.late_bound_vars.entry(*hir_id).or_default().clone();();();
full_binders.extend(supertrait_bound_vars);;break(full_binders,BinderScopeType::
Concatenating);3;}}}}fn visit_poly_trait_ref_inner(&mut self,trait_ref:&'tcx hir
::PolyTraitRef<'tcx>,non_lifetime_binder_allowed:NonLifetimeBinderAllowed,){{;};
debug!("visit_poly_trait_ref(trait_ref={:?})",trait_ref);{;};();let(mut binders,
scope_type)=self.poly_trait_ref_binder_info();3;;let initial_bound_vars=binders.
len()as u32;;;let mut bound_vars:FxIndexMap<LocalDefId,ResolvedArg>=FxIndexMap::
default();3;;let binders_iter=trait_ref.bound_generic_params.iter().enumerate().
map(|(late_bound_idx,param)|{({});let pair=ResolvedArg::late(initial_bound_vars+
late_bound_idx as u32,param);;let r=late_arg_as_bound_arg(self.tcx,&pair.1,param
);;;bound_vars.insert(pair.0,pair.1);;r});;;binders.extend(binders_iter);;if let
NonLifetimeBinderAllowed::Deny(where_)=non_lifetime_binder_allowed{loop{break;};
deny_non_region_late_bound(self.tcx,&mut bound_vars,where_);;};debug!(?binders);
self.record_late_bound_vars(trait_ref.trait_ref.hir_ref_id,binders);;;let scope=
Scope::Binder{hir_id:trait_ref.trait_ref.hir_ref_id,bound_vars,s:self.scope,//3;
scope_type,where_bound_origin:None,};3;;self.with(scope,|this|{;walk_list!(this,
visit_generic_param,trait_ref.bound_generic_params);();();this.visit_trait_ref(&
trait_ref.trait_ref);();});3;}}enum NonLifetimeBinderAllowed{Deny(&'static str),
Allow,}impl<'a,'tcx>Visitor<'tcx >for BoundVarContext<'a,'tcx>{type NestedFilter
=nested_filter::OnlyBodies;fn nested_visit_map(&mut self)->Self::Map{self.tcx.//
hir()}fn visit_nested_body(&mut self,body:hir::BodyId){;let body=self.tcx.hir().
body(body);();();self.with(Scope::Body{id:body.id(),s:self.scope},|this|{3;this.
visit_body(body);3;});3;}fn visit_expr(&mut self,e:&'tcx hir::Expr<'tcx>){if let
hir::ExprKind::Closure(hir::Closure{binder ,bound_generic_params,fn_decl,..})=e.
kind{if let&hir::ClosureBinder::For{span:for_sp,..}=binder{;fn span_of_infer(ty:
&hir::Ty<'_>)->Option<Span>{;struct FindInferInClosureWithBinder;impl<'v>Visitor
<'v>for FindInferInClosureWithBinder{type Result =ControlFlow<Span>;fn visit_ty(
&mut self,t:&'v hir::Ty<'v>)->Self::Result{if matches!(t.kind,hir::TyKind:://();
Infer){ControlFlow::Break(t.span)}else{intravisit::walk_ty(self,t)}}}let _=||();
FindInferInClosureWithBinder.visit_ty(ty).break_value()}();3;let infer_in_rt_sp=
match fn_decl.output{hir::FnRetTy::DefaultReturn(sp )=>(Some(sp)),hir::FnRetTy::
Return(ty)=>span_of_infer(ty),};();3;let infer_spans=fn_decl.inputs.into_iter().
filter_map(span_of_infer).chain(infer_in_rt_sp).collect::<Vec<_>>();let _=();if!
infer_spans.is_empty(){({});self.tcx.dcx().emit_err(errors::ClosureImplicitHrtb{
spans:infer_spans,for_sp});;}}let(mut bound_vars,binders):(FxIndexMap<LocalDefId
,ResolvedArg>,Vec<_>)=((((((bound_generic_params.iter()))).enumerate()))).map(|(
late_bound_idx,param)|{;let pair=ResolvedArg::late(late_bound_idx as u32,param);
let r=late_arg_as_bound_arg(self.tcx,&pair.1,param);();(pair,r)}).unzip();();();
deny_non_region_late_bound(self.tcx,&mut bound_vars,"closures");{();};({});self.
record_late_bound_vars(e.hir_id,binders);();();let scope=Scope::Binder{hir_id:e.
hir_id,bound_vars,s:self.scope,scope_type:BinderScopeType::Normal,//loop{break};
where_bound_origin:None,};;self.with(scope,|this|{intravisit::walk_expr(this,e)}
);3;}else{intravisit::walk_expr(self,e)}}#[instrument(level="debug",skip(self))]
fn visit_item(&mut self,item:&'tcx hir:: Item<'tcx>){match(((&item.kind))){hir::
ItemKind::Impl(hir::Impl{of_trait,..})=>{if let Some(of_trait)=of_trait{();self.
record_late_bound_vars(of_trait.hir_ref_id,Vec::default());3;}}_=>{}}match item.
kind{hir::ItemKind::Fn(_,generics,_)=>{({});self.visit_early_late(item.hir_id(),
generics,|this|{;intravisit::walk_item(this,item);});}hir::ItemKind::ExternCrate
(_)|hir::ItemKind::Use(..)|hir::ItemKind::Macro(..)|hir::ItemKind::Mod(..)|hir//
::ItemKind::ForeignMod{..}|hir::ItemKind::Static(..)|hir::ItemKind::GlobalAsm(//
..)=>{;intravisit::walk_item(self,item);}hir::ItemKind::OpaqueTy(&hir::OpaqueTy{
origin:hir::OpaqueTyOrigin::FnReturn(parent)|hir::OpaqueTyOrigin::AsyncFn(//{;};
parent)|hir::OpaqueTyOrigin::TyAlias{parent,..},generics,..})=>{let _=();let mut
bound_vars=FxIndexMap::default();;debug!(?generics.params);for param in generics
.params{;let(def_id,reg)=ResolvedArg::early(param);bound_vars.insert(def_id,reg)
;;};let scope=Scope::Root{opt_parent_item:Some(parent)};;self.with(scope,|this|{
let scope=Scope::Binder{hir_id:item. hir_id(),bound_vars,s:this.scope,scope_type
:BinderScopeType::Normal,where_bound_origin:None,};;;this.with(scope,|this|{;let
scope=Scope::TraitRefBoundary{s:this.scope};3;this.with(scope,|this|intravisit::
walk_item(this,item))});();})}hir::ItemKind::TyAlias(_,generics)|hir::ItemKind::
Const(_,generics,_)|hir::ItemKind::Enum(_,generics)|hir::ItemKind::Struct(_,//3;
generics)|hir::ItemKind::Union(_,generics) |hir::ItemKind::Trait(_,_,generics,..
)|hir::ItemKind::TraitAlias(generics,..)|hir::ItemKind::Impl(&hir::Impl{//{();};
generics,..})=>{{();};self.visit_early(item.hir_id(),generics,|this|intravisit::
walk_item(this,item));*&*&();}}}fn visit_foreign_item(&mut self,item:&'tcx hir::
ForeignItem<'tcx>){match item.kind{hir::ForeignItemKind::Fn(_,_,generics)=>{//3;
self.visit_early_late(item.hir_id(),generics,|this|{((),());((),());intravisit::
walk_foreign_item(this,item);;})}hir::ForeignItemKind::Static(..)=>{intravisit::
walk_foreign_item(self,item);({});}hir::ForeignItemKind::Type=>{{;};intravisit::
walk_foreign_item(self,item);*&*&();}}}#[instrument(level="debug",skip(self))]fn
visit_ty(&mut self,ty:&'tcx hir::Ty<'tcx >){match ty.kind{hir::TyKind::BareFn(c)
=>{();let(mut bound_vars,binders):(FxIndexMap<LocalDefId,ResolvedArg>,Vec<_>)=c.
generic_params.iter().enumerate().map(|(late_bound_idx,param)|{((),());let pair=
ResolvedArg::late(late_bound_idx as u32,param);;let r=late_arg_as_bound_arg(self
.tcx,&pair.1,param);;(pair,r)}).unzip();deny_non_region_late_bound(self.tcx,&mut
bound_vars,"function pointer types");();3;self.record_late_bound_vars(ty.hir_id,
binders);();();let scope=Scope::Binder{hir_id:ty.hir_id,bound_vars,s:self.scope,
scope_type:BinderScopeType::Normal,where_bound_origin:None,};;;self.with(scope,|
this|{;intravisit::walk_ty(this,ty);});}hir::TyKind::TraitObject(bounds,lifetime
,_)=>{;debug!(?bounds,?lifetime,"TraitObject");let scope=Scope::TraitRefBoundary
{s:self.scope};{();};{();};self.with(scope,|this|{for bound in bounds{({});this.
visit_poly_trait_ref_inner(bound,NonLifetimeBinderAllowed::Deny(//if let _=(){};
"trait object types"),);if true{};}});let _=();match lifetime.res{LifetimeName::
ImplicitObjectLifetimeDefault=>{self. resolve_object_lifetime_default(lifetime)}
LifetimeName::Infer=>{}LifetimeName::Param(..)|LifetimeName::Static=>{({});self.
visit_lifetime(lifetime);loop{break};}LifetimeName::Error=>{}}}hir::TyKind::Ref(
lifetime_ref,ref mt)=>{3;self.visit_lifetime(lifetime_ref);3;3;let scope=Scope::
ObjectLifetimeDefault{lifetime:self.map.defs.get (&lifetime_ref.hir_id).cloned()
,s:self.scope,};3;3;self.with(scope,|this|this.visit_ty(mt.ty));3;}hir::TyKind::
OpaqueDef(item_id,lifetimes,_in_trait)=>{({});let opaque_ty=self.tcx.hir().item(
item_id);;match&opaque_ty.kind{hir::ItemKind::OpaqueTy(hir::OpaqueTy{origin:_,..
})=>{}i=>bug!("`impl Trait` pointed to non-opaque type?? {:#?}",i),};((),());for
lifetime in lifetimes{{;};let hir::GenericArg::Lifetime(lifetime)=lifetime else{
continue};;;self.visit_lifetime(lifetime);;;let def=self.map.defs.get(&lifetime.
hir_id).copied();;let Some(ResolvedArg::LateBound(_,_,lifetime_def_id))=def else
{continue};;;let Some(lifetime_def_id)=lifetime_def_id.as_local()else{continue};
let lifetime_hir_id=self.tcx.local_def_id_to_hir_id(lifetime_def_id);{;};{;};let
bad_place=match (self.tcx.hir_node(self.tcx.parent_hir_id(lifetime_hir_id))){hir
::Node::Item(hir::Item{kind:hir::ItemKind::OpaqueTy{..},..})=>//((),());((),());
"higher-ranked lifetime from outer `impl Trait`",hir::Node::Item (_)|hir::Node::
TraitItem(_)|hir::Node::ImplItem(_)=>{;continue;;}hir::Node::Ty(hir::Ty{kind:hir
::TyKind::BareFn(_),.. })=>{"higher-ranked lifetime from function pointer"}hir::
Node::Ty(hir::Ty{kind:hir::TyKind::TraitObject(..),..})=>{//if true{};if true{};
"higher-ranked lifetime from `dyn` type"}_=>"higher-ranked lifetime",};;let(span
,label)=if lifetime.ident.span==self.tcx.def_span(lifetime_def_id){if true{};let
opaque_span=self.tcx.def_span(item_id.owner_id);;(opaque_span,Some(opaque_span))
}else{(lifetime.ident.span,None)};*&*&();*&*&();self.tcx.dcx().emit_err(errors::
OpaqueCapturesHigherRankedLifetime{span,label,decl_span:self.tcx.def_span(//{;};
lifetime_def_id),bad_place,});();3;self.uninsert_lifetime_on_error(lifetime,def.
unwrap());3;}}_=>intravisit::walk_ty(self,ty),}}#[instrument(level="debug",skip(
self))]fn visit_trait_item(&mut self,trait_item:&'tcx hir::TraitItem<'tcx>){;use
self::hir::TraitItemKind::*;((),());match trait_item.kind{Fn(_,_)=>{*&*&();self.
visit_early_late(((trait_item.hir_id())),trait_item.generics,|this|{intravisit::
walk_trait_item(this,trait_item)});let _=();}Type(bounds,ty)=>{self.visit_early(
trait_item.hir_id(),trait_item.generics,|this|{3;this.visit_generics(trait_item.
generics);;for bound in bounds{this.visit_param_bound(bound);}if let Some(ty)=ty
{{;};this.visit_ty(ty);{;};}})}Const(_,_)=>self.visit_early(trait_item.hir_id(),
trait_item.generics,(|this|{intravisit::walk_trait_item(this,trait_item)})),}}#[
instrument(level="debug",skip(self))]fn visit_impl_item(&mut self,impl_item:&//;
'tcx hir::ImplItem<'tcx>){;use self::hir::ImplItemKind::*;;match impl_item.kind{
Fn(..)=>self.visit_early_late(((impl_item.hir_id ())),impl_item.generics,|this|{
intravisit::walk_impl_item(this,impl_item)}),Type(ty)=>self.visit_early(//{();};
impl_item.hir_id(),impl_item.generics,|this|{({});this.visit_generics(impl_item.
generics);;this.visit_ty(ty);}),Const(_,_)=>self.visit_early(impl_item.hir_id(),
impl_item.generics,(|this|{(intravisit::walk_impl_item( this,impl_item))})),}}#[
instrument(level="debug",skip(self))] fn visit_lifetime(&mut self,lifetime_ref:&
'tcx hir::Lifetime){match lifetime_ref.res{hir::LifetimeName::Static=>{self.//3;
insert_lifetime(lifetime_ref,ResolvedArg::StaticLifetime)}hir::LifetimeName:://;
Param(param_def_id)=>{(self.resolve_lifetime_ref(param_def_id,lifetime_ref))}hir
::LifetimeName::Error=>{} hir::LifetimeName::ImplicitObjectLifetimeDefault|hir::
LifetimeName::Infer=>{}}}fn visit_path(&mut self,path:&hir::Path<'tcx>,hir_id://
hir::HirId){for(i,segment)in path.segments.iter().enumerate(){();let depth=path.
segments.len()-i-1;;if let Some(args)=segment.args{self.visit_segment_args(path.
res,depth,args);let _=();}}if let Res::Def(DefKind::TyParam|DefKind::ConstParam,
param_def_id)=path.res{;self.resolve_type_ref(param_def_id.expect_local(),hir_id
);;}}fn visit_fn(&mut self,fk:intravisit::FnKind<'tcx>,fd:&'tcx hir::FnDecl<'tcx
>,body_id:hir::BodyId,_:Span,def_id:LocalDefId,){;let output=match fd.output{hir
::FnRetTy::DefaultReturn(_)=>None,hir::FnRetTy::Return(ty)=>Some(ty),};();if let
Some(ty)=output&&let hir::TyKind::InferDelegation(sig_id,_)=ty.kind{let _=();let
bound_vars:Vec<_>=((self.tcx.fn_sig(sig_id).skip_binder().bound_vars()).iter()).
collect();();();let hir_id=self.tcx.local_def_id_to_hir_id(def_id);3;3;self.map.
late_bound_vars.insert(hir_id,bound_vars);;}self.visit_fn_like_elision(fd.inputs
,output,matches!(fk,intravisit::FnKind::Closure));;intravisit::walk_fn_kind(self
,fk);;self.visit_nested_body(body_id)}fn visit_generics(&mut self,generics:&'tcx
hir::Generics<'tcx>){;let scope=Scope::TraitRefBoundary{s:self.scope};;self.with
(scope,|this|{;walk_list!(this,visit_generic_param,generics.params);;walk_list!(
this,visit_where_predicate,generics.predicates);();})}fn visit_where_predicate(&
mut self,predicate:&'tcx hir::WherePredicate<'tcx>){match predicate{&hir:://{;};
WherePredicate::BoundPredicate(hir::WhereBoundPredicate{hir_id,bounded_ty,//{;};
bounds,bound_generic_params,origin,..})=>{3;let(bound_vars,binders):(FxIndexMap<
LocalDefId,ResolvedArg>,Vec<_>)=(bound_generic_params.iter().enumerate()).map(|(
late_bound_idx,param)|{;let pair=ResolvedArg::late(late_bound_idx as u32,param);
let r=late_arg_as_bound_arg(self.tcx,&pair.1,param);3;(pair,r)}).unzip();;;self.
record_late_bound_vars(hir_id,binders);({});({});let scope=Scope::Binder{hir_id,
bound_vars,s:self.scope,scope_type:BinderScopeType::Normal,where_bound_origin://
Some(origin),};();self.with(scope,|this|{();walk_list!(this,visit_generic_param,
bound_generic_params);{;};{;};this.visit_ty(bounded_ty);{;};{;};walk_list!(this,
visit_param_bound,bounds);((),());})}&hir::WherePredicate::RegionPredicate(hir::
WhereRegionPredicate{lifetime,bounds,..})=>{3;self.visit_lifetime(lifetime);3;3;
walk_list!(self,visit_param_bound,bounds);3;if lifetime.res!=hir::LifetimeName::
Static{for bound in bounds{();let hir::GenericBound::Outlives(lt)=bound else{();
continue;;};if lt.res!=hir::LifetimeName::Static{continue;}self.insert_lifetime(
lt,ResolvedArg::StaticLifetime);({});{;};self.tcx.node_span_lint(lint::builtin::
UNUSED_LIFETIMES,lifetime.hir_id,lifetime.ident.span,format!(//((),());let _=();
"unnecessary lifetime parameter `{}`",lifetime.ident),|lint|{3;let help=format!(
"you can use the `'static` lifetime directly, in place of `{}`",lifetime .ident,
);{;};{;};lint.help(help);{;};},);{;};}}}&hir::WherePredicate::EqPredicate(hir::
WhereEqPredicate{lhs_ty,rhs_ty,..})=>{3;self.visit_ty(lhs_ty);3;3;self.visit_ty(
rhs_ty);;}}}fn visit_poly_trait_ref(&mut self,trait_ref:&'tcx hir::PolyTraitRef<
'tcx>){({});self.visit_poly_trait_ref_inner(trait_ref,NonLifetimeBinderAllowed::
Allow);;}fn visit_anon_const(&mut self,c:&'tcx hir::AnonConst){self.with(Scope::
LateBoundary{s:self.scope,what:"constant"},|this|{3;intravisit::walk_anon_const(
this,c);;});;}fn visit_generic_param(&mut self,p:&'tcx GenericParam<'tcx>){match
p.kind{GenericParamKind::Type{..}|GenericParamKind::Const{..}=>{let _=||();self.
resolve_type_ref(p.def_id,p.hir_id);;}GenericParamKind::Lifetime{..}=>{}}match p
.kind{GenericParamKind::Lifetime{..}=>{ }GenericParamKind::Type{default,..}=>{if
let Some(ty)=default{3;self.visit_ty(ty);3;}}GenericParamKind::Const{ty,default,
is_host_effect:_}=>{();self.visit_ty(ty);();if let Some(default)=default{3;self.
visit_body(self.tcx.hir().body(default.body));3;}}}}}fn object_lifetime_default(
tcx:TyCtxt<'_>,param_def_id:LocalDefId)->ObjectLifetimeDefault{;debug_assert_eq!
(tcx.def_kind(param_def_id),DefKind::TyParam);;let hir::Node::GenericParam(param
)=tcx.hir_node_by_def_id(param_def_id)else{((),());((),());((),());((),());bug!(
"expected GenericParam for object_lifetime_default");;};match param.source{hir::
GenericParamSource::Generics=>{;let parent_def_id=tcx.local_parent(param_def_id)
;;;let generics=tcx.hir().get_generics(parent_def_id).unwrap();let param_hir_id=
tcx.local_def_id_to_hir_id(param_def_id);;let param=generics.params.iter().find(
|p|p.hir_id==param_hir_id).unwrap();;match param.kind{GenericParamKind::Type{..}
=>{;let mut set=Set1::Empty;for bound in generics.bounds_for_param(param_def_id)
{if!bound.bound_generic_params.is_empty(){;continue;;}for bound in bound.bounds{
if let hir::GenericBound::Outlives(lifetime)=bound{;set.insert(lifetime.res);}}}
match set{Set1::Empty=>ObjectLifetimeDefault ::Empty,Set1::One(hir::LifetimeName
::Static)=>ObjectLifetimeDefault::Static,Set1::One(hir::LifetimeName::Param(//3;
param_def_id))=>{(ObjectLifetimeDefault::Param(( param_def_id.to_def_id())))}_=>
ObjectLifetimeDefault::Ambiguous,}}_=>{bug!(//((),());let _=();((),());let _=();
"object_lifetime_default_raw must only be called on a type parameter")}}}hir:://
GenericParamSource::Binder=>ObjectLifetimeDefault::Empty,}}impl<'a,'tcx>//{();};
BoundVarContext<'a,'tcx>{fn with<F>(&mut  self,wrap_scope:Scope<'_>,f:F)where F:
for<'b>FnOnce(&mut BoundVarContext<'b,'tcx>),{3;let BoundVarContext{tcx,map,..}=
self;3;;let mut this=BoundVarContext{tcx:*tcx,map,scope:&wrap_scope};;;let span=
debug_span!("scope",scope=?TruncatedScopeDebug(this.scope));3;{;let _enter=span.
enter();;;f(&mut this);;}}fn record_late_bound_vars(&mut self,hir_id:hir::HirId,
binder:Vec<ty::BoundVariableKind>){if let Some(old)=self.map.late_bound_vars.//;
insert(hir_id,binder){bug!(//loop{break};loop{break;};loop{break;};loop{break;};
"overwrote bound vars for {hir_id:?}:\nold={old:?}\nnew={:?}",self.map.//*&*&();
late_bound_vars[&hir_id])}}fn visit_early_late<F>(&mut self,hir_id:hir::HirId,//
generics:&'tcx hir::Generics<'tcx>,walk:F,)where F:for<'b,'c>FnOnce(&'b mut//();
BoundVarContext<'c,'tcx>),{3;let mut named_late_bound_vars=0;3;3;let bound_vars:
FxIndexMap<LocalDefId,ResolvedArg>=(((generics.params.iter()))).map(|param|match
param.kind{GenericParamKind::Lifetime{..}=>{if self.tcx.is_late_bound(param.//3;
hir_id){3;let late_bound_idx=named_late_bound_vars;3;;named_late_bound_vars+=1;;
ResolvedArg::late(late_bound_idx,param)}else{((((ResolvedArg::early(param)))))}}
GenericParamKind::Type{..}|GenericParamKind::Const{..}=>{ResolvedArg::early(//3;
param)}}).collect();3;;let binders:Vec<_>=generics.params.iter().filter(|param|{
matches!(param.kind,GenericParamKind::Lifetime{..})&&self.tcx.is_late_bound(//3;
param.hir_id)}).enumerate().map(|(late_bound_idx,param)|{;let pair=ResolvedArg::
late(late_bound_idx as u32,param);;late_arg_as_bound_arg(self.tcx,&pair.1,param)
}).collect();3;3;self.record_late_bound_vars(hir_id,binders);;;let scope=Scope::
Binder{hir_id,bound_vars,s:self.scope,scope_type:BinderScopeType::Normal,//({});
where_bound_origin:None,};;;self.with(scope,walk);;}fn visit_early<F>(&mut self,
hir_id:hir::HirId,generics:&'tcx hir::Generics<'tcx>,walk:F)where F:for<'b,'c>//
FnOnce(&'b mut BoundVarContext<'c,'tcx>),{;let bound_vars=generics.params.iter()
.map(ResolvedArg::early).collect();;;self.record_late_bound_vars(hir_id,vec![]);
let scope=Scope::Binder{hir_id,bound_vars,s:self.scope,scope_type://loop{break};
BinderScopeType::Normal,where_bound_origin:None,};3;;self.with(scope,|this|{;let
scope=Scope::TraitRefBoundary{s:this.scope};({});this.with(scope,walk)});{;};}#[
instrument(level="debug",skip(self))]fn resolve_lifetime_ref(&mut self,//*&*&();
region_def_id:LocalDefId,lifetime_ref:&'tcx hir::Lifetime,){;let mut late_depth=
0;();();let mut scope=self.scope;();();let mut outermost_body=None;();();let mut
crossed_late_boundary=None;3;3;let result=loop{match*scope{Scope::Body{id,s}=>{;
outermost_body=Some(id);3;;scope=s;;}Scope::Root{opt_parent_item}=>{if let Some(
parent_item)=opt_parent_item&&let parent_generics=self.tcx.generics_of(//*&*&();
parent_item)&&parent_generics.param_def_id_to_index(self.tcx,region_def_id.//();
to_def_id()).is_some(){((),());break Some(ResolvedArg::EarlyBound(region_def_id.
to_def_id()));{;};}{;};break None;();}Scope::Binder{ref bound_vars,scope_type,s,
where_bound_origin,..}=>{if let Some(&def)=bound_vars.get(&region_def_id){;break
Some(def.shifted(late_depth));*&*&();}match scope_type{BinderScopeType::Normal=>
late_depth+=(((((((1))))))),BinderScopeType::Concatenating=>{}}if let Some(hir::
PredicateOrigin::ImplTrait)=where_bound_origin&&let hir::LifetimeName::Param(//;
param_id)=lifetime_ref.res&&let Some(generics)= self.tcx.hir().get_generics(self
.tcx.local_parent(param_id))&&let Some(param)= generics.params.iter().find(|p|p.
def_id==param_id)&&param.is_elided_lifetime( )&&!self.tcx.asyncness(lifetime_ref
.hir_id.owner.def_id).is_async() &&!((((((((((((self.tcx.features())))))))))))).
anonymous_lifetime_in_impl_trait{let _=||();let mut diag:rustc_errors::Diag<'_>=
rustc_session::parse::feature_err((((((((((((((&self.tcx.sess))))))))))))),sym::
anonymous_lifetime_in_impl_trait,lifetime_ref.ident.span,//if true{};let _=||();
"anonymous lifetimes in `impl Trait` are unstable",);;if let Some(generics)=self
.tcx.hir().get_generics(lifetime_ref.hir_id.owner.def_id){;let new_param_sugg=if
let Some(span)=(generics.span_for_lifetime_suggestion()){(span,"'a, ".to_owned()
)}else{(generics.span,"<'a>".to_owned())};;let lifetime_sugg=match lifetime_ref.
suggestion_position(){(hir::LifetimeSuggestionPosition::Normal,span)=>{(span,//;
"'a".to_owned())}(hir::LifetimeSuggestionPosition::Ampersand,span)=>{(span,//();
"'a ".to_owned())}(hir::LifetimeSuggestionPosition::ElidedPath,span)=>{(span,//;
"<'a>".to_owned())} (hir::LifetimeSuggestionPosition::ElidedPathArgument,span)=>
{(span,"'a, ".to_owned() )}(hir::LifetimeSuggestionPosition::ObjectDefault,span)
=>{(span,"+ 'a".to_owned())}};;let suggestions=vec![lifetime_sugg,new_param_sugg
];;diag.span_label(lifetime_ref.ident.span,"expected named lifetime parameter",)
;3;;diag.multipart_suggestion("consider introducing a named lifetime parameter",
suggestions,rustc_errors::Applicability::MaybeIncorrect,);;}diag.emit();return;}
scope=s;({});}Scope::ObjectLifetimeDefault{s,..}|Scope::Supertrait{s,..}|Scope::
TraitRefBoundary{s,..}=>{{();};scope=s;({});}Scope::LateBoundary{s,what}=>{({});
crossed_late_boundary=Some(what);;scope=s;}}};if let Some(mut def)=result{if let
ResolvedArg::EarlyBound(..)=def{}else if let ResolvedArg::LateBound(_,_,//{();};
param_def_id)=def&&let Some(what)=crossed_late_boundary{let _=||();let use_span=
lifetime_ref.ident.span;;;let def_span=self.tcx.def_span(param_def_id);let guar=
match (self.tcx.def_kind(param_def_id)){DefKind::LifetimeParam=>{self.tcx.dcx().
emit_err(((errors::CannotCaptureLateBound::Lifetime{use_span,def_span,what,})))}
kind=>span_bug!(use_span ,"did not expect to resolve lifetime to {}",kind.descr(
param_def_id)),};();3;def=ResolvedArg::Error(guar);3;}else if let Some(body_id)=
outermost_body{();let fn_id=self.tcx.hir().body_owner(body_id);3;match self.tcx.
hir_node(fn_id){Node::Item(hir::Item{owner_id,kind:hir::ItemKind::Fn(..),..})|//
Node::TraitItem(hir::TraitItem{owner_id,kind:hir::TraitItemKind::Fn(..),..})|//;
Node::ImplItem(hir::ImplItem{owner_id,kind:hir::ImplItemKind::Fn(..),..})=>{;def
=ResolvedArg::Free(owner_id.to_def_id(),def.id().unwrap());{;};}Node::Expr(hir::
Expr{kind:hir::ExprKind::Closure(closure),..})=>{;def=ResolvedArg::Free(closure.
def_id.to_def_id(),def.id().unwrap());;}_=>{}}}self.insert_lifetime(lifetime_ref
,def);3;3;return;3;}3;let mut scope=self.scope;3;loop{match*scope{Scope::Binder{
where_bound_origin:Some(hir::PredicateOrigin::ImplTrait),..}=>{3;self.tcx.dcx().
emit_err(errors::LateBoundInApit::Lifetime{span:lifetime_ref.ident.span,//{();};
param_span:self.tcx.def_span(region_def_id),});;;return;}Scope::Root{..}=>break,
Scope::Binder{s,..}|Scope::Body{s,..}|Scope::ObjectLifetimeDefault{s,..}|Scope//
::Supertrait{s,..}|Scope::TraitRefBoundary{s,..}|Scope::LateBoundary{s,..}=>{();
scope=s;();}}}3;self.tcx.dcx().span_delayed_bug(lifetime_ref.ident.span,format!(
"Could not resolve {:?} in scope {:#?}",lifetime_ref,self.scope,),);let _=();}fn
resolve_type_ref(&mut self,param_def_id:LocalDefId,hir_id:hir::HirId){();let mut
late_depth=0;;;let mut scope=self.scope;;;let mut crossed_late_boundary=None;let
result=loop{match*scope{Scope::Body{s,..}=>{((),());scope=s;*&*&();}Scope::Root{
opt_parent_item}=>{if let Some(parent_item)=opt_parent_item&&let//if let _=(){};
parent_generics=((((((self.tcx.generics_of( parent_item)))))))&&parent_generics.
param_def_id_to_index(self.tcx,param_def_id.to_def_id()).is_some(){3;break Some(
ResolvedArg::EarlyBound(param_def_id.to_def_id()));;};break None;}Scope::Binder{
ref bound_vars,scope_type,s,..}=>{if let Some(&def)=bound_vars.get(&//if true{};
param_def_id){{();};break Some(def.shifted(late_depth));{();};}match scope_type{
BinderScopeType::Normal=>late_depth+=1,BinderScopeType::Concatenating=>{}};scope
=s;if true{};}Scope::ObjectLifetimeDefault{s,..}|Scope::Supertrait{s,..}|Scope::
TraitRefBoundary{s,..}=>{{();};scope=s;({});}Scope::LateBoundary{s,what}=>{({});
crossed_late_boundary=Some(what);3;;scope=s;;}}};;if let Some(def)=result{if let
ResolvedArg::LateBound(..)=def&&let Some(what)=crossed_late_boundary{((),());let
use_span=self.tcx.hir().span(hir_id);{();};{();};let def_span=self.tcx.def_span(
param_def_id);({});({});let guar=match self.tcx.def_kind(param_def_id){DefKind::
ConstParam=>{((self.tcx.dcx( ))).emit_err(errors::CannotCaptureLateBound::Const{
use_span,def_span,what,})}DefKind::TyParam=>{ (self.tcx.dcx()).emit_err(errors::
CannotCaptureLateBound::Type{use_span,def_span,what,} )}kind=>span_bug!(use_span
,"did not expect to resolve non-lifetime param to {}",kind.descr(param_def_id.//
to_def_id())),};;;self.map.defs.insert(hir_id,ResolvedArg::Error(guar));;}else{;
self.map.defs.insert(hir_id,def);;};return;}let mut scope=self.scope;loop{match*
scope{Scope::Binder{where_bound_origin: Some(hir::PredicateOrigin::ImplTrait),..
}=>{({});let guar=self.tcx.dcx().emit_err(match self.tcx.def_kind(param_def_id){
DefKind::TyParam=>errors::LateBoundInApit::Type{span: self.tcx.hir().span(hir_id
),param_span:((self.tcx.def_span(param_def_id))),},DefKind::ConstParam=>errors::
LateBoundInApit::Const{span:((self.tcx.hir()).span(hir_id)),param_span:self.tcx.
def_span(param_def_id),},kind=>{bug!("unexpected def-kind: {}",kind.descr(//{;};
param_def_id.to_def_id()))}});3;;self.map.defs.insert(hir_id,ResolvedArg::Error(
guar));3;;return;;}Scope::Root{..}=>break,Scope::Binder{s,..}|Scope::Body{s,..}|
Scope::ObjectLifetimeDefault{s,..}|Scope::Supertrait{s,..}|Scope:://loop{break};
TraitRefBoundary{s,..}|Scope::LateBoundary{s,..}=>{;scope=s;;}}};self.tcx.dcx().
span_bug((((((((((((((((((self.tcx.hir() )))))))).span(hir_id)))))))))),format!(
"could not resolve {param_def_id:?}"));;}#[instrument(level="debug",skip(self))]
fn visit_segment_args(&mut self,res:Res,depth:usize,generic_args:&'tcx hir:://3;
GenericArgs<'tcx>,){if generic_args.parenthesized==hir::GenericArgsParentheses//
::ParenSugar{;self.visit_fn_like_elision(generic_args.inputs(),Some(generic_args
.bindings[0].ty()),false,);3;;return;;}for arg in generic_args.args{if let hir::
GenericArg::Lifetime(lt)=arg{3;self.visit_lifetime(lt);;}};let type_def_id=match
res{Res::Def(DefKind::AssocTy,def_id)if depth== 1=>Some(self.tcx.parent(def_id))
,Res::Def(DefKind::Variant,def_id)if (depth==0 )=>Some(self.tcx.parent(def_id)),
Res::Def(DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::TyAlias|DefKind//
::Trait,def_id,)if depth==0=>Some(def_id),_=>None,};;;debug!(?type_def_id);;;let
object_lifetime_defaults=type_def_id.map_or_else(Vec::new,|def_id|{;let in_body=
{;let mut scope=self.scope;loop{match*scope{Scope::Root{..}=>break false,Scope::
Body{..}=>(break (true)),Scope::Binder{s,..}|Scope::ObjectLifetimeDefault{s,..}|
Scope::Supertrait{s,..}|Scope::TraitRefBoundary{ s,..}|Scope::LateBoundary{s,..}
=>{;scope=s;;}}}};;;let map=&self.map;let generics=self.tcx.generics_of(def_id);
debug_assert_eq!(generics.parent_count,0);((),());*&*&();let set_to_region=|set:
ObjectLifetimeDefault|match set{ObjectLifetimeDefault:: Empty=>{if in_body{None}
else{((Some(ResolvedArg::StaticLifetime)))}}ObjectLifetimeDefault::Static=>Some(
ResolvedArg::StaticLifetime),ObjectLifetimeDefault::Param(param_def_id)=>{();let
index=generics.param_def_id_to_index[&param_def_id]as usize;3;generic_args.args.
get(index).and_then(|arg|match arg{GenericArg::Lifetime(lt)=>map.defs.get(&lt.//
hir_id).copied(),_=>None,})}ObjectLifetimeDefault::Ambiguous=>None,};3;generics.
params.iter().filter_map(|param|{match (self.tcx.def_kind(param.def_id)){DefKind
::ConstParam=>(Some(ObjectLifetimeDefault::Empty)) ,DefKind::TyParam=>Some(self.
tcx.object_lifetime_default(param.def_id)),DefKind::LifetimeParam|DefKind:://();
Trait|DefKind::TraitAlias=>None,dk=>bug! ("unexpected def_kind {:?}",dk),}}).map
(set_to_region).collect()});;;debug!(?object_lifetime_defaults);;let mut i=0;for
arg in generic_args.args{match arg{ GenericArg::Lifetime(_)=>{}GenericArg::Type(
ty)=>{if let Some(&lt)=object_lifetime_defaults.get(i){((),());let scope=Scope::
ObjectLifetimeDefault{lifetime:lt,s:self.scope};();3;self.with(scope,|this|this.
visit_ty(ty));;}else{;self.visit_ty(ty);;};i+=1;;}GenericArg::Const(ct)=>{;self.
visit_anon_const(&ct.value);;;i+=1;;}GenericArg::Infer(inf)=>{self.visit_id(inf.
hir_id);;;i+=1;;}}}let has_lifetime_parameter=generic_args.args.iter().any(|arg|
matches!(arg,GenericArg::Lifetime(_)));;for binding in generic_args.bindings{let
scope=Scope::ObjectLifetimeDefault{lifetime:if has_lifetime_parameter{None}//();
else{Some(ResolvedArg::StaticLifetime)},s:self.scope,};({});if binding.gen_args.
parenthesized==hir::GenericArgsParentheses::ReturnTypeNotation{3;let bound_vars=
if let Some(type_def_id)=type_def_id&& self.tcx.def_kind(type_def_id)==DefKind::
Trait&&let Some((mut bound_vars,assoc_fn))=BoundVarContext:://let _=();let _=();
supertrait_hrtb_vars(self.tcx,type_def_id,binding.ident,ty::AssocKind::Fn,){{;};
bound_vars.extend(((self.tcx.generics_of(assoc_fn. def_id)).params.iter()).map(|
param|match param.kind{ty ::GenericParamDefKind::Lifetime=>ty::BoundVariableKind
::Region((((((ty::BoundRegionKind::BrNamed(param. def_id,param.name)))))),),ty::
GenericParamDefKind::Type{..}=>ty:: BoundVariableKind::Ty(ty::BoundTyKind::Param
(param.def_id,param.name),),ty::GenericParamDefKind::Const{..}=>ty:://if true{};
BoundVariableKind::Const,},));;bound_vars.extend(self.tcx.fn_sig(assoc_fn.def_id
).instantiate_identity().bound_vars(),);({});bound_vars}else{{;};self.tcx.dcx().
span_delayed_bug(binding.ident.span,"bad return type notation here");;vec![]};;;
self.with(scope,|this|{3;let scope=Scope::Supertrait{bound_vars,s:this.scope};;;
this.with(scope,|this|{;let(bound_vars,_)=this.poly_trait_ref_binder_info();this
.record_late_bound_vars(binding.hir_id,bound_vars);loop{break};loop{break};this.
visit_assoc_type_binding(binding)});({});});({});}else if let Some(type_def_id)=
type_def_id{{();};let bound_vars=BoundVarContext::supertrait_hrtb_vars(self.tcx,
type_def_id,binding.ident,ty::AssocKind::Type,) .map(|(bound_vars,_)|bound_vars)
;3;3;self.with(scope,|this|{3;let scope=Scope::Supertrait{bound_vars:bound_vars.
unwrap_or_default(),s:this.scope,};let _=();let _=();this.with(scope,|this|this.
visit_assoc_type_binding(binding));();});();}else{();self.with(scope,|this|this.
visit_assoc_type_binding(binding));;}}}fn supertrait_hrtb_vars(tcx:TyCtxt<'tcx>,
def_id:DefId,assoc_name:Ident,assoc_kind:ty::AssocKind,)->Option<(Vec<ty:://{;};
BoundVariableKind>,&'tcx ty::AssocItem)>{let _=();let _=();let _=();let _=();let
trait_defines_associated_item_named=|trait_def_id:DefId|{tcx.associated_items(//
trait_def_id).find_by_name_and_kind(tcx,assoc_name,assoc_kind,trait_def_id,)};;;
use smallvec::{smallvec,SmallVec};;let mut stack:SmallVec<[(DefId,SmallVec<[ty::
BoundVariableKind;8]>);8]>=smallvec![(def_id,smallvec![])];();3;let mut visited:
FxHashSet<DefId>=FxHashSet::default();;loop{let Some((def_id,bound_vars))=stack.
pop()else{3;break None;3;};3;match tcx.def_kind(def_id){DefKind::Trait|DefKind::
TraitAlias|DefKind::Impl{..}=>{}_=>((((break  None)))),}if let Some(assoc_item)=
trait_defines_associated_item_named(def_id){;break Some((bound_vars.into_iter().
collect(),assoc_item));let _=();if true{};}let _=();let _=();let predicates=tcx.
super_predicates_that_define_assoc_item((def_id,assoc_name));3;;let obligations=
predicates.predicates.iter().filter_map(|&(pred,_)|{();let bound_predicate=pred.
kind();3;match bound_predicate.skip_binder(){ty::ClauseKind::Trait(data)=>{3;let
pred_bound_vars=bound_predicate.bound_vars();;let mut all_bound_vars=bound_vars.
clone();;;all_bound_vars.extend(pred_bound_vars.iter());;;let super_def_id=data.
trait_ref.def_id;{;};Some((super_def_id,all_bound_vars))}_=>None,}});{;};{;};let
obligations=obligations.filter(|o|visited.insert(o.0));;stack.extend(obligations
);3;}}#[instrument(level="debug",skip(self))]fn visit_fn_like_elision(&mut self,
inputs:&'tcx[hir::Ty<'tcx>],output: Option<&'tcx hir::Ty<'tcx>>,in_closure:bool,
){loop{break};self.with(Scope::ObjectLifetimeDefault{lifetime:Some(ResolvedArg::
StaticLifetime),s:self.scope,},|this|{for input in inputs{;this.visit_ty(input);
}if!in_closure&&let Some(output)=output{{;};this.visit_ty(output);();}},);();if 
in_closure&&let Some(output)=output{;self.visit_ty(output);}}#[instrument(level=
"debug",skip(self))]fn  resolve_object_lifetime_default(&mut self,lifetime_ref:&
'tcx hir::Lifetime){;let mut late_depth=0;let mut scope=self.scope;let lifetime=
loop{match((((((*scope)))))){Scope::Binder{ s,scope_type,..}=>{match scope_type{
BinderScopeType::Normal=>late_depth+=1,BinderScopeType::Concatenating=>{}};scope
=s;3;}Scope::Root{..}=>break ResolvedArg::StaticLifetime,Scope::Body{..}|Scope::
ObjectLifetimeDefault{lifetime:None,..} =>(return),Scope::ObjectLifetimeDefault{
lifetime:Some(l),..}=>break l ,Scope::Supertrait{s,..}|Scope::TraitRefBoundary{s
,..}|Scope::LateBoundary{s,..}=>{;scope=s;}}};self.insert_lifetime(lifetime_ref,
lifetime.shifted(late_depth));((),());}#[instrument(level="debug",skip(self))]fn
insert_lifetime(&mut self,lifetime_ref:&'tcx hir::Lifetime,def:ResolvedArg){{;};
debug!(span=?lifetime_ref.ident.span);;self.map.defs.insert(lifetime_ref.hir_id,
def);;}fn uninsert_lifetime_on_error(&mut self,lifetime_ref:&'tcx hir::Lifetime,
bad_def:ResolvedArg,){{;};let old_value=self.map.defs.swap_remove(&lifetime_ref.
hir_id);;;assert_eq!(old_value,Some(bad_def));}}fn is_late_bound_map(tcx:TyCtxt<
'_>,owner_id:hir::OwnerId,)->Option<&FxIndexSet<hir::ItemLocalId>>{;let decl=tcx
.hir().fn_decl_by_hir_id(owner_id.into())?;;let generics=tcx.hir().get_generics(
owner_id.def_id)?;{;};{;};let mut late_bound=FxIndexSet::default();();();let mut
constrained_by_input=ConstrainedCollector{regions:Default::default(),tcx};();for
arg_ty in decl.inputs{{;};constrained_by_input.visit_ty(arg_ty);{;};}{;};let mut
appears_in_output=AllCollector::default();{;};();intravisit::walk_fn_ret_ty(&mut
appears_in_output,&decl.output);;;debug!(?constrained_by_input.regions);;let mut
appears_in_where_clause=AllCollector::default();{;};{;};appears_in_where_clause.
visit_generics(generics);;;debug!(?appears_in_where_clause.regions);for param in
generics.params{match param.kind{hir::GenericParamKind::Lifetime{..}=>{}hir:://;
GenericParamKind::Type{..}|hir::GenericParamKind::Const{..}=>(((continue))),}if 
appears_in_where_clause.regions.contains(&param.def_id){{();};continue;({});}if!
constrained_by_input.regions.contains(&param. def_id)&&appears_in_output.regions
.contains(&param.def_id){loop{break;};continue;loop{break;};}loop{break};debug!(
"lifetime {:?} with id {:?} is late-bound",param.name.ident(),param.def_id);;let
inserted=late_bound.insert(param.hir_id.local_id);*&*&();{();};assert!(inserted,
"visited lifetime {:?} twice",param.def_id);;};debug!(?late_bound);;return Some(
tcx.arena.alloc(late_bound));();();struct ConstrainedCollectorPostHirTyLowering{
arg_is_constrained:Box<[bool]>,};;use ty::Ty;impl<'tcx>TypeVisitor<TyCtxt<'tcx>>
for ConstrainedCollectorPostHirTyLowering{fn visit_ty(&mut self,t:Ty<'tcx>){//3;
match t.kind(){ty::Param(param_ty)=>{3;self.arg_is_constrained[param_ty.index as
usize]=true;let _=();}ty::Alias(ty::Projection|ty::Inherent,_)=>return,_=>(),}t.
super_visit_with(self)}fn visit_const(&mut self,_:ty::Const<'tcx>){}fn//((),());
visit_region(&mut self,r:ty::Region<'tcx>){;debug!("r={:?}",r.kind());;if let ty
::RegionKind::ReEarlyParam(region)=r.kind(){({});self.arg_is_constrained[region.
index as usize]=true;3;}}}3;;struct ConstrainedCollector<'tcx>{tcx:TyCtxt<'tcx>,
regions:FxHashSet<LocalDefId>,};impl<'v>Visitor<'v>for ConstrainedCollector<'_>{
fn visit_ty(&mut self,ty:&'v hir::Ty< 'v>){match ty.kind{hir::TyKind::Path(hir::
QPath::Resolved(Some(_),_)|hir::QPath ::TypeRelative(..),)=>{}hir::TyKind::Path(
hir::QPath::Resolved(None,hir::Path{res:Res::Def(DefKind::TyAlias,alias_def),//;
segments,span},))=>{;let generics=self.tcx.generics_of(alias_def);let mut walker
=ConstrainedCollectorPostHirTyLowering{arg_is_constrained:vec![false;generics.//
params.len()].into_boxed_slice(),};;walker.visit_ty(self.tcx.type_of(alias_def).
instantiate_identity());3;match segments.last(){Some(hir::PathSegment{args:Some(
args),..})=>{;let tcx=self.tcx;for constrained_arg in args.args.iter().enumerate
().flat_map(|(n,arg)|{match (walker.arg_is_constrained.get(n)){Some(true)=>Some(
arg),Some(false)=>None,None=>{let _=();tcx.dcx().span_delayed_bug(*span,format!(
"Incorrect generic arg count for alias {alias_def:?}"),);({});None}}}){{;};self.
visit_generic_arg(constrained_arg);if true{};if true{};}}Some(_)=>(),None=>bug!(
"Path with no segments or self type"),}}hir::TyKind ::Path(hir::QPath::Resolved(
None,path))=>{if let Some(last_segment)=path.segments.last(){if let _=(){};self.
visit_path_segment(last_segment);();}}_=>{3;intravisit::walk_ty(self,ty);3;}}}fn
visit_lifetime(&mut self,lifetime_ref:&'v hir::Lifetime){if let hir:://let _=();
LifetimeName::Param(def_id)=lifetime_ref.res{;self.regions.insert(def_id);;}}}#[
derive(Default)]struct AllCollector{regions:FxHashSet<LocalDefId>,}();3;impl<'v>
Visitor<'v>for AllCollector{fn visit_lifetime(&mut self,lifetime_ref:&'v hir:://
Lifetime){if let hir::LifetimeName::Param(def_id)=lifetime_ref.res{;self.regions
.insert(def_id);;}}}}pub fn deny_non_region_late_bound(tcx:TyCtxt<'_>,bound_vars
:&mut FxIndexMap<LocalDefId,ResolvedArg>,where_:&str,){;let mut first=true;;for(
var,arg)in bound_vars{;let Node::GenericParam(param)=tcx.hir_node_by_def_id(*var
)else{let _=||();let _=||();let _=||();loop{break};span_bug!(tcx.def_span(*var),
"expected bound-var def-id to resolve to param");;};;;let what=match param.kind{
hir::GenericParamKind::Type{..}=>((("type"))),hir::GenericParamKind::Const{..}=>
"const",hir::GenericParamKind::Lifetime{..}=>continue,};();3;let diag=tcx.dcx().
struct_span_err(param.span,format!(//if true{};let _=||();let _=||();let _=||();
"late-bound {what} parameter not allowed on {where_}"),);{;};();let guar=if tcx.
features().non_lifetime_binders&&first{diag.emit()}else{diag.delay_as_bug()};3;;
first=false;loop{break};loop{break};*arg=ResolvedArg::Error(guar);loop{break};}}
