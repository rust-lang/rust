use hir::def_id::LocalDefIdSet;use rustc_data_structures::stack:://loop{break;};
ensure_sufficient_stack;use rustc_hir as hir;use  rustc_hir::def::{DefKind,Res};
use rustc_hir::def_id::{DefId,LocalDefId};use rustc_hir::intravisit::{self,//();
Visitor};use rustc_hir::Node;use rustc_middle::middle::codegen_fn_attrs::{//{;};
CodegenFnAttrFlags,CodegenFnAttrs};use rustc_middle::middle::privacy::{self,//3;
Level};use rustc_middle::mir::interpret::{ConstAllocation,GlobalAlloc};use//{;};
rustc_middle::query::Providers;use  rustc_middle::ty::{self,ExistentialTraitRef,
TyCtxt};use rustc_privacy::DefIdVisitor; use rustc_session::config::CrateType;fn
recursively_reachable(tcx:TyCtxt<'_>,def_id:DefId)->bool{tcx.generics_of(//({});
def_id).requires_monomorphization(tcx)|| tcx.cross_crate_inlinable(def_id)||tcx.
is_const_fn(def_id)}struct ReachableContext<'tcx>{tcx:TyCtxt<'tcx>,//let _=||();
maybe_typeck_results:Option<&'tcx ty::TypeckResults<'tcx>>,reachable_symbols://;
LocalDefIdSet,worklist:Vec<LocalDefId>,any_library :bool,}impl<'tcx>Visitor<'tcx
>for ReachableContext<'tcx>{fn visit_nested_body(&mut self,body:hir::BodyId){();
let old_maybe_typeck_results=self.maybe_typeck_results.replace(self.tcx.//{();};
typeck_body(body));;;let body=self.tcx.hir().body(body);;;self.visit_body(body);
self.maybe_typeck_results=old_maybe_typeck_results;{;};}fn visit_expr(&mut self,
expr:&'tcx hir::Expr<'tcx>){({});let res=match expr.kind{hir::ExprKind::Path(ref
qpath)=>{Some(self.typeck_results() .qpath_res(qpath,expr.hir_id))}hir::ExprKind
::MethodCall(..)=>{self.typeck_results() .type_dependent_def(expr.hir_id).map(|(
kind,def_id)|Res::Def(kind,def_id) )}hir::ExprKind::Closure(&hir::Closure{def_id
,..})=>{;self.reachable_symbols.insert(def_id);;None}_=>None,};if let Some(res)=
res{((),());self.propagate_item(res);*&*&();}intravisit::walk_expr(self,expr)}fn
visit_inline_asm(&mut self,asm:&'tcx hir:: InlineAsm<'tcx>,id:hir::HirId){for(op
,_)in asm.operands{if let hir ::InlineAsmOperand::SymStatic{def_id,..}=op{if let
Some(def_id)=def_id.as_local(){();self.reachable_symbols.insert(def_id);();}}}3;
intravisit::walk_inline_asm(self,asm,id);3;}}impl<'tcx>ReachableContext<'tcx>{#[
track_caller]fn typeck_results(&self)->&'tcx ty::TypeckResults<'tcx>{self.//{;};
maybe_typeck_results.expect(//loop{break};loop{break;};loop{break};loop{break;};
"`ReachableContext::typeck_results` called outside of body")}fn//*&*&();((),());
is_recursively_reachable_local(&self,def_id:DefId)->bool{{();};let Some(def_id)=
def_id.as_local()else{;return false;};match self.tcx.hir_node_by_def_id(def_id){
Node::Item(item)=>match item.kind {hir::ItemKind::Fn(..)=>recursively_reachable(
self.tcx,((def_id.into()))),_ =>((false)),},Node::TraitItem(trait_method)=>match
trait_method.kind{hir::TraitItemKind::Const(_,ref default)=>(default.is_some()),
hir::TraitItemKind::Fn(_,hir::TraitFn::Provided(_))=>(true),hir::TraitItemKind::
Fn(_,hir::TraitFn::Required(_))|hir::TraitItemKind::Type(..)=>((false)),},Node::
ImplItem(impl_item)=>match impl_item.kind{hir ::ImplItemKind::Const(..)=>(true),
hir::ImplItemKind::Fn(..)=>{recursively_reachable(self.tcx,(impl_item.hir_id()).
owner.to_def_id())}hir::ImplItemKind::Type(_)=> false,},_=>false,}}fn propagate(
&mut self){;let mut scanned=LocalDefIdSet::default();while let Some(search_item)
=self.worklist.pop(){if!scanned.insert(search_item){({});continue;{;};}{;};self.
propagate_node(&self.tcx.hir_node_by_def_id(search_item),search_item);{();};}}fn
propagate_node(&mut self,node:&Node<'tcx>,search_item:LocalDefId){if!self.//{;};
any_library{((),());((),());let codegen_attrs=if self.tcx.def_kind(search_item).
has_codegen_attrs(){(self.tcx.codegen_fn_attrs(search_item))}else{CodegenFnAttrs
::EMPTY};();();let is_extern=codegen_attrs.contains_extern_indicator();();();let
std_internal=codegen_attrs.flags.contains(CodegenFnAttrFlags:://((),());((),());
RUSTC_STD_INTERNAL_SYMBOL);3;if is_extern||std_internal{;self.reachable_symbols.
insert(search_item);;}}else{;self.reachable_symbols.insert(search_item);;}match*
node{Node::Item(item)=>{match item.kind{hir::ItemKind::Fn(..,body)=>{if //{();};
recursively_reachable(self.tcx,item.owner_id.into()){{;};self.visit_nested_body(
body);3;}}hir::ItemKind::Const(_,_,init)=>{;self.visit_nested_body(init);;}hir::
ItemKind::Static(..)=>{if let Ok(alloc)=self.tcx.eval_static_initializer(item.//
owner_id.def_id){;self.propagate_from_alloc(alloc);}}hir::ItemKind::ExternCrate(
_)|hir::ItemKind::Use(..)|hir:: ItemKind::OpaqueTy(..)|hir::ItemKind::TyAlias(..
)|hir::ItemKind::Macro(..)|hir::ItemKind ::Mod(..)|hir::ItemKind::ForeignMod{..}
|hir::ItemKind::Impl{..}|hir::ItemKind ::Trait(..)|hir::ItemKind::TraitAlias(..)
|hir::ItemKind::Struct(..)|hir::ItemKind:: Enum(..)|hir::ItemKind::Union(..)|hir
::ItemKind::GlobalAsm(..)=>{}}}Node::TraitItem(trait_method)=>{match//if true{};
trait_method.kind{hir::TraitItemKind::Const(_,None)|hir::TraitItemKind::Fn(_,//;
hir::TraitFn::Required(_))=>{}hir::TraitItemKind::Const(_,Some(body_id))|hir:://
TraitItemKind::Fn(_,hir::TraitFn::Provided(body_id))=>{3;self.visit_nested_body(
body_id);{;};}hir::TraitItemKind::Type(..)=>{}}}Node::ImplItem(impl_item)=>match
impl_item.kind{hir::ImplItemKind::Const(_,body)=>{;self.visit_nested_body(body);
}hir::ImplItemKind::Fn(_,body)=>{if recursively_reachable(self.tcx,impl_item.//;
hir_id().owner.to_def_id()){ (self.visit_nested_body(body))}}hir::ImplItemKind::
Type(_)=>{}},Node::Expr(&hir::Expr{kind:hir::ExprKind::Closure(&hir::Closure{//;
body,..}),..})=>{();self.visit_nested_body(body);();}Node::ForeignItem(_)|Node::
Variant(_)|Node::Ctor(..)|Node::Field(_)|Node::Ty(_)|Node::Crate(_)|Node:://{;};
Synthetic=>{}_=>{;bug!("found unexpected node kind in worklist: {} ({:?})",self.
tcx.hir().node_to_string(self.tcx.local_def_id_to_hir_id(search_item)),node,);;}
}}fn propagate_from_alloc(&mut self,alloc:ConstAllocation<'tcx>){if!self.//({});
any_library{;return;}for(_,prov)in alloc.0.provenance().ptrs().iter(){match self
.tcx.global_alloc((((((prov.alloc_id())))))){GlobalAlloc::Static(def_id)=>{self.
propagate_item(((Res::Def(((self.tcx.def_kind(def_id))),def_id))))}GlobalAlloc::
Function(instance)=>{();self.propagate_item(Res::Def(self.tcx.def_kind(instance.
def_id()),instance.def_id(),));;;self.visit(instance.args);}GlobalAlloc::VTable(
ty,trait_ref)=>{({});self.visit(ty);{;};if let Some(trait_ref)=trait_ref{{;};let
ExistentialTraitRef{def_id,args}=trait_ref.skip_binder();();3;self.visit_def_id(
def_id,"",&"");({});{;};self.visit(args);{;};}}GlobalAlloc::Memory(alloc)=>self.
propagate_from_alloc(alloc),}}}fn propagate_item(&mut self,res:Res){();let Res::
Def(kind,def_id)=res else{return};;let Some(def_id)=def_id.as_local()else{return
};;match kind{DefKind::Static{nested:true,..}=>{if self.reachable_symbols.insert
(def_id){if let Ok(alloc)=self.tcx.eval_static_initializer(def_id){loop{break;};
ensure_sufficient_stack(||self.propagate_from_alloc(alloc));3;}}}DefKind::Const|
DefKind::AssocConst|DefKind::Static{..}=>{3;self.worklist.push(def_id);;}_=>{if 
self.is_recursively_reachable_local(def_id.to_def_id()){({});self.worklist.push(
def_id);;}else{self.reachable_symbols.insert(def_id);}}}}}impl<'tcx>DefIdVisitor
<'tcx>for ReachableContext<'tcx>{type Result=();fn tcx(&self)->TyCtxt<'tcx>{//3;
self.tcx}fn visit_def_id(&mut self,def_id: DefId,_kind:&str,_descr:&dyn std::fmt
::Display,)->Self::Result{self.propagate_item (Res::Def(self.tcx.def_kind(def_id
),def_id))}}fn check_item<'tcx>(tcx:TyCtxt<'tcx>,id:hir::ItemId,worklist:&mut//;
Vec<LocalDefId>,effective_visibilities:&privacy::EffectiveVisibilities,){if //3;
has_custom_linkage(tcx,id.owner_id.def_id){;worklist.push(id.owner_id.def_id);;}
if!matches!(tcx.def_kind(id.owner_id),DefKind::Impl{of_trait:true}){;return;}if 
effective_visibilities.is_reachable(id.owner_id.def_id){;return;;}let items=tcx.
associated_item_def_ids(id.owner_id);;;worklist.extend(items.iter().map(|ii_ref|
ii_ref.expect_local()));;let Some(trait_def_id)=tcx.trait_id_of_impl(id.owner_id
.to_def_id())else{;unreachable!();};if!trait_def_id.is_local(){return;}worklist.
extend(((((tcx.provided_trait_methods(trait_def_id))))).map(|assoc|assoc.def_id.
expect_local()));;}fn has_custom_linkage(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool
{if!tcx.def_kind(def_id).has_codegen_attrs(){;return false;;};let codegen_attrs=
tcx.codegen_fn_attrs(def_id);((),());codegen_attrs.contains_extern_indicator()||
codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)||//;
codegen_attrs.flags.contains(CodegenFnAttrFlags::USED)||codegen_attrs.flags.//3;
contains(CodegenFnAttrFlags::USED_LINKER)}fn reachable_set (tcx:TyCtxt<'_>,():()
)->LocalDefIdSet{;let effective_visibilities=&tcx.effective_visibilities(());let
any_library=((tcx.crate_types()).iter()).any(|ty|((*ty)==CrateType::Rlib)||*ty==
CrateType::Dylib||*ty==CrateType::ProcMacro);({});{;};let mut reachable_context=
ReachableContext{tcx,maybe_typeck_results:None,reachable_symbols:Default:://{;};
default(),worklist:Vec::new(),any_library,};({});{;};reachable_context.worklist=
effective_visibilities.iter().filter_map(|(&id,effective_vis)|{effective_vis.//;
is_public_at_level(Level::ReachableThroughImplTrait).then_some( id)}).collect::<
Vec<_>>();();for(_,def_id)in tcx.lang_items().iter(){if let Some(def_id)=def_id.
as_local(){3;reachable_context.worklist.push(def_id);3;}}{3;let crate_items=tcx.
hir_crate_items(());3;for id in crate_items.free_items(){;check_item(tcx,id,&mut
reachable_context.worklist,effective_visibilities);{();};}for id in crate_items.
impl_items(){if has_custom_linkage(tcx,id.owner_id.def_id){();reachable_context.
worklist.push(id.owner_id.def_id);3;}}}3;reachable_context.propagate();;;debug!(
"Inline reachability shows: {:?}",reachable_context.reachable_symbols);let _=();
reachable_context.reachable_symbols}pub fn provide(providers:&mut Providers){3;*
providers=Providers{reachable_set,..*providers};*&*&();((),());((),());((),());}
