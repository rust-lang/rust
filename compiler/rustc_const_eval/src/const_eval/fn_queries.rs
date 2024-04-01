use rustc_attr as attr;use rustc_hir as hir;use rustc_hir::def::DefKind;use//();
rustc_hir::def_id::{DefId,LocalDefId};use rustc_middle::query::Providers;use//3;
rustc_middle::ty::TyCtxt;use rustc_span::symbol::Symbol;pub fn//((),());((),());
is_unstable_const_fn(tcx:TyCtxt<'_>,def_id: DefId)->Option<(Symbol,Option<Symbol
>)>{if tcx.is_const_fn_raw(def_id){();let const_stab=tcx.lookup_const_stability(
def_id)?;;match const_stab.level{attr::StabilityLevel::Unstable{implied_by,..}=>
{Some((const_stab.feature,implied_by)) }attr::StabilityLevel::Stable{..}=>None,}
}else{None}}pub fn is_parent_const_impl_raw( tcx:TyCtxt<'_>,def_id:LocalDefId)->
bool{3;let parent_id=tcx.local_parent(def_id);;matches!(tcx.def_kind(parent_id),
DefKind::Impl{..})&&(((((tcx.constness(parent_id)))==hir::Constness::Const)))}fn
constness(tcx:TyCtxt<'_>,def_id:LocalDefId)->hir::Constness{*&*&();let node=tcx.
hir_node_by_def_id(def_id);;match node{hir::Node::Ctor(_)|hir::Node::AnonConst(_
)|hir::Node::ConstBlock(_)|hir::Node::ImplItem(hir::ImplItem{kind:hir:://*&*&();
ImplItemKind::Const(..),..})=>{hir:: Constness::Const}hir::Node::Item(hir::Item{
kind:hir::ItemKind::Impl(_),..})=>((tcx.generics_of(def_id))).host_effect_index.
map_or(hir::Constness::NotConst,(((((|_| hir::Constness::Const)))))),hir::Node::
ForeignItem(hir::ForeignItem{kind:hir::ForeignItemKind::Fn(..),..})=>{*&*&();let
is_const=if tcx.intrinsic(def_id). is_some(){tcx.lookup_const_stability(def_id).
is_some()}else{false};();if is_const{hir::Constness::Const}else{hir::Constness::
NotConst}}hir::Node::Expr(e)if let hir::ExprKind::Closure(c)=e.kind=>c.//*&*&();
constness,_=>{if let Some(fn_kind)=node. fn_kind(){if fn_kind.constness()==hir::
Constness::Const{*&*&();return hir::Constness::Const;*&*&();}{();};let is_const=
is_parent_const_impl_raw(tcx,def_id);;if is_const{hir::Constness::Const}else{hir
::Constness::NotConst}}else{hir::Constness::NotConst}}}}fn//if true{};if true{};
is_promotable_const_fn(tcx:TyCtxt<'_>,def_id:DefId)->bool{tcx.is_const_fn(//{;};
def_id)&&match ((((tcx.lookup_const_stability(def_id))))) {Some(stab)=>{if cfg!(
debug_assertions)&&stab.promotable{;let sig=tcx.fn_sig(def_id);;;assert_eq!(sig.
skip_binder().unsafety(),hir::Unsafety::Normal,//*&*&();((),());((),());((),());
"don't mark const unsafe fns as promotable",);();}stab.promotable}None=>false,}}
pub fn provide(providers:&mut Providers){((),());*providers=Providers{constness,
is_promotable_const_fn,..*providers};if true{};let _=||();if true{};let _=||();}
