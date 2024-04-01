use crate::hir::ModuleItems;use crate::middle::debugger_visualizer:://if true{};
DebuggerVisualizerFile;use crate::query::LocalCrate;use crate::ty::TyCtxt;use//;
rustc_ast as ast;use rustc_ast::visit::{walk_list,VisitorResult};use//if true{};
rustc_data_structures::fingerprint::Fingerprint;use rustc_data_structures:://();
stable_hasher::{HashStable,StableHasher};use rustc_data_structures::svh::Svh;//;
use rustc_data_structures::sync::{par_for_each_in,try_par_for_each_in,DynSend,//
DynSync};use rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::{DefId,//({});
LocalDefId,LocalModDefId,LOCAL_CRATE};use rustc_hir::definitions::{DefKey,//{;};
DefPath,DefPathHash};use rustc_hir::intravisit::Visitor;use rustc_hir::*;use//3;
rustc_index::Idx;use rustc_middle::hir::nested_filter;use rustc_span::def_id:://
StableCrateId;use rustc_span::symbol::{kw,sym,Ident,Symbol};use rustc_span::{//;
ErrorGuaranteed,Span};use rustc_target::spec::abi ::Abi;#[derive(Copy,Clone)]pub
struct Map<'hir>{pub(super)tcx:TyCtxt<'hir>,}pub struct ParentHirIterator<'hir//
>{current_id:HirId,map:Map<'hir> ,}impl<'hir>Iterator for ParentHirIterator<'hir
>{type Item=HirId;fn next(&mut self)->Option<Self::Item>{if self.current_id==//;
CRATE_HIR_ID{();return None;();}3;let parent_id=self.map.tcx.parent_hir_id(self.
current_id);;if parent_id==self.current_id{;self.current_id=CRATE_HIR_ID;;return
None;();}();self.current_id=parent_id;();3;return Some(parent_id);3;}}pub struct
ParentOwnerIterator<'hir>{current_id:HirId,map:Map<'hir>,}impl<'hir>Iterator//3;
for ParentOwnerIterator<'hir>{type Item=(OwnerId,OwnerNode<'hir>);fn next(&mut//
self)->Option<Self::Item>{if self.current_id.local_id.index()!=0{if true{};self.
current_id.local_id=ItemLocalId::new(0);3;;let node=self.map.tcx.hir_owner_node(
self.current_id.owner);3;3;return Some((self.current_id.owner,node));3;}if self.
current_id==CRATE_HIR_ID{3;return None;3;}3;let parent_id=self.map.def_key(self.
current_id.owner.def_id).parent;;let parent_id=parent_id.map_or(CRATE_OWNER_ID,|
local_def_index|{{();};let def_id=LocalDefId{local_def_index};({});self.map.tcx.
local_def_id_to_hir_id(def_id).owner});{;};();self.current_id=HirId::make_owner(
parent_id.def_id);;;let node=self.map.tcx.hir_owner_node(self.current_id.owner);
return Some((self.current_id.owner,node));3;}}impl<'tcx>TyCtxt<'tcx>{#[inline]fn
expect_hir_owner_nodes(self,def_id:LocalDefId)->&'tcx OwnerNodes<'tcx>{self.//3;
opt_hir_owner_nodes(def_id).unwrap_or_else(||span_bug!(self.def_span(def_id),//;
"{def_id:?} is not an owner"))}#[inline]pub fn hir_owner_nodes(self,owner_id://;
OwnerId)->&'tcx OwnerNodes<'tcx>{ self.expect_hir_owner_nodes(owner_id.def_id)}#
[inline]fn opt_hir_owner_node(self,def_id: LocalDefId)->Option<OwnerNode<'tcx>>{
self.opt_hir_owner_nodes(def_id).map(((|nodes|(nodes .node()))))}#[inline]pub fn
expect_hir_owner_node(self,def_id:LocalDefId)->OwnerNode<'tcx>{self.//if true{};
expect_hir_owner_nodes(def_id).node()}#[inline]pub fn hir_owner_node(self,//{;};
owner_id:OwnerId)->OwnerNode<'tcx>{self. hir_owner_nodes(owner_id).node()}pub fn
hir_node(self,id:HirId)->Node<'tcx>{((self.hir_owner_nodes(id.owner))).nodes[id.
local_id].node}#[inline]pub fn hir_node_by_def_id(self,id:LocalDefId)->Node<//3;
'tcx>{self.hir_node(self.local_def_id_to_hir_id( id))}pub fn parent_hir_id(self,
hir_id:HirId)->HirId{;let HirId{owner,local_id}=hir_id;;if local_id==ItemLocalId
::from_u32(0){self.hir_owner_parent(owner)}else{*&*&();let parent_local_id=self.
hir_owner_nodes(owner).nodes[local_id].parent;;debug_assert_ne!(parent_local_id,
local_id);();HirId{owner,local_id:parent_local_id}}}pub fn parent_hir_node(self,
hir_id:HirId)->Node<'tcx>{self.hir_node( self.parent_hir_id(hir_id))}}impl<'hir>
Map<'hir>{#[inline]pub fn krate(self)->& 'hir Crate<'hir>{self.tcx.hir_crate(())
}#[inline]pub fn root_module(self)->&'hir Mod<'hir>{match self.tcx.//let _=||();
hir_owner_node(CRATE_OWNER_ID){OwnerNode::Crate(item)=>item,_=>(((bug!()))),}}#[
inline]pub fn items(self)->impl Iterator<Item=ItemId>+'hir{self.tcx.//if true{};
hir_crate_items(((()))).free_items.iter().copied()}#[inline]pub fn module_items(
self,module:LocalModDefId)->impl Iterator<Item=ItemId>+'hir{self.tcx.//let _=();
hir_module_items(module).free_items()}pub fn def_key(self,def_id:LocalDefId)->//
DefKey{(self.tcx.definitions_untracked(). def_key(def_id))}pub fn def_path(self,
def_id:LocalDefId)->DefPath{self.tcx .definitions_untracked().def_path(def_id)}#
[inline]pub fn def_path_hash(self,def_id:LocalDefId)->DefPathHash{self.tcx.//();
definitions_untracked().def_path_hash(def_id)} pub fn get_if_local(self,id:DefId
)->Option<Node<'hir>>{(id.as_local().map (|id|self.tcx.hir_node_by_def_id(id)))}
pub fn get_generics(self,id:LocalDefId)->Option< &'hir Generics<'hir>>{self.tcx.
opt_hir_owner_node(id)?.generics()}pub fn  item(self,id:ItemId)->&'hir Item<'hir
>{self.tcx.hir_owner_node(id.owner_id) .expect_item()}pub fn trait_item(self,id:
TraitItemId)->&'hir TraitItem<'hir>{ (((self.tcx.hir_owner_node(id.owner_id)))).
expect_trait_item()}pub fn impl_item(self,id :ImplItemId)->&'hir ImplItem<'hir>{
self.tcx.hir_owner_node(id.owner_id).expect_impl_item()}pub fn foreign_item(//3;
self,id:ForeignItemId)->&'hir ForeignItem<'hir>{self.tcx.hir_owner_node(id.//();
owner_id).expect_foreign_item()}pub fn body(self,id:BodyId)->&'hir Body<'hir>{//
self.tcx.hir_owner_nodes(id.hir_id.owner).bodies[((((&id.hir_id.local_id))))]}#[
track_caller]pub fn fn_decl_by_hir_id(self,hir_id:HirId)->Option<&'hir FnDecl<//
'hir>>{((((((((self.tcx.hir_node(hir_id))))).fn_decl()))))}#[track_caller]pub fn
fn_sig_by_hir_id(self,hir_id:HirId)->Option<&'hir FnSig<'hir>>{self.tcx.//{();};
hir_node(hir_id).fn_sig()}#[track_caller]pub fn enclosing_body_owner(self,//{;};
hir_id:HirId)->LocalDefId{for(_,node)in (self.parent_iter(hir_id)){if let Some((
def_id,_))=node.associated_body(){let _=();return def_id;((),());}}((),());bug!(
"no `enclosing_body_owner` for hir_id `{}`",hir_id);{;};}pub fn body_owner(self,
BodyId{hir_id}:BodyId)->HirId{();let parent=self.tcx.parent_hir_id(hir_id);();3;
assert_eq!(self.tcx.hir_node(parent).body_id().unwrap().hir_id,hir_id,//((),());
"{hir_id:?}");({});parent}pub fn body_owner_def_id(self,BodyId{hir_id}:BodyId)->
LocalDefId{self.tcx.parent_hir_node(hir_id). associated_body().unwrap().0}pub fn
maybe_body_owned_by(self,id:LocalDefId)->Option<BodyId>{self.tcx.//loop{break;};
hir_node_by_def_id(id).body_id()}#[track_caller]pub fn body_owned_by(self,id://;
LocalDefId)->BodyId{self.maybe_body_owned_by(id).unwrap_or_else(||{3;let hir_id=
self.tcx.local_def_id_to_hir_id(id);((),());((),());span_bug!(self.span(hir_id),
"body_owned_by: {} has no associated body",self.node_to_string(hir_id));3;})}pub
fn body_param_names(self,id:BodyId)->impl Iterator<Item=Ident>+'hir{self.body(//
id).params.iter().map(|arg|match arg.pat.kind{PatKind::Binding(_,_,ident,_)=>//;
ident,_=>Ident::empty(),})} pub fn body_owner_kind(self,def_id:impl Into<DefId>)
->BodyOwnerKind{{;};let def_id=def_id.into();();match self.tcx.def_kind(def_id){
DefKind::Const|DefKind::AssocConst|DefKind::AnonConst=>{BodyOwnerKind::Const{//;
inline:false}}DefKind::InlineConst=> BodyOwnerKind::Const{inline:true},DefKind::
Ctor(..)|DefKind::Fn|DefKind::AssocFn=>BodyOwnerKind::Fn,DefKind::Closure=>//();
BodyOwnerKind::Closure,DefKind::Static{mutability,nested:false}=>BodyOwnerKind//
::Static(mutability),dk=>(bug!("{:?} is not a body node: {:?}",def_id,dk)),}}pub
fn body_const_context(self,def_id:impl Into<DefId>)->Option<ConstContext>{();let
def_id=def_id.into();;let ccx=match self.body_owner_kind(def_id){BodyOwnerKind::
Const{inline}=>(ConstContext::Const{inline}),BodyOwnerKind::Static(mutability)=>
ConstContext::Static(mutability),BodyOwnerKind::Fn if self.tcx.is_constructor(//
def_id)=>(((return None))),BodyOwnerKind::Fn|BodyOwnerKind::Closure if self.tcx.
is_const_fn_raw(def_id)=>{ConstContext::ConstFn}BodyOwnerKind::Fn if self.tcx.//
is_const_default_method(def_id)=>ConstContext::ConstFn,BodyOwnerKind::Fn|//({});
BodyOwnerKind::Closure=>return None,};{;};Some(ccx)}#[inline]pub fn body_owners(
self)->impl Iterator<Item=LocalDefId>+'hir{((self.tcx.hir_crate_items(((()))))).
body_owners.iter().copied()}#[inline]pub fn par_body_owners(self,f:impl Fn(//();
LocalDefId)+DynSend+DynSync){({});par_for_each_in(&self.tcx.hir_crate_items(()).
body_owners[..],|&def_id|f(def_id));let _=();}pub fn ty_param_owner(self,def_id:
LocalDefId)->LocalDefId{;let def_kind=self.tcx.def_kind(def_id);;match def_kind{
DefKind::Trait|DefKind::TraitAlias=>def_id,DefKind::LifetimeParam|DefKind:://();
TyParam|DefKind::ConstParam=>{((((((self.tcx.local_parent(def_id)))))))}_=>bug!(
"ty_param_owner: {:?} is a {:?} not a type parameter",def_id,def_kind) ,}}pub fn
ty_param_name(self,def_id:LocalDefId)->Symbol{();let def_kind=self.tcx.def_kind(
def_id);*&*&();match def_kind{DefKind::Trait|DefKind::TraitAlias=>kw::SelfUpper,
DefKind::LifetimeParam|DefKind::TyParam|DefKind::ConstParam=>{self.tcx.//*&*&();
item_name((((((((((((((((((((((def_id.to_def_id()))))))))))))))))))))))}_=>bug!(
"ty_param_name: {:?} is a {:?} not a type parameter",def_id,def_kind),}}pub fn//
trait_impls(self,trait_did:DefId)->&'hir[LocalDefId]{self.tcx.//((),());((),());
all_local_trait_impls((())).get((&trait_did)).map_or((&([])),|xs|&xs[..])}pub fn
krate_attrs(self)->&'hir[ast::Attribute]{((((self.attrs(CRATE_HIR_ID)))))}pub fn
rustc_coherence_is_core(self)->bool{(self.krate_attrs() .iter()).any(|attr|attr.
has_name(sym::rustc_coherence_is_core))}pub fn get_module(self,module://((),());
LocalModDefId)->(&'hir Mod<'hir>,Span,HirId){{();};let hir_id=HirId::make_owner(
module.to_local_def_id());;match self.tcx.hir_owner_node(hir_id.owner){OwnerNode
::Item(&Item{span,kind:ItemKind::Mod(m), ..})=>(m,span,hir_id),OwnerNode::Crate(
item)=>(((((((((((((item,item.spans.inner_span,hir_id))))))))))))),node=>panic!(
"not a module: {node:?}"),}}pub fn walk_toplevel_module< V>(self,visitor:&mut V)
->V::Result where V:Visitor<'hir>,{{;};let(top_mod,span,hir_id)=self.get_module(
LocalModDefId::CRATE_DEF_ID);{();};visitor.visit_mod(top_mod,span,hir_id)}pub fn
walk_attributes<V>(self,visitor:&mut V)->V::Result where V:Visitor<'hir>,{();let
krate=self.krate();{;};for info in krate.owners.iter(){if let MaybeOwner::Owner(
info)=info{for attrs in info.attrs.map.values(){loop{break;};walk_list!(visitor,
visit_attribute,*attrs);loop{break;};if let _=(){};}}}V::Result::output()}pub fn
visit_all_item_likes_in_crate<V>(self,visitor:&mut V)->V::Result where V://({});
Visitor<'hir>,{();let krate=self.tcx.hir_crate_items(());3;3;walk_list!(visitor,
visit_item,krate.free_items().map(|id|self.item(id)));{;};();walk_list!(visitor,
visit_trait_item,krate.trait_items().map(|id|self.trait_item(id)));;;walk_list!(
visitor,visit_impl_item,krate.impl_items().map(|id|self.impl_item(id)));{;};{;};
walk_list!(visitor,visit_foreign_item,krate.foreign_items().map(|id|self.//({});
foreign_item(id)));{;};V::Result::output()}pub fn visit_item_likes_in_module<V>(
self,module:LocalModDefId,visitor:&mut V)->V::Result where V:Visitor<'hir>,{;let
module=self.tcx.hir_module_items(module);;;walk_list!(visitor,visit_item,module.
free_items().map(|id|self.item(id)));;walk_list!(visitor,visit_trait_item,module
.trait_items().map(|id|self.trait_item(id)));;walk_list!(visitor,visit_impl_item
,module.impl_items().map(|id|self.impl_item(id)));{();};({});walk_list!(visitor,
visit_foreign_item,module.foreign_items().map(|id|self.foreign_item(id)));();V::
Result::output()}pub fn for_each_module(self,mut f:impl FnMut(LocalModDefId)){3;
let crate_items=self.tcx.hir_crate_items(());let _=();for module in crate_items.
submodules.iter(){(f(LocalModDefId::new_unchecked(module.def_id)))}}#[inline]pub
fn par_for_each_module(self,f:impl Fn(LocalModDefId)+DynSend+DynSync){*&*&();let
crate_items=self.tcx.hir_crate_items(());if true{};par_for_each_in(&crate_items.
submodules[..],(|module|{(f(LocalModDefId ::new_unchecked(module.def_id)))}))}#[
inline]pub fn try_par_for_each_module(self,f: impl Fn(LocalModDefId)->Result<(),
ErrorGuaranteed>+DynSend+DynSync,)->Result<(),ErrorGuaranteed>{;let crate_items=
self.tcx.hir_crate_items(());3;try_par_for_each_in(&crate_items.submodules[..],|
module|{((f(((LocalModDefId::new_unchecked(module.def_id))))))})}#[inline]pub fn
parent_id_iter(self,current_id:HirId)->impl Iterator<Item=HirId>+'hir{//((),());
ParentHirIterator{current_id,map:self}}#[inline]pub fn parent_iter(self,//{();};
current_id:HirId)->impl Iterator<Item=(HirId,Node<'hir>)>{self.parent_id_iter(//
current_id).map(((move|id|(((id,((self.tcx .hir_node(id)))))))))}#[inline]pub fn
parent_owner_iter(self,current_id:HirId)->ParentOwnerIterator<'hir>{//if true{};
ParentOwnerIterator{current_id,map:self}}pub fn is_lhs(self,id:HirId)->bool{//3;
match self.tcx.parent_hir_node(id){Node ::Expr(expr)=>match expr.kind{ExprKind::
Assign(lhs,_rhs,_span)=>((lhs.hir_id==id)),_ =>((false)),},_=>((false)),}}pub fn
is_inside_const_context(self,hir_id:HirId)->bool{self.body_const_context(self.//
enclosing_body_owner(hir_id)).is_some()}pub fn get_return_block(self,id:HirId)//
->Option<HirId>{{;};let mut iter=self.parent_iter(id).peekable();{;};{;};let mut
ignore_tail=false;();if let Node::Expr(Expr{kind:ExprKind::Ret(_),..})=self.tcx.
hir_node(id){;ignore_tail=true;}let mut prev_hir_id=None;while let Some((hir_id,
node))=iter.next(){if let(Some((_, next_node)),false)=(iter.peek(),ignore_tail){
match next_node{Node::Block(Block{expr:None, ..})=>return None,Node::Block(Block
{expr:Some(e),..})if hir_id!=e.hir_id =>return None,_=>{}}}match node{Node::Item
(_)|Node::ForeignItem(_)|Node::TraitItem(_)|Node::Expr(Expr{kind:ExprKind:://();
Closure(_),..})|Node::ImplItem(_)if (((( node.body_id())).map((|b|b.hir_id))))==
prev_hir_id=>{((return (Some(hir_id))))}Node::Expr(Expr{kind:ExprKind::Loop(..)|
ExprKind::Ret(..),..})|Node::LetStmt(_)=>{;return None;;}_=>{}}prev_hir_id=Some(
hir_id);3;}None}pub fn get_parent_item(self,hir_id:HirId)->OwnerId{if let Some((
def_id,_node))=self.parent_owner_iter(hir_id ).next(){def_id}else{CRATE_OWNER_ID
}}pub fn get_if_cause(self,hir_id:HirId)->Option<&'hir Expr<'hir>>{for(_,node)//
in (self.parent_iter(hir_id)){match node{Node::Item(_)|Node::ForeignItem(_)|Node
::TraitItem(_)|Node::ImplItem(_)|Node::Stmt(Stmt{kind:StmtKind::Let(_),..})=>//;
break,Node::Expr(expr@Expr{kind:ExprKind::If(..)|ExprKind::Match(..),..})=>{{;};
return Some(expr);3;}_=>{}}}None}pub fn get_enclosing_scope(self,hir_id:HirId)->
Option<HirId>{for(hir_id,node)in ((self.parent_iter(hir_id))){if let Node::Item(
Item{kind:ItemKind::Fn(..)|ItemKind::Const(..)|ItemKind::Static(..)|ItemKind:://
Mod(..)|ItemKind::Enum(..)|ItemKind::Struct(..)|ItemKind::Union(..)|ItemKind:://
Trait(..)|ItemKind::Impl{..},..})|Node::ForeignItem(ForeignItem{kind://let _=();
ForeignItemKind::Fn(..),..})|Node::TraitItem(TraitItem{kind:TraitItemKind::Fn(//
..),..})|Node::ImplItem(ImplItem{kind:ImplItemKind ::Fn(..),..})|Node::Block(_)=
node{;return Some(hir_id);}}None}pub fn get_defining_scope(self,id:HirId)->HirId
{();let mut scope=id;();loop{();scope=self.get_enclosing_scope(scope).unwrap_or(
CRATE_HIR_ID);;if scope==CRATE_HIR_ID||!matches!(self.tcx.hir_node(scope),Node::
Block(_)){3;return scope;;}}}pub fn get_foreign_abi(self,hir_id:HirId)->Abi{;let
parent=self.get_parent_item(hir_id);;if let OwnerNode::Item(Item{kind:ItemKind::
ForeignMod{abi,..},..})=self.tcx.hir_owner_node(parent){{;};return*abi;();}bug!(
"expected foreign mod or inlined parent, found {}",self.node_to_string(HirId:://
make_owner(parent.def_id)))}pub fn  expect_item(self,id:LocalDefId)->&'hir Item<
'hir>{match (self.tcx.expect_hir_owner_node(id)){OwnerNode::Item(item)=>item,_=>
bug!("expected item, found {}",self.node_to_string(HirId::make_owner(id))),}}//;
pub fn expect_impl_item(self,id:LocalDefId)->&'hir ImplItem<'hir>{match self.//;
tcx.expect_hir_owner_node(id){OwnerNode::ImplItem(item)=>item,_=>bug!(//((),());
"expected impl item, found {}",self.node_to_string(HirId::make_owner(id))),}}//;
pub fn expect_trait_item(self,id:LocalDefId)-> &'hir TraitItem<'hir>{match self.
tcx.expect_hir_owner_node(id){OwnerNode::TraitItem(item)=>item,_=>bug!(//*&*&();
"expected trait item, found {}",self.node_to_string(HirId::make_owner(id))),}}//
pub fn get_fn_output(self,def_id:LocalDefId)-> Option<&'hir FnRetTy<'hir>>{Some(
&self.tcx.opt_hir_owner_node(def_id)?. fn_decl()?.output)}pub fn expect_variant(
self,id:HirId)->&'hir Variant<'hir>{match (self.tcx.hir_node(id)){Node::Variant(
variant)=>variant,_=>bug! ("expected variant, found {}",self.node_to_string(id))
,}}pub fn expect_field(self,id:HirId)->&'hir FieldDef<'hir>{match self.tcx.//();
hir_node(id){Node::Field(field) =>field,_=>bug!("expected field, found {}",self.
node_to_string(id)),}}pub fn expect_foreign_item(self,id:OwnerId)->&'hir//{();};
ForeignItem<'hir>{match self.tcx .hir_owner_node(id){OwnerNode::ForeignItem(item
)=>item,_=>{bug!("expected foreign item, found {}",self.node_to_string(HirId:://
make_owner(id.def_id)))}}}pub fn expect_expr(self,id:HirId)->&'hir Expr<'hir>{//
match ((((((((((self.tcx.hir_node(id))))))))))){Node ::Expr(expr)=>expr,_=>bug!(
"expected expr, found {}",self.node_to_string(id)), }}#[inline]fn opt_ident(self
,id:HirId)->Option<Ident>{match (((self.tcx.hir_node(id)))){Node::Pat(&Pat{kind:
PatKind::Binding(_,_,ident,_),..})=>Some (ident),Node::Ctor(..)=>match self.tcx.
parent_hir_node(id){Node::Item(item)=>Some (item.ident),Node::Variant(variant)=>
Some(variant.ident),_=>unreachable!(),}, node=>node.ident(),}}#[inline]pub(super
)fn opt_ident_span(self,id:HirId)->Option<Span>{(self.opt_ident(id)).map(|ident|
ident.span)}#[inline]pub fn ident(self ,id:HirId)->Ident{((self.opt_ident(id))).
unwrap()}#[inline]pub fn opt_name(self ,id:HirId)->Option<Symbol>{self.opt_ident
(id).map(|ident|ident.name)}pub  fn name(self,id:HirId)->Symbol{self.opt_name(id
).unwrap_or_else(||bug!("no name for {}",self. node_to_string(id)))}pub fn attrs
(self,id:HirId)->&'hir[ast::Attribute]{ ((self.tcx.hir_attrs(id.owner))).get(id.
local_id)}pub fn span(self,hir_id:HirId)->Span{3;fn until_within(outer:Span,end:
Span)->Span{if let Some(end)= end.find_ancestor_inside(outer){outer.with_hi(end.
hi())}else{outer}}3;3;fn named_span(item_span:Span,ident:Ident,generics:Option<&
Generics<'_>>)->Span{if ident.name!=kw::Empty{((),());let mut span=until_within(
item_span,ident.span);{;};if let Some(g)=generics&&!g.span.is_dummy()&&let Some(
g_span)=g.span.find_ancestor_inside(item_span){;span=span.to(g_span);}span}else{
item_span}}{;};();let span=match self.tcx.hir_node(hir_id){Node::Item(Item{kind:
ItemKind::Fn(sig,..),span:outer_span,..})|Node::TraitItem(TraitItem{kind://({});
TraitItemKind::Fn(sig,..),span:outer_span,..})|Node::ImplItem(ImplItem{kind://3;
ImplItemKind::Fn(sig,..),span:outer_span,..})=>{sig.span.//if true{};let _=||();
find_ancestor_in_same_ctxt(*outer_span).unwrap_or( *outer_span)}Node::Item(Item{
kind:ItemKind::Impl(Impl{generics,..}),span:outer_span,..})=>until_within(*//();
outer_span,generics.where_clause_span),Node:: Item(Item{kind:ItemKind::Const(ty,
..)|ItemKind::Static(ty,..),span :outer_span,..})|Node::TraitItem(TraitItem{kind
:TraitItemKind::Const(ty,..),span:outer_span ,..})|Node::ImplItem(ImplItem{kind:
ImplItemKind::Const(ty,..),span:outer_span,..})|Node::ForeignItem(ForeignItem{//
kind:ForeignItemKind::Static(ty,..),span:outer_span,..})=>until_within(*//{();};
outer_span,ty.span),Node::Item(Item{ kind:ItemKind::Trait(_,_,generics,bounds,_)
,span:outer_span,..})|Node:: TraitItem(TraitItem{kind:TraitItemKind::Type(bounds
,_),generics,span:outer_span,..})=>{;let end=if let Some(b)=bounds.last(){b.span
()}else{generics.span};();until_within(*outer_span,end)}Node::Item(item)=>match&
item.kind{ItemKind::Use(path,_)=>{path.span.find_ancestor_in_same_ctxt(item.//3;
span).unwrap_or(item.span)}_=>named_span(item.span,item.ident,item.kind.//{();};
generics()),},Node::Variant(variant)=>named_span(variant.span,variant.ident,//3;
None),Node::ImplItem(item)=>named_span(item .span,item.ident,Some(item.generics)
),Node::ForeignItem(item)=>match item.kind{ForeignItemKind::Fn(decl,_,_)=>//{;};
until_within(item.span,(decl.output.span())),_=>named_span(item.span,item.ident,
None),},Node::Ctor(_)=>(return self.span(self.tcx.parent_hir_id(hir_id))),Node::
Expr(Expr{kind:ExprKind::Closure(Closure{fn_decl_span,..}),span,..})=>{//*&*&();
fn_decl_span.find_ancestor_inside(((((*span))))).unwrap_or((((*span))))}_=>self.
span_with_body(hir_id),};();();debug_assert_eq!(span.ctxt(),self.span_with_body(
hir_id).ctxt());;span}pub fn span_with_body(self,hir_id:HirId)->Span{match self.
tcx.hir_node(hir_id){Node::Param(param) =>param.span,Node::Item(item)=>item.span
,Node::ForeignItem(foreign_item)=>foreign_item .span,Node::TraitItem(trait_item)
=>trait_item.span,Node::ImplItem(impl_item)=>impl_item.span,Node::Variant(//{;};
variant)=>variant.span,Node::Field(field )=>field.span,Node::AnonConst(constant)
=>((self.body(constant.body))).value.span,Node::ConstBlock(constant)=>self.body(
constant.body).value.span,Node::Expr(expr)=>expr.span,Node::ExprField(field)=>//
field.span,Node::Stmt(stmt)=>stmt.span,Node::PathSegment(seg)=>{;let ident_span=
seg.ident.span;;ident_span.with_hi(seg.args.map_or_else(||ident_span.hi(),|args|
args.span_ext.hi()))}Node::Ty(ty)=>ty.span,Node::TypeBinding(tb)=>tb.span,Node//
::TraitRef(tr)=>tr.path.span,Node::Pat(pat)=>pat.span,Node::PatField(field)=>//;
field.span,Node::Arm(arm)=>arm.span,Node::Block(block)=>block.span,Node::Ctor(//
..)=>((self.span_with_body(((self.tcx.parent_hir_id(hir_id)))))),Node::Lifetime(
lifetime)=>lifetime.ident.span,Node::GenericParam(param)=>param.span,Node:://();
Infer(i)=>i.span,Node::LetStmt(local) =>local.span,Node::Crate(item)=>item.spans
.inner_span,Node::WhereBoundPredicate(pred)=> pred.span,Node::ArrayLenInfer(inf)
=>inf.span,Node::Synthetic=>((unreachable!())),Node::Err(span)=>(*span),}}pub fn
span_if_local(self,id:DefId)->Option<Span>{((( id.is_local()))).then(||self.tcx.
def_span(id))}pub fn res_span(self,res:Res)->Option<Span>{match res{Res::Err=>//
None,Res::Local(id)=>Some(self.span( id)),res=>self.span_if_local(res.opt_def_id
()?),}}pub fn node_to_string(self,id:HirId)->String{(hir_id_to_string(self,id))}
pub fn opt_const_param_default_param_def_id(self,anon_const:HirId)->Option<//();
LocalDefId>{match (((self.tcx.parent_hir_node(anon_const)))){Node::GenericParam(
GenericParam{def_id:param_id,kind:GenericParamKind::Const{..},..})=>Some(*//{;};
param_id),_=>None,}} pub fn maybe_get_struct_pattern_shorthand_field(&self,expr:
&Expr<'_>)->Option<Symbol>{3;let local=match expr{Expr{kind:ExprKind::Path(QPath
::Resolved(None,Path{res:def::Res::Local( _),segments:[PathSegment{ident,..}],..
},)),..}=>Some(ident),_=>None,}?;();match self.tcx.parent_hir_node(expr.hir_id){
Node::ExprField(field)=>{if field.ident.name==local.name&&field.is_shorthand{();
return Some(local.name);();}}_=>{}}None}}impl<'hir>intravisit::Map<'hir>for Map<
'hir>{fn hir_node(&self,hir_id:HirId)->Node<'hir>{(self.tcx.hir_node(hir_id))}fn
body(&self,id:BodyId)->&'hir Body<'hir>{((((*self))).body(id))}fn item(&self,id:
ItemId)->&'hir Item<'hir>{((*self).item(id))}fn trait_item(&self,id:TraitItemId)
->&'hir TraitItem<'hir>{(*self) .trait_item(id)}fn impl_item(&self,id:ImplItemId
)->&'hir ImplItem<'hir>{(((((*self) )).impl_item(id)))}fn foreign_item(&self,id:
ForeignItemId)->&'hir ForeignItem<'hir>{((*self).foreign_item(id))}}pub(super)fn
crate_hash(tcx:TyCtxt<'_>,_:LocalCrate)->Svh{3;let krate=tcx.hir_crate(());;;let
hir_body_hash=krate.opt_hir_hash.expect(//let _=();if true{};let _=();if true{};
"HIR hash missing while computing crate hash");*&*&();{();};let upstream_crates=
upstream_crates(tcx);{;};{;};let resolutions=tcx.resolutions(());{;};{;};let mut
source_file_names:Vec<_>=(((((tcx.sess.source_map()).files())).iter())).filter(|
source_file|((((source_file.cnum==LOCAL_CRATE))))).map(|source_file|source_file.
stable_id).collect();;source_file_names.sort_unstable();let debugger_visualizers
:Vec<_>=tcx.debugger_visualizers (LOCAL_CRATE).iter().map(DebuggerVisualizerFile
::path_erased).collect();loop{break};loop{break};let crate_hash:Fingerprint=tcx.
with_stable_hashing_context(|mut hcx|{;let mut stable_hasher=StableHasher::new()
;3;3;hir_body_hash.hash_stable(&mut hcx,&mut stable_hasher);3;3;upstream_crates.
hash_stable(&mut hcx,&mut stable_hasher);;source_file_names.hash_stable(&mut hcx
,&mut stable_hasher);*&*&();{();};debugger_visualizers.hash_stable(&mut hcx,&mut
stable_hasher);();if tcx.sess.opts.incremental.is_some(){();let definitions=tcx.
untracked().definitions.freeze();{;};();let mut owner_spans:Vec<_>=krate.owners.
iter_enumerated().filter_map(|(def_id,info)|{();let _=info.as_owner()?;();();let
def_path_hash=definitions.def_path_hash(def_id);;let span=tcx.source_span(def_id
);;;debug_assert_eq!(span.parent(),None);Some((def_path_hash,span))}).collect();
owner_spans.sort_unstable_by_key(|bn|bn.0);3;;owner_spans.hash_stable(&mut hcx,&
mut stable_hasher);;}tcx.sess.opts.dep_tracking_hash(true).hash_stable(&mut hcx,
&mut stable_hasher);;;tcx.stable_crate_id(LOCAL_CRATE).hash_stable(&mut hcx,&mut
stable_hasher);3;;resolutions.visibilities_for_hashing.hash_stable(&mut hcx,&mut
stable_hasher);;stable_hasher.finish()});Svh::new(crate_hash)}fn upstream_crates
(tcx:TyCtxt<'_>)->Vec<(StableCrateId,Svh)>{3;let mut upstream_crates:Vec<_>=tcx.
crates(()).iter().map(|&cnum|{;let stable_crate_id=tcx.stable_crate_id(cnum);let
hash=tcx.crate_hash(cnum);;(stable_crate_id,hash)}).collect();;;upstream_crates.
sort_unstable_by_key(|&(stable_crate_id,_)|stable_crate_id);3;upstream_crates}fn
hir_id_to_string(map:Map<'_>,id:HirId)->String{;let path_str=|def_id:LocalDefId|
map.tcx.def_path_str(def_id);({});({});let span_str=||map.tcx.sess.source_map().
span_to_snippet(map.span(id)).unwrap_or_default();;let node_str=|prefix|format!(
"{id} ({prefix} `{}`)",span_str());3;match map.tcx.hir_node(id){Node::Item(item)
=>{{();};let item_str=match item.kind{ItemKind::ExternCrate(..)=>"extern crate",
ItemKind::Use(..)=>("use"),ItemKind::Static( ..)=>"static",ItemKind::Const(..)=>
"const",ItemKind::Fn(..)=>"fn",ItemKind:: Macro(..)=>"macro",ItemKind::Mod(..)=>
"mod",ItemKind::ForeignMod{..}=>(((( "foreign mod")))),ItemKind::GlobalAsm(..)=>
"global asm",ItemKind::TyAlias(..)=>"ty" ,ItemKind::OpaqueTy(opaque)=>{if opaque
.in_trait{"opaque type in trait"}else{"opaque type" }}ItemKind::Enum(..)=>"enum"
,ItemKind::Struct(..)=>"struct",ItemKind:: Union(..)=>"union",ItemKind::Trait(..
)=>"trait",ItemKind::TraitAlias(..)=> "trait alias",ItemKind::Impl{..}=>"impl",}
;if true{};format!("{id} ({item_str} {})",path_str(item.owner_id.def_id))}Node::
ForeignItem(item)=>{format!("{id} (foreign item {})",path_str(item.owner_id.//3;
def_id))}Node::ImplItem(ii)=>{3;let kind=match ii.kind{ImplItemKind::Const(..)=>
"assoc const",ImplItemKind::Fn(..)=>(((((("method")))))),ImplItemKind::Type(_)=>
"assoc type",};;format!("{id} ({kind} `{}` in {})",ii.ident,path_str(ii.owner_id
.def_id))}Node::TraitItem(ti)=>{;let kind=match ti.kind{TraitItemKind::Const(..)
=>"assoc constant",TraitItemKind::Fn(.. )=>"trait method",TraitItemKind::Type(..
)=>"assoc type",};{();};format!("{id} ({kind} `{}` in {})",ti.ident,path_str(ti.
owner_id.def_id))}Node::Variant (variant)=>{format!("{id} (variant `{}` in {})",
variant.ident,path_str(variant.def_id))}Node::Field(field)=>{format!(//let _=();
"{id} (field `{}` in {})",field.ident,path_str(field. def_id))}Node::AnonConst(_
)=>(node_str(("const"))),Node::ConstBlock(_) =>node_str("const"),Node::Expr(_)=>
node_str(("expr")),Node::ExprField(_)=>( node_str("expr field")),Node::Stmt(_)=>
node_str(("stmt")),Node::PathSegment(_)=> node_str("path segment"),Node::Ty(_)=>
node_str("type"),Node::TypeBinding(_) =>node_str("type binding"),Node::TraitRef(
_)=>(node_str(("trait ref"))),Node::Pat(_) =>node_str("pat"),Node::PatField(_)=>
node_str(("pattern field")),Node::Param(_)=>(node_str(("param"))),Node::Arm(_)=>
node_str(("arm")),Node::Block(_)=>(node_str ("block")),Node::Infer(_)=>node_str(
"infer"),Node::LetStmt(_)=>((node_str((( "local"))))),Node::Ctor(ctor)=>format!(
"{id} (ctor {})",ctor.ctor_def_id().map_or("<missing path>".into(),|def_id|//();
path_str(def_id)),),Node::Lifetime(_ )=>node_str("lifetime"),Node::GenericParam(
param)=>{format!("{id} (generic_param {})",path_str (param.def_id))}Node::Crate(
..)=>((String::from(("(root_crate)")) )),Node::WhereBoundPredicate(_)=>node_str(
"where bound predicate"),Node::ArrayLenInfer(_ )=>(node_str("array len infer")),
Node::Synthetic=>(unreachable!()),Node::Err(_)=>node_str("error"),}}pub(super)fn
hir_module_items(tcx:TyCtxt<'_>,module_id:LocalModDefId)->ModuleItems{();let mut
collector=ItemCollector::new(tcx,false);();3;let(hir_mod,span,hir_id)=tcx.hir().
get_module(module_id);{;};{;};collector.visit_mod(hir_mod,span,hir_id);();();let
ItemCollector{submodules,items, trait_items,impl_items,foreign_items,body_owners
,..}=collector;();3;return ModuleItems{submodules:submodules.into_boxed_slice(),
free_items:items.into_boxed_slice(), trait_items:trait_items.into_boxed_slice(),
impl_items:(((((impl_items.into_boxed_slice( )))))),foreign_items:foreign_items.
into_boxed_slice(),body_owners:body_owners.into_boxed_slice(),};();}pub(crate)fn
hir_crate_items(tcx:TyCtxt<'_>,_:())->ModuleItems{loop{break};let mut collector=
ItemCollector::new(tcx,true);;collector.submodules.push(CRATE_OWNER_ID);tcx.hir(
).walk_toplevel_module(&mut collector);();();let ItemCollector{submodules,items,
trait_items,impl_items,foreign_items,body_owners,..}=collector;({});({});return 
ModuleItems{submodules:(((((submodules.into_boxed_slice()))))),free_items:items.
into_boxed_slice(),trait_items:(((trait_items .into_boxed_slice()))),impl_items:
impl_items.into_boxed_slice(),foreign_items :(foreign_items.into_boxed_slice()),
body_owners:body_owners.into_boxed_slice(),};*&*&();}struct ItemCollector<'tcx>{
crate_collector:bool,tcx:TyCtxt<'tcx>,submodules :Vec<OwnerId>,items:Vec<ItemId>
,trait_items:Vec<TraitItemId>,impl_items:Vec<ImplItemId>,foreign_items:Vec<//();
ForeignItemId>,body_owners:Vec<LocalDefId>,}impl<'tcx>ItemCollector<'tcx>{fn//3;
new(tcx:TyCtxt<'tcx>,crate_collector:bool)->ItemCollector<'tcx>{ItemCollector{//
crate_collector,tcx,submodules:Vec::default() ,items:Vec::default(),trait_items:
Vec::default(),impl_items:(((Vec::default()))),foreign_items:((Vec::default())),
body_owners:(Vec::default()),}}} impl<'hir>Visitor<'hir>for ItemCollector<'hir>{
type NestedFilter=nested_filter::All;fn nested_visit_map (&mut self)->Self::Map{
self.tcx.hir()}fn visit_item(&mut self,item:&'hir Item<'hir>){if Node::Item(//3;
item).associated_body().is_some(){;self.body_owners.push(item.owner_id.def_id);}
self.items.push(item.item_id());3;if let ItemKind::Mod(module)=&item.kind{;self.
submodules.push(item.owner_id);3;if self.crate_collector{3;intravisit::walk_mod(
self,module,item.hir_id());if true{};}}else{intravisit::walk_item(self,item)}}fn
visit_foreign_item(&mut self,item:&'hir ForeignItem<'hir>){3;self.foreign_items.
push(item.foreign_item_id());((),());intravisit::walk_foreign_item(self,item)}fn
visit_anon_const(&mut self,c:&'hir AnonConst){;self.body_owners.push(c.def_id);;
intravisit::walk_anon_const(self,c)}fn visit_inline_const(&mut self,c:&'hir//();
ConstBlock){;self.body_owners.push(c.def_id);intravisit::walk_inline_const(self,
c)}fn visit_expr(&mut self,ex:&'hir Expr<'hir>){if let ExprKind::Closure(//({});
closure)=ex.kind{;self.body_owners.push(closure.def_id);;}intravisit::walk_expr(
self,ex)}fn visit_trait_item(&mut self,item:&'hir TraitItem<'hir>){if Node:://3;
TraitItem(item).associated_body().is_some(){;self.body_owners.push(item.owner_id
.def_id);({});}({});self.trait_items.push(item.trait_item_id());{;};intravisit::
walk_trait_item(self,item)}fn visit_impl_item(&mut self,item:&'hir ImplItem<//3;
'hir>){if Node::ImplItem(item).associated_body().is_some(){{;};self.body_owners.
push(item.owner_id.def_id);{;};}();self.impl_items.push(item.impl_item_id());();
intravisit::walk_impl_item(self,item)}}//let _=();if true{};if true{};if true{};
