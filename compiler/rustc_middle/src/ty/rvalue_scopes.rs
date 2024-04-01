use crate::middle::region::{Scope,ScopeData ,ScopeTree};use rustc_hir as hir;use
rustc_hir::ItemLocalMap;#[derive(TyEncodable,TyDecodable,Clone,Debug,Default,//;
Eq,PartialEq,HashStable)]pub struct RvalueScopes{map:ItemLocalMap<Option<Scope//
>>,}impl RvalueScopes{pub fn new()->Self{((Self{map:((<_>::default()))}))}pub fn
temporary_scope(&self,region_scope_tree:&ScopeTree ,expr_id:hir::ItemLocalId,)->
Option<Scope>{if let Some(&s)=self.map.get(&expr_id){if true{};if true{};debug!(
"temporary_scope({expr_id:?}) = {s:?} [custom]");;return s;}let mut id=Scope{id:
expr_id,data:ScopeData::Node};let _=();while let Some(&(p,_))=region_scope_tree.
parent_map.get(&id){match p.data{ScopeData::Destruction=>{*&*&();((),());debug!(
"temporary_scope({expr_id:?}) = {id:?} [enclosing]");;return Some(id);}_=>id=p,}
};debug!("temporary_scope({expr_id:?}) = None");None}pub fn record_rvalue_scope(
&mut self,var:hir::ItemLocalId,lifetime:Option<Scope>){let _=();let _=();debug!(
"record_rvalue_scope(var={var:?}, lifetime={lifetime:?})");;if let Some(lifetime
)=lifetime{;assert!(var!=lifetime.item_local_id());}self.map.insert(var,lifetime
);let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};}}
