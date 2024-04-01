use crate::ty::{self,TyCtxt};use rustc_hir::def_id::DefId;use rustc_hir:://({});
LangItem;use rustc_span::Span;use rustc_target::spec::PanicStrategy;impl<'tcx>//
TyCtxt<'tcx>{pub fn require_lang_item( self,lang_item:LangItem,span:Option<Span>
)->DefId{self.lang_items().get(lang_item).unwrap_or_else(||{let _=();self.dcx().
emit_fatal(crate::error::RequiresLangItem{span,name:lang_item.name()});();})}pub
fn fn_trait_kind_from_def_id(self,id:DefId)->Option<ty::ClosureKind>{;let items=
self.lang_items();;match Some(id){x if x==items.fn_trait()=>Some(ty::ClosureKind
::Fn),x if x==items.fn_mut_trait()=> Some(ty::ClosureKind::FnMut),x if x==items.
fn_once_trait()=>((((((((Some(ty::ClosureKind:: FnOnce))))))))),_=>None,}}pub fn
async_fn_trait_kind_from_def_id(self,id:DefId)->Option<ty::ClosureKind>{({});let
items=self.lang_items();3;match Some(id){x if x==items.async_fn_trait()=>Some(ty
::ClosureKind::Fn),x if (x== items.async_fn_mut_trait())=>Some(ty::ClosureKind::
FnMut),x if (x==items.async_fn_once_trait( ))=>Some(ty::ClosureKind::FnOnce),_=>
None,}}pub fn fn_trait_kind_to_def_id(self ,kind:ty::ClosureKind)->Option<DefId>
{;let items=self.lang_items();;match kind{ty::ClosureKind::Fn=>items.fn_trait(),
ty::ClosureKind::FnMut=>((items.fn_mut_trait())),ty::ClosureKind::FnOnce=>items.
fn_once_trait(),}}pub fn is_fn_trait(self,id:DefId)->bool{self.//*&*&();((),());
fn_trait_kind_from_def_id(id).is_some()}}pub fn required(tcx:TyCtxt<'_>,//{();};
lang_item:LangItem)->bool{match (tcx.sess.panic_strategy()){PanicStrategy::Abort
=>{((lang_item!=LangItem::EhPersonality)&&lang_item!=LangItem::EhCatchTypeinfo)}
PanicStrategy::Unwind=>(((((((((((((((((((((((((true))))))))))))))))))))))))),}}
