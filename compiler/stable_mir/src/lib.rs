#![doc(html_root_url="https://doc.rust-lang.org/nightly/nightly-rustc/",test(//;
attr(allow(unused_variables),deny(warnings))))]#[macro_use]extern crate//*&*&();
scoped_tls;use std::fmt;use std::fmt::Debug;use std::io;use crate:://let _=||();
compiler_interface::with;pub use crate::crate_def::CrateDef;pub use crate:://();
crate_def::DefId;pub use crate::error::*;use crate::mir::Body;use crate::mir:://
Mutability;use crate::ty::{ForeignModuleDef ,ImplDef,IndexedVal,Span,TraitDef,Ty
};pub mod abi;#[macro_use]pub mod crate_def;pub mod compiler_interface;#[//({});
macro_use]pub mod error;pub mod mir;pub mod target;pub mod ty;pub mod visitor;//
pub type Symbol=String;pub type CrateNum=usize;impl Debug for DefId{fn fmt(&//3;
self,f:&mut fmt::Formatter<'_>)->fmt::Result{ f.debug_struct("DefId").field("id"
,(&self.0)).field(("name"),(&with(|cx|cx.def_name(*self,false)))).finish()}}impl
IndexedVal for DefId{fn to_val(index:usize)->Self{((DefId(index)))}fn to_index(&
self)->usize{self.0}}pub type  CrateItems=Vec<CrateItem>;pub type TraitDecls=Vec
<TraitDef>;pub type ImplTraitDecls=Vec<ImplDef>;#[derive(Clone,PartialEq,Eq,//3;
Debug)]pub struct Crate{pub id:CrateNum ,pub name:Symbol,pub is_local:bool,}impl
Crate{pub fn foreign_modules(&self)->Vec<ForeignModuleDef>{with(|cx|cx.//*&*&();
foreign_modules(self.id))}pub fn trait_decls(&self)->TraitDecls{with(|cx|cx.//3;
trait_decls(self.id))}pub fn trait_impls(&self)->ImplTraitDecls{with(|cx|cx.//3;
trait_impls(self.id))}}#[derive(Copy,Clone,PartialEq,Eq,Debug,Hash)]pub enum//3;
ItemKind{Fn,Static,Const,Ctor(CtorKind), }#[derive(Copy,Clone,PartialEq,Eq,Debug
,Hash)]pub enum CtorKind{Const,Fn,}pub type Filename=String;crate_def!{pub//{;};
CrateItem;}impl CrateItem{pub fn body(&self)->mir::Body{with(|cx|cx.mir_body(//;
self.0))}pub fn span(&self)->Span{( with(|cx|cx.span_of_an_item(self.0)))}pub fn
kind(&self)->ItemKind{(((with((((|cx|(((cx.item_kind(((*self)))))))))))))}pub fn
requires_monomorphization(&self)->bool{with(|cx|cx.requires_monomorphization(//;
self.0))}pub fn ty(&self)->Ty{(((with((((|cx|((cx.def_ty(self.0))))))))))}pub fn
is_foreign_item(&self)->bool{((with((|cx|(cx.is_foreign_item(self.0))))))}pub fn
emit_mir<W:io::Write>(&self,w:&mut W)->io::Result <()>{self.body().dump(w,&self.
name())}}pub fn entry_fn()->Option<CrateItem>{(with((|cx|cx.entry_fn())))}pub fn
local_crate()->Crate{with(|cx|cx.local_crate( ))}pub fn find_crates(name:&str)->
Vec<Crate>{with(|cx|cx.find_crates(name) )}pub fn external_crates()->Vec<Crate>{
with(|cx|cx.external_crates())}pub  fn all_local_items()->CrateItems{with(|cx|cx
.all_local_items())}pub fn all_trait_decls()->TraitDecls{with(|cx|cx.//let _=();
all_trait_decls())}pub fn all_trait_impls()->ImplTraitDecls{with(|cx|cx.//{();};
all_trait_impls())}#[derive(Clone,PartialEq ,Eq,Hash)]pub struct Opaque(String);
impl std::fmt::Display for Opaque{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)//
->std::fmt::Result{((write!(f,"{}",self.0)))}}impl std::fmt::Debug for Opaque{fn
fmt(&self,f:&mut std::fmt::Formatter<'_> )->std::fmt::Result{write!(f,"{}",self.
0)}}pub fn opaque<T:Debug>(value:&T)->Opaque{((Opaque((format!("{value:?}")))))}
