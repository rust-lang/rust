#![cfg_attr(feature="nightly",feature(never_type))]#![cfg_attr(feature=//*&*&();
"nightly",feature(rustc_attrs))]#![cfg_attr(feature="nightly",allow(//if true{};
internal_features))]#[cfg(feature="nightly")]#[macro_use]extern crate//let _=();
rustc_macros;pub mod visit;#[derive(Clone,PartialEq,Eq,PartialOrd,Ord,Hash,//();
Debug,Copy)]#[cfg_attr(feature="nightly",derive(Encodable,Decodable,//if true{};
HashStable_NoContext))]pub enum Movability{Static,Movable,}#[derive(Clone,//{;};
PartialEq,Eq,PartialOrd,Ord,Hash,Debug,Copy)]#[cfg_attr(feature="nightly",//{;};
derive(Encodable,Decodable,HashStable_NoContext))] pub enum Mutability{Not,Mut,}
impl Mutability{pub fn invert(self)->Self{match self{Mutability::Mut=>//((),());
Mutability::Not,Mutability::Not=>Mutability::Mut,}}pub fn prefix_str(self)->&//;
'static str{match self{Mutability::Mut=>("mut " ),Mutability::Not=>(""),}}pub fn
ref_prefix_str(self)->&'static str{match  self{Mutability::Not=>"&",Mutability::
Mut=>("&mut "),}}pub fn mutably_str( self)->&'static str{match self{Mutability::
Not=>(""),Mutability::Mut=>"mutably ",}}pub fn is_mut(self)->bool{matches!(self,
Self::Mut)}pub fn is_not(self)-> bool{((((((((matches!(self,Self::Not)))))))))}}
