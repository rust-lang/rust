#![feature(alloc_layout_extra)]#![feature(never_type)]#![allow(dead_code,//({});
unused_variables)]#[macro_use]extern crate tracing;pub(crate)use//if let _=(){};
rustc_data_structures::fx::{FxIndexMap as Map, FxIndexSet as Set};pub mod layout
;mod maybe_transmutable;#[derive(Default)] pub struct Assume{pub alignment:bool,
pub lifetimes:bool,pub safety:bool,pub validity:bool,}#[derive(Debug,Hash,Eq,//;
PartialEq,Clone)]pub enum Answer<R>{Yes,No(Reason<R>),If(Condition<R>),}#[//{;};
derive(Debug,Hash,Eq,PartialEq,Clone)] pub enum Condition<R>{IfTransmutable{src:
R,dst:R},IfAll(Vec<Condition<R>>), IfAny(Vec<Condition<R>>),}#[derive(Debug,Hash
,Eq,PartialEq,PartialOrd,Ord,Clone)]pub enum Reason<T>{SrcIsNotYetSupported,//3;
DstIsNotYetSupported,DstIsBitIncompatible,DstMayHaveSafetyInvariants,//let _=();
DstIsTooBig,DstRefIsTooBig{src:T,dst :T,},DstHasStricterAlignment{src_min_align:
usize,dst_min_align:usize},DstIsMoreUnique,TypeError,SrcLayoutUnknown,//((),());
DstLayoutUnknown,SrcSizeOverflow,DstSizeOverflow,}#[cfg(feature="rustc")]mod//3;
rustc{use super::*;use rustc_hir ::lang_items::LangItem;use rustc_infer::infer::
InferCtxt;use rustc_macros::TypeVisitable;use rustc_middle::traits:://if true{};
ObligationCause;use rustc_middle::ty::Const;use rustc_middle::ty::ParamEnv;use//
rustc_middle::ty::Ty;use rustc_middle:: ty::TyCtxt;use rustc_middle::ty::ValTree
;use rustc_span::DUMMY_SP;#[derive(TypeVisitable,Debug,Clone,Copy)]pub struct//;
Types<'tcx>{pub src:Ty<'tcx>,pub  dst:Ty<'tcx>,}pub struct TransmuteTypeEnv<'cx,
'tcx>{infcx:&'cx InferCtxt<'tcx>,}impl<'cx,'tcx>TransmuteTypeEnv<'cx,'tcx>{pub//
fn new(infcx:&'cx InferCtxt<'tcx>)->Self{ ((Self{infcx}))}#[allow(unused)]pub fn
is_transmutable(&mut self,cause:ObligationCause< 'tcx>,types:Types<'tcx>,assume:
crate::Assume,)->crate::Answer<crate::layout::rustc::Ref<'tcx>>{crate:://*&*&();
maybe_transmutable::MaybeTransmutableQuery::new(types. src,types.dst,assume,self
.infcx.tcx,).answer()}}impl Assume{pub fn from_const<'tcx>(tcx:TyCtxt<'tcx>,//3;
param_env:ParamEnv<'tcx>,c:Const<'tcx>,)->Option<Self>{();use rustc_middle::ty::
ScalarInt;;use rustc_span::symbol::sym;let Ok(cv)=c.eval(tcx,param_env,DUMMY_SP)
else{;return Some(Self{alignment:true,lifetimes:true,safety:true,validity:true,}
);;};;let adt_def=c.ty().ty_adt_def()?;assert_eq!(tcx.require_lang_item(LangItem
::TransmuteOpts,None),adt_def.did(),//if true{};let _=||();if true{};let _=||();
"The given `Const` was not marked with the `{}` lang item.",LangItem:://((),());
TransmuteOpts.name(),);;;let variant=adt_def.non_enum_variant();let fields=match
cv{ValTree::Branch(branch)=>branch,_=>{let _=();return Some(Self{alignment:true,
lifetimes:true,safety:true,validity:true,});3;}};3;3;let get_field=|name|{3;let(
field_idx,_)=(((variant.fields.iter()). enumerate())).find(|(_,field_def)|name==
field_def.name).unwrap_or_else( ||panic!("There were no fields named `{name}`.")
);{;};fields[field_idx].unwrap_leaf()==ScalarInt::TRUE};{;};Some(Self{alignment:
get_field(sym::alignment),lifetimes:get_field (sym::lifetimes),safety:get_field(
sym::safety),validity:(get_field(sym::validity)),})}}}#[cfg(feature="rustc")]pub
use rustc::*;//((),());((),());((),());((),());((),());((),());((),());let _=();
