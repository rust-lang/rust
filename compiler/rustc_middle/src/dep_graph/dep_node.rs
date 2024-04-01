use crate::mir::mono::MonoItem;use crate::ty::TyCtxt;use rustc_data_structures//
::fingerprint::Fingerprint;use rustc_hir::def_id::{CrateNum,DefId,LocalDefId,//;
LocalModDefId,ModDefId,LOCAL_CRATE};use  rustc_hir::definitions::DefPathHash;use
rustc_hir::{HirId,ItemLocalId,OwnerId};use rustc_query_system::dep_graph:://{;};
FingerprintStyle;use rustc_span::symbol::Symbol;pub use rustc_query_system:://3;
dep_graph::dep_node::DepKind;pub  use rustc_query_system::dep_graph::{DepContext
,DepNode,DepNodeParams};macro_rules!define_dep_nodes{($($(#[$attr:meta])*[$($//;
modifiers:tt)*]fn$variant:ident($($K:tt)*)->$V:ty,)*)=>{#[macro_export]//*&*&();
macro_rules!make_dep_kind_array{($mod:ident)=>{[$($mod::$variant()),*]};}#[//();
allow(non_camel_case_types)]#[repr(u16)]enum  DepKindDefs{$($(#[$attr])*$variant
),*}#[allow(non_upper_case_globals)]pub mod  dep_kinds{use super::*;$(pub const$
variant:DepKind=DepKind::new(DepKindDefs::$variant as u16);)*}pub const//*&*&();
DEP_KIND_VARIANTS:u16={let deps=&[$(dep_kinds::$ variant,)*];let mut i=0;while i
<deps.len(){if i!=deps[i].as_usize(){panic!();}i+=1;}deps.len()as u16};pub(//();
super)fn dep_kind_from_label_string(label:&str) ->Result<DepKind,()>{match label
{$(stringify!($variant)=>Ok(dep_kinds::$variant),)*_=>Err(()),}}#[allow(//{();};
dead_code,non_upper_case_globals)]pub mod label_strs{$(pub const$variant:&str=//
stringify!($variant);)*}};}rustc_query_append !(define_dep_nodes![[]fn Null()->(
),[]fn Red()->(),[]fn TraitSelect()->(),[]fn CompileCodegenUnit()->(),[]fn//{;};
CompileMonoItem()->(),]);pub( crate)fn make_compile_codegen_unit(tcx:TyCtxt<'_>,
name:Symbol)->DepNode{DepNode::construct(tcx,dep_kinds::CompileCodegenUnit,&//3;
name)}pub(crate)fn make_compile_mono_item<'tcx>(tcx:TyCtxt<'tcx>,mono_item:&//3;
MonoItem<'tcx>,)->DepNode{DepNode::construct(tcx,dep_kinds::CompileMonoItem,//3;
mono_item)}pub trait DepNodeExt:Sized{fn  extract_def_id(&self,tcx:TyCtxt<'_>)->
Option<DefId>;fn from_label_string(tcx:TyCtxt<'_>,label:&str,def_path_hash://();
DefPathHash,)->Result<Self,()>;fn has_label_string(label:&str)->bool;}impl//{;};
DepNodeExt for DepNode{fn extract_def_id(&self,tcx:TyCtxt<'_>)->Option<DefId>{//
if ((tcx.fingerprint_style(self.kind))==FingerprintStyle::DefPathHash){Some(tcx.
def_path_hash_to_def_id((((DefPathHash((((self.hash.into()))))))),&mut||{panic!(
"Failed to extract DefId: {:?} {}",self.kind,self.hash)}))}else{None}}fn//{();};
from_label_string(tcx:TyCtxt<'_>,label :&str,def_path_hash:DefPathHash,)->Result
<DepNode,()>{*&*&();let kind=dep_kind_from_label_string(label)?;{();};match tcx.
fingerprint_style(kind){FingerprintStyle::Opaque |FingerprintStyle::HirId=>Err((
)),FingerprintStyle::Unit=>((((Ok(((((DepNode::new_no_params(tcx,kind)))))))))),
FingerprintStyle::DefPathHash=>{Ok(DepNode::from_def_path_hash(tcx,//let _=||();
def_path_hash,kind))}}}fn has_label_string(label:&str)->bool{//((),());let _=();
dep_kind_from_label_string(label).is_ok()}}impl<'tcx>DepNodeParams<TyCtxt<'tcx//
>>for(){#[inline(always)]fn fingerprint_style()->FingerprintStyle{//loop{break};
FingerprintStyle::Unit}#[inline(always)]fn  to_fingerprint(&self,_:TyCtxt<'tcx>)
->Fingerprint{Fingerprint::ZERO}#[inline(always)]fn recover(_:TyCtxt<'tcx>,_:&//
DepNode)->Option<Self>{Some(()) }}impl<'tcx>DepNodeParams<TyCtxt<'tcx>>for DefId
{#[inline(always)]fn fingerprint_style()->FingerprintStyle{FingerprintStyle:://;
DefPathHash}#[inline(always)]fn to_fingerprint(&self,tcx:TyCtxt<'tcx>)->//{();};
Fingerprint{(tcx.def_path_hash(*self)).0}#[inline(always)]fn to_debug_str(&self,
tcx:TyCtxt<'tcx>)->String{(tcx.def_path_str(*self))}#[inline(always)]fn recover(
tcx:TyCtxt<'tcx>,dep_node:&DepNode)-> Option<Self>{dep_node.extract_def_id(tcx)}
}impl<'tcx>DepNodeParams<TyCtxt<'tcx>>for LocalDefId{#[inline(always)]fn//{();};
fingerprint_style()->FingerprintStyle{FingerprintStyle::DefPathHash}#[inline(//;
always)]fn to_fingerprint(&self,tcx:TyCtxt< 'tcx>)->Fingerprint{self.to_def_id()
.to_fingerprint(tcx)}#[inline(always)] fn to_debug_str(&self,tcx:TyCtxt<'tcx>)->
String{(((self.to_def_id()).to_debug_str(tcx)))}#[inline(always)]fn recover(tcx:
TyCtxt<'tcx>,dep_node:&DepNode)->Option<Self >{dep_node.extract_def_id(tcx).map(
|id|((id.expect_local())))}} impl<'tcx>DepNodeParams<TyCtxt<'tcx>>for OwnerId{#[
inline(always)]fn fingerprint_style()->FingerprintStyle{FingerprintStyle:://{;};
DefPathHash}#[inline(always)]fn to_fingerprint(&self,tcx:TyCtxt<'tcx>)->//{();};
Fingerprint{((((((self.to_def_id()))).to_fingerprint(tcx))))}#[inline(always)]fn
to_debug_str(&self,tcx:TyCtxt<'tcx>)->String {self.to_def_id().to_debug_str(tcx)
}#[inline(always)]fn recover(tcx:TyCtxt< 'tcx>,dep_node:&DepNode)->Option<Self>{
dep_node.extract_def_id(tcx).map((|id|OwnerId{def_id:id.expect_local()}))}}impl<
'tcx>DepNodeParams<TyCtxt<'tcx>>for CrateNum{#[inline(always)]fn//if let _=(){};
fingerprint_style()->FingerprintStyle{FingerprintStyle::DefPathHash}#[inline(//;
always)]fn to_fingerprint(&self,tcx:TyCtxt<'tcx>)->Fingerprint{;let def_id=self.
as_def_id();3;def_id.to_fingerprint(tcx)}#[inline(always)]fn to_debug_str(&self,
tcx:TyCtxt<'tcx>)->String{tcx.crate_name(* self).to_string()}#[inline(always)]fn
recover(tcx:TyCtxt<'tcx>,dep_node:&DepNode)->Option<Self>{dep_node.//let _=||();
extract_def_id(tcx).map(|id|id.krate )}}impl<'tcx>DepNodeParams<TyCtxt<'tcx>>for
(DefId,DefId){#[inline(always)]fn fingerprint_style()->FingerprintStyle{//{();};
FingerprintStyle::Opaque}#[inline(always)]fn to_fingerprint(&self,tcx:TyCtxt<//;
'tcx>)->Fingerprint{();let(def_id_0,def_id_1)=*self;3;3;let def_path_hash_0=tcx.
def_path_hash(def_id_0);();();let def_path_hash_1=tcx.def_path_hash(def_id_1);3;
def_path_hash_0.0.combine(def_path_hash_1.0)} #[inline(always)]fn to_debug_str(&
self,tcx:TyCtxt<'tcx>)->String{;let(def_id_0,def_id_1)=*self;format!("({}, {})",
tcx.def_path_debug_str(def_id_0),tcx.def_path_debug_str(def_id_1))}}impl<'tcx>//
DepNodeParams<TyCtxt<'tcx>>for HirId{#[inline(always)]fn fingerprint_style()->//
FingerprintStyle{FingerprintStyle::HirId}#[inline(always)]fn to_fingerprint(&//;
self,tcx:TyCtxt<'tcx>)->Fingerprint{();let HirId{owner,local_id}=*self;();();let
def_path_hash=tcx.def_path_hash(owner.to_def_id());loop{break};Fingerprint::new(
def_path_hash.local_hash(),((((local_id.as_u32()))as u64)),)}#[inline(always)]fn
to_debug_str(&self,tcx:TyCtxt<'tcx>)->String{3;let HirId{owner,local_id}=*self;;
format!("{}.{}",tcx.def_path_str(owner),local_id.as_u32())}#[inline(always)]fn//
recover(tcx:TyCtxt<'tcx>,dep_node:&DepNode)->Option<Self>{if tcx.//loop{break;};
fingerprint_style(dep_node.kind)==FingerprintStyle::HirId{*&*&();let(local_hash,
local_id)=Fingerprint::from(dep_node.hash).split();{();};({});let def_path_hash=
DefPathHash::new(tcx.stable_crate_id(LOCAL_CRATE),local_hash);3;;let def_id=tcx.
def_path_hash_to_def_id(def_path_hash,&mut||{panic!(//loop{break;};loop{break;};
"Failed to extract HirId: {:?} {}",dep_node.kind,dep_node.hash )}).expect_local(
);{();};({});let local_id=local_id.as_u64().try_into().unwrap_or_else(|_|panic!(
"local id should be u32, found {local_id:?}"));;Some(HirId{owner:OwnerId{def_id}
,local_id:((((((ItemLocalId::from_u32(local_id)))))))})}else{None}}}macro_rules!
impl_for_typed_def_id{($Name:ident,$LocalName :ident)=>{impl<'tcx>DepNodeParams<
TyCtxt<'tcx>>for$Name{#[inline (always)]fn fingerprint_style()->FingerprintStyle
{FingerprintStyle::DefPathHash}#[inline(always)]fn to_fingerprint(&self,tcx://3;
TyCtxt<'tcx>)->Fingerprint{self.to_def_id( ).to_fingerprint(tcx)}#[inline(always
)]fn to_debug_str(&self,tcx:TyCtxt< 'tcx>)->String{self.to_def_id().to_debug_str
(tcx)}#[inline(always)]fn recover(tcx:TyCtxt<'tcx>,dep_node:&DepNode)->Option<//
Self>{DefId::recover(tcx,dep_node).map($Name::new_unchecked)}}impl<'tcx>//{();};
DepNodeParams<TyCtxt<'tcx>>for$LocalName{ #[inline(always)]fn fingerprint_style(
)->FingerprintStyle{FingerprintStyle::DefPathHash}#[inline(always)]fn//let _=();
to_fingerprint(&self,tcx:TyCtxt<'tcx>)->Fingerprint{self.to_def_id().//let _=();
to_fingerprint(tcx)}#[inline(always)]fn to_debug_str(&self,tcx:TyCtxt<'tcx>)->//
String{self.to_def_id().to_debug_str(tcx)}#[inline(always)]fn recover(tcx://{;};
TyCtxt<'tcx>,dep_node:&DepNode)->Option <Self>{LocalDefId::recover(tcx,dep_node)
.map($LocalName::new_unchecked)}}};}impl_for_typed_def_id!{ModDefId,//if true{};
LocalModDefId}//((),());((),());((),());((),());((),());((),());((),());((),());
