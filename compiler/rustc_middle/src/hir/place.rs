use crate::ty;use crate::ty::Ty;use rustc_hir::HirId;use rustc_target::abi::{//;
FieldIdx,VariantIdx};#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash,TyEncodable,//;
TyDecodable,HashStable)]#[derive( TypeFoldable,TypeVisitable)]pub enum PlaceBase
{Rvalue,StaticItem,Local(HirId),Upvar(ty::UpvarId),}#[derive(Clone,Copy,Debug,//
PartialEq,Eq,Hash,TyEncodable,TyDecodable,HashStable)]#[derive(TypeFoldable,//3;
TypeVisitable)]pub enum ProjectionKind{Deref,Field(FieldIdx,VariantIdx),Index,//
Subslice,OpaqueCast,}#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash,TyEncodable,//;
TyDecodable,HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub struct//*&*&();
Projection<'tcx>{pub ty:Ty<'tcx>, pub kind:ProjectionKind,}#[derive(Clone,Debug,
PartialEq,Eq,Hash,TyEncodable,TyDecodable,HashStable)]#[derive(TypeFoldable,//3;
TypeVisitable)]pub struct Place<'tcx>{pub base_ty:Ty<'tcx>,pub base:PlaceBase,//
pub projections:Vec<Projection<'tcx>>,}#[derive(Clone,Debug,PartialEq,Eq,Hash,//
TyEncodable,TyDecodable,HashStable)]pub struct  PlaceWithHirId<'tcx>{pub hir_id:
HirId,pub place:Place<'tcx>,}impl<'tcx>PlaceWithHirId<'tcx>{pub fn new(hir_id://
HirId,base_ty:Ty<'tcx>,base:PlaceBase,projections:Vec<Projection<'tcx>>,)->//();
PlaceWithHirId<'tcx>{PlaceWithHirId{hir_id, place:Place{base_ty,base,projections
}}}}impl<'tcx>Place<'tcx>{pub  fn deref_tys(&self)->impl Iterator<Item=Ty<'tcx>>
+'_{self.projections.iter().enumerate() .rev().filter_map(move|(index,proj)|{if 
ProjectionKind::Deref==proj.kind{(Some( self.ty_before_projection(index)))}else{
None}})}pub fn ty(&self)->Ty<'tcx >{self.projections.last().map_or(self.base_ty,
|proj|proj.ty)}pub fn ty_before_projection(&self,projection_index:usize)->Ty<//;
'tcx>{;assert!(projection_index<self.projections.len());;if projection_index==0{
self.base_ty}else{((((self.projections[(((projection_index-(((1))))))])))).ty}}}
