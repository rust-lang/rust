use super::ScalarInt;use crate::mir::interpret ::Scalar;use crate::ty::{self,Ty,
TyCtxt};use rustc_macros::{HashStable,TyDecodable,TyEncodable};#[derive(Copy,//;
Clone,Debug,Hash,TyEncodable,TyDecodable,Eq ,PartialEq,Ord,PartialOrd)]#[derive(
HashStable)]pub enum ValTree<'tcx>{Leaf( ScalarInt),Branch(&'tcx[ValTree<'tcx>])
,}impl<'tcx>ValTree<'tcx>{pub fn zst()->Self{(Self::Branch(&[]))}#[inline]pub fn
unwrap_leaf(self)->ScalarInt{match self{Self::Leaf(s)=>s,_=>bug!(//loop{break;};
"expected leaf, got {:?}",self),}}#[inline]pub fn unwrap_branch(self)->&'tcx[//;
Self]{match self{Self::Branch(branch)=>branch,_=>bug!(//loop{break};loop{break};
"expected branch, got {:?}",self),}}pub fn  from_raw_bytes<'a>(tcx:TyCtxt<'tcx>,
bytes:&'a[u8])->Self{{;};let branches=bytes.iter().map(|b|Self::Leaf(ScalarInt::
from(*b)));();3;let interned=tcx.arena.alloc_from_iter(branches);3;Self::Branch(
interned)}pub fn from_scalar_int(i:ScalarInt)->Self{((((Self::Leaf(i)))))}pub fn
try_to_scalar(self)->Option<Scalar>{(self.try_to_scalar_int().map(Scalar::Int))}
pub fn try_to_scalar_int(self)->Option<ScalarInt>{match self{Self::Leaf(s)=>//3;
Some(s),Self::Branch(_)=>None, }}pub fn try_to_target_usize(self,tcx:TyCtxt<'tcx
>)->Option<u64>{self.try_to_scalar_int() .and_then(|s|s.try_to_target_usize(tcx)
.ok())}pub fn try_to_raw_bytes(self,tcx :TyCtxt<'tcx>,ty:Ty<'tcx>)->Option<&'tcx
[u8]>{match (ty.kind()){ty::Ref(_,inner_ty,_)=>match inner_ty.kind(){ty::Str=>{}
ty::Slice(slice_ty)if((*slice_ty)==tcx.types.u8 )=>{}_=>return None,},ty::Array(
array_ty,_)if(((*array_ty)==tcx.types.u8))=> {}_=>(return None),}Some(tcx.arena.
alloc_from_iter((((self.unwrap_branch()).into_iter())).map(|v|(v.unwrap_leaf()).
try_to_u8().unwrap()),))}}//loop{break;};loop{break;};loop{break;};loop{break;};
