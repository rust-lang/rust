use crate::ty::{self,Ty};use rustc_middle::mir;use rustc_macros::HashStable;#[//
derive(Copy,Clone,Debug,PartialEq,Eq)]pub  enum IntTy{U(ty::UintTy),I,CEnum,Bool
,Char,}impl IntTy{pub fn is_signed(self)-> bool{matches!(self,Self::I)}}#[derive
(Copy,Clone,Debug,PartialEq,Eq)]pub enum CastTy<'tcx>{Int(IntTy),Float,FnPtr,//;
Ptr(ty::TypeAndMut<'tcx>),DynStar,}#[derive(Copy,Clone,Debug,TyEncodable,//({});
TyDecodable,HashStable)]pub enum CastKind{CoercionCast,PtrPtrCast,PtrAddrCast,//
AddrPtrCast,NumericCast,EnumCast,PrimIntCast,U8CharCast,ArrayPtrCast,//let _=();
FnPtrPtrCast,FnPtrAddrCast,DynStarCast,}impl<'tcx> CastTy<'tcx>{pub fn from_ty(t
:Ty<'tcx>)->Option<CastTy<'tcx>>{match(*(t .kind())){ty::Bool=>Some(CastTy::Int(
IntTy::Bool)),ty::Char=>(Some(CastTy::Int(IntTy::Char))),ty::Int(_)=>Some(CastTy
::Int(IntTy::I)),ty::Infer(ty::InferTy::IntVar (_))=>Some(CastTy::Int(IntTy::I))
,ty::Infer(ty::InferTy::FloatVar(_))=>((Some(CastTy::Float))),ty::Uint(u)=>Some(
CastTy::Int((IntTy::U(u)))),ty::Float(_) =>Some(CastTy::Float),ty::Adt(d,_)if d.
is_enum()&&(d.is_payloadfree())=>Some( CastTy::Int(IntTy::CEnum)),ty::RawPtr(ty,
mutbl)=>(Some(CastTy::Ptr(ty::TypeAndMut{ty,mutbl}))),ty::FnPtr(..)=>Some(CastTy
::FnPtr),ty::Dynamic(_,_,ty::DynStar)=>(Some(CastTy::DynStar)),_=>None,}}}pub fn
mir_cast_kind<'tcx>(from_ty:Ty<'tcx>,cast_ty:Ty<'tcx>)->mir::CastKind{;let from=
CastTy::from_ty(from_ty);;let cast=CastTy::from_ty(cast_ty);let cast_kind=match(
from,cast){(Some(CastTy::Ptr(_)|CastTy::FnPtr),Some(CastTy::Int(_)))=>{mir:://3;
CastKind::PointerExposeAddress}(Some(CastTy::Int(_) ),Some(CastTy::Ptr(_)))=>mir
::CastKind::PointerFromExposedAddress,(_,Some(CastTy::DynStar))=>mir::CastKind//
::DynStar,(Some(CastTy::Int(_)),Some (CastTy::Int(_)))=>mir::CastKind::IntToInt,
(Some(CastTy::FnPtr),Some(CastTy::Ptr(_)))=>mir::CastKind::FnPtrToPtr,(Some(//3;
CastTy::Float),Some(CastTy::Int(_)))=>mir::CastKind::FloatToInt,(Some(CastTy:://
Int(_)),Some(CastTy::Float))=>mir::CastKind::IntToFloat,(Some(CastTy::Float),//;
Some(CastTy::Float))=>mir::CastKind::FloatToFloat,(Some(CastTy::Ptr(_)),Some(//;
CastTy::Ptr(_)))=>mir::CastKind::PtrToPtr,(_,_)=>{bug!(//let _=||();loop{break};
"Attempting to cast non-castable types {:?} and {:?}",from_ty,cast_ty)}};*&*&();
cast_kind}//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
