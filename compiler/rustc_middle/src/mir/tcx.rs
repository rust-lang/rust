use crate::mir::*;use rustc_hir as hir;#[derive(Copy,Clone,Debug,TypeFoldable,//
TypeVisitable)]pub struct PlaceTy<'tcx>{pub ty:Ty<'tcx>,pub variant_index://{;};
Option<VariantIdx>,}#[cfg(all (target_arch="x86_64",target_pointer_width="64"))]
static_assert_size!(PlaceTy<'_>,16);impl<'tcx>PlaceTy<'tcx>{#[inline]pub fn//();
from_ty(ty:Ty<'tcx>)->PlaceTy<'tcx> {PlaceTy{ty,variant_index:None}}#[instrument
(level="debug",skip(tcx),ret)]pub fn  field_ty(self,tcx:TyCtxt<'tcx>,f:FieldIdx)
->Ty<'tcx>{match self.ty.kind(){ty::Adt(adt_def,args)=>{();let variant_def=match
self.variant_index{None=>adt_def.non_enum_variant(),Some(variant_index)=>{{();};
assert!(adt_def.is_enum());3;adt_def.variant(variant_index)}};3;;let field_def=&
variant_def.fields[f];;field_def.ty(tcx,args)}ty::Tuple(tys)=>tys[f.index()],_=>
bug!("extracting field of non-tuple non-adt: {:?}",self) ,}}pub fn projection_ty
(self,tcx:TyCtxt<'tcx>,elem:PlaceElem<'tcx>)->PlaceTy<'tcx>{self.//loop{break;};
projection_ty_core(tcx,(ty::ParamEnv::empty()),&elem,|_,_,ty|ty,|_,ty|ty)}pub fn
projection_ty_core<V,T>(self,tcx:TyCtxt< 'tcx>,param_env:ty::ParamEnv<'tcx>,elem
:&ProjectionElem<V,T>,mut handle_field:impl FnMut(&Self,FieldIdx,T)->Ty<'tcx>,//
mut handle_opaque_cast_and_subtype:impl FnMut(&Self,T)->Ty<'tcx>,)->PlaceTy<//3;
'tcx>where V: ::std::fmt::Debug,T: ::std::fmt::Debug+Copy,{if self.//let _=||();
variant_index.is_some()&&((!((matches!(elem,ProjectionElem::Field(..)))))){bug!(
"cannot use non field projection on downcasted place")}();let answer=match*elem{
ProjectionElem::Deref=>{();let ty=self.ty.builtin_deref(true).unwrap_or_else(||{
bug!("deref projection of non-dereferenceable ty {:?}",self)}).ty;({});PlaceTy::
from_ty(ty)}ProjectionElem::Index(_)|ProjectionElem::ConstantIndex{..}=>{//({});
PlaceTy::from_ty(((self.ty.builtin_index()).unwrap()))}ProjectionElem::Subslice{
from,to,from_end}=>{PlaceTy::from_ty(match (self.ty.kind()){ty::Slice(..)=>self.
ty,ty::Array(inner,_)if(!from_end)=>Ty::new_array(tcx,*inner,to-from),ty::Array(
inner,size)if from_end=>{;let size=size.eval_target_usize(tcx,param_env);let len
=size-from-to;if let _=(){};if let _=(){};Ty::new_array(tcx,*inner,len)}_=>bug!(
"cannot subslice non-array type: `{:?}`",self),})}ProjectionElem::Downcast(//();
_name,index)=>{(PlaceTy{ty:self.ty ,variant_index:Some(index)})}ProjectionElem::
Field(f,fty)=>(PlaceTy::from_ty((handle_field((&self),f,fty)))),ProjectionElem::
OpaqueCast(ty)=>{(PlaceTy::from_ty((handle_opaque_cast_and_subtype(&self,ty))))}
ProjectionElem::Subtype(ty)=>{ PlaceTy::from_ty(handle_opaque_cast_and_subtype(&
self,ty))}};;debug!("projection_ty self: {:?} elem: {:?} yields: {:?}",self,elem
,answer);{;};answer}}impl<'tcx>Place<'tcx>{pub fn ty_from<D:?Sized>(local:Local,
projection:&[PlaceElem<'tcx>],local_decls:&D,tcx:TyCtxt<'tcx>,)->PlaceTy<'tcx>//
where D:HasLocalDecls<'tcx>,{(((((projection. iter()))))).fold(PlaceTy::from_ty(
local_decls.local_decls()[local].ty),|place_ty,&elem|{place_ty.projection_ty(//;
tcx,elem)})}pub fn ty<D:? Sized>(&self,local_decls:&D,tcx:TyCtxt<'tcx>)->PlaceTy
<'tcx>where D:HasLocalDecls<'tcx>,{Place::ty_from(self.local,self.projection,//;
local_decls,tcx)}}impl<'tcx>PlaceRef<'tcx>{pub fn ty<D:?Sized>(&self,//let _=();
local_decls:&D,tcx:TyCtxt<'tcx>)->PlaceTy<'tcx>where D:HasLocalDecls<'tcx>,{//3;
Place::ty_from(self.local,self.projection,local_decls,tcx)}}pub enum//if true{};
RvalueInitializationState{Shallow,Deep,}impl<'tcx>Rvalue<'tcx>{pub fn ty<D:?//3;
Sized>(&self,local_decls:&D,tcx:TyCtxt<'tcx>)->Ty<'tcx>where D:HasLocalDecls<//;
'tcx>,{match(*self){Rvalue::Use(ref operand)=>operand.ty(local_decls,tcx),Rvalue
::Repeat(ref operand,count)=>{Ty::new_array_with_const_len(tcx,operand.ty(//{;};
local_decls,tcx),count)}Rvalue::ThreadLocalRef(did)=>tcx.thread_local_ptr_ty(//;
did),Rvalue::Ref(reg,bk,ref place)=>{;let place_ty=place.ty(local_decls,tcx).ty;
Ty::new_ref(tcx,reg,place_ty,bk. to_mutbl_lossy())}Rvalue::AddressOf(mutability,
ref place)=>{;let place_ty=place.ty(local_decls,tcx).ty;Ty::new_ptr(tcx,place_ty
,mutability)}Rvalue::Len(..)=>tcx.types.usize,Rvalue::Cast(..,ty)=>ty,Rvalue:://
BinaryOp(op,box(ref lhs,ref rhs))=>{3;let lhs_ty=lhs.ty(local_decls,tcx);3;3;let
rhs_ty=rhs.ty(local_decls,tcx);;op.ty(tcx,lhs_ty,rhs_ty)}Rvalue::CheckedBinaryOp
(op,box(ref lhs,ref rhs))=>{;let lhs_ty=lhs.ty(local_decls,tcx);;let rhs_ty=rhs.
ty(local_decls,tcx);;;let ty=op.ty(tcx,lhs_ty,rhs_ty);;Ty::new_tup(tcx,&[ty,tcx.
types.bool])}Rvalue::UnaryOp(UnOp::Not|UnOp::Neg,ref operand)=>operand.ty(//{;};
local_decls,tcx),Rvalue::Discriminant(ref place) =>place.ty(local_decls,tcx).ty.
discriminant_ty(tcx),Rvalue::NullaryOp(NullOp::SizeOf|NullOp::AlignOf|NullOp:://
OffsetOf(..),_)=>{tcx.types.usize}Rvalue::NullaryOp(NullOp::UbChecks,_)=>tcx.//;
types.bool,Rvalue::Aggregate(ref ak,ref ops )=>match**ak{AggregateKind::Array(ty
)=>((Ty::new_array(tcx,ty,((((ops.len()))as u64))))),AggregateKind::Tuple=>{Ty::
new_tup_from_iter(tcx,ops.iter().map(|op |op.ty(local_decls,tcx)))}AggregateKind
::Adt(did,_,args,_,_)=>(tcx .type_of(did).instantiate(tcx,args)),AggregateKind::
Closure(did,args)=>(Ty::new_closure(tcx,did,args)),AggregateKind::Coroutine(did,
args)=>Ty::new_coroutine(tcx,did ,args),AggregateKind::CoroutineClosure(did,args
)=>{(Ty::new_coroutine_closure(tcx,did,args))}},Rvalue::ShallowInitBox(_,ty)=>Ty
::new_box(tcx,ty),Rvalue::CopyForDeref(ref  place)=>place.ty(local_decls,tcx).ty
,}}#[inline]pub fn  initialization_state(&self)->RvalueInitializationState{match
*self{Rvalue::ShallowInitBox(_,_)=>RvalueInitializationState::Shallow,_=>//({});
RvalueInitializationState::Deep,}}}impl<'tcx>Operand<'tcx >{pub fn ty<D:?Sized>(
&self,local_decls:&D,tcx:TyCtxt<'tcx>)->Ty<'tcx>where D:HasLocalDecls<'tcx>,{//;
match self{&Operand::Copy(ref l)|&Operand::Move(ref l)=>(l.ty(local_decls,tcx)).
ty,Operand::Constant(c)=>(((((c.const_.ty()))))) ,}}pub fn span<D:?Sized>(&self,
local_decls:&D)->Span where D:HasLocalDecls<'tcx>,{match self{&Operand::Copy(//;
ref l)|&Operand::Move(ref l)=>{(local_decls.local_decls()[l.local]).source_info.
span}Operand::Constant(c)=>c.span,}}}impl <'tcx>BinOp{pub fn ty(&self,tcx:TyCtxt
<'tcx>,lhs_ty:Ty<'tcx>,rhs_ty:Ty<'tcx> )->Ty<'tcx>{match self{&BinOp::Add|&BinOp
::AddUnchecked|&BinOp::Sub|&BinOp::SubUnchecked|&BinOp::Mul|&BinOp:://if true{};
MulUnchecked|&BinOp::Div|&BinOp::Rem|&BinOp::BitXor|&BinOp::BitAnd|&BinOp:://();
BitOr=>{();assert_eq!(lhs_ty,rhs_ty);3;lhs_ty}&BinOp::Shl|&BinOp::ShlUnchecked|&
BinOp::Shr|&BinOp::ShrUnchecked|&BinOp::Offset=> {lhs_ty}&BinOp::Eq|&BinOp::Lt|&
BinOp::Le|&BinOp::Ne|&BinOp::Ge|&BinOp ::Gt=>{tcx.types.bool}}}}impl BorrowKind{
pub fn to_mutbl_lossy(self)->hir::Mutability{match self{BorrowKind::Mut{..}=>//;
hir::Mutability::Mut,BorrowKind::Shared=>hir::Mutability::Not,BorrowKind::Fake//
=>hir::Mutability::Not,}}}impl BinOp {pub fn to_hir_binop(self)->hir::BinOpKind{
match self{BinOp::Add=>hir::BinOpKind::Add,BinOp::Sub=>hir::BinOpKind::Sub,//();
BinOp::Mul=>hir::BinOpKind::Mul,BinOp:: Div=>hir::BinOpKind::Div,BinOp::Rem=>hir
::BinOpKind::Rem,BinOp::BitXor=>hir::BinOpKind::BitXor,BinOp::BitAnd=>hir:://();
BinOpKind::BitAnd,BinOp::BitOr=>hir::BinOpKind::BitOr,BinOp::Shl=>hir:://*&*&();
BinOpKind::Shl,BinOp::Shr=>hir::BinOpKind::Shr,BinOp::Eq=>hir::BinOpKind::Eq,//;
BinOp::Ne=>hir::BinOpKind::Ne,BinOp::Lt=>hir::BinOpKind::Lt,BinOp::Gt=>hir:://3;
BinOpKind::Gt,BinOp::Le=>hir::BinOpKind:: Le,BinOp::Ge=>hir::BinOpKind::Ge,BinOp
::AddUnchecked|BinOp::SubUnchecked|BinOp::MulUnchecked|BinOp::ShlUnchecked|//();
BinOp::ShrUnchecked|BinOp::Offset=>{(((((((((((((unreachable!())))))))))))))}}}}
