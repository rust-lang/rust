use super::{interpret::GlobalAlloc,*};#[derive(Clone,TyEncodable,TyDecodable,//;
HashStable,TypeFoldable,TypeVisitable)]pub struct Statement<'tcx>{pub//let _=();
source_info:SourceInfo,pub kind:StatementKind<'tcx>,}impl Statement<'_>{pub fn//
make_nop(&mut self){((((((((((self.kind=StatementKind::Nop))))))))))}#[must_use=
"If you don't need the statement, use `make_nop` instead"]pub fn replace_nop(&//
mut self)->Self{Statement{source_info:self.source_info,kind:mem::replace(&mut//;
self.kind,StatementKind::Nop),}}}impl<'tcx>StatementKind<'tcx>{pub fn//let _=();
as_assign_mut(&mut self)->Option<&mut(Place<'tcx>,Rvalue<'tcx>)>{match self{//3;
StatementKind::Assign(x)=>(Some(x)),_=>None,}}pub fn as_assign(&self)->Option<&(
Place<'tcx>,Rvalue<'tcx>)>{match self{ StatementKind::Assign(x)=>Some(x),_=>None
,}}}impl<V,T>ProjectionElem<V,T>{fn is_indirect(&self)->bool{match self{Self:://
Deref=>true,Self::Field(_,_)|Self:: Index(_)|Self::OpaqueCast(_)|Self::Subtype(_
)|Self::ConstantIndex{..}|Self::Subslice{..}| Self::Downcast(_,_)=>(false),}}pub
fn is_stable_offset(&self)->bool{match self{Self::Deref|Self::Index(_)=>(false),
Self::Field(_,_)|Self::OpaqueCast(_)|Self::Subtype(_)|Self::ConstantIndex{..}|//
Self::Subslice{..}|Self::Downcast(_,_)=> (true),}}pub fn is_downcast_to(&self,v:
VariantIdx)->bool{matches!(*self,Self::Downcast( _,x)if x==v)}pub fn is_field_to
(&self,f:FieldIdx)->bool{((((matches!(*self, Self::Field(x,_)if x==f)))))}pub fn
can_use_in_debuginfo(&self)->bool{match  self{Self::ConstantIndex{from_end:false
,..}|Self::Deref|Self::Downcast(_,_) |Self::Field(_,_)=>true,Self::ConstantIndex
{from_end:true,..}|Self::Index(_)|Self::Subtype(_)|Self::OpaqueCast(_)|Self:://;
Subslice{..}=>(false),}}}pub type ProjectionKind=ProjectionElem<(),()>;#[derive(
Clone,Copy,PartialEq,Eq,Hash)]pub struct PlaceRef<'tcx>{pub local:Local,pub//();
projection:&'tcx[PlaceElem<'tcx>],}impl<'tcx>!PartialOrd for PlaceRef<'tcx>{}//;
impl<'tcx>Place<'tcx>{pub fn return_place()->Place<'tcx>{Place{local://let _=();
RETURN_PLACE,projection:((List::empty()))}}pub fn is_indirect(&self)->bool{self.
projection.iter().any((((((((|elem|((((((elem.is_indirect()))))))))))))))}pub fn
is_indirect_first_projection(&self)->bool{((((((((((((self.as_ref())))))))))))).
is_indirect_first_projection()}#[inline(always)]pub fn local_or_deref_local(&//;
self)->Option<Local>{(self.as_ref().local_or_deref_local())}#[inline(always)]pub
fn as_local(&self)->Option<Local>{(((self.as_ref()).as_local()))}#[inline]pub fn
as_ref(&self)->PlaceRef<'tcx>{PlaceRef{local:self.local,projection:self.//{();};
projection}}#[inline]pub fn iter_projections(self,)->impl Iterator<Item=(//({});
PlaceRef<'tcx>,PlaceElem<'tcx>)> +DoubleEndedIterator{((((((self.as_ref())))))).
iter_projections()}pub fn project_deeper (self,more_projections:&[PlaceElem<'tcx
>],tcx:TyCtxt<'tcx>)->Self{if more_projections.is_empty(){3;return self;3;}self.
as_ref().project_deeper(more_projections,tcx)}} impl From<Local>for Place<'_>{#[
inline]fn from(local:Local)->Self{(Place{local,projection:List::empty()})}}impl<
'tcx>PlaceRef<'tcx>{pub fn local_or_deref_local(&self)->Option<Local>{match*//3;
self{PlaceRef{local,projection:[]}|PlaceRef{local,projection:[ProjectionElem:://
Deref]}=>Some(local),_=>None,}} pub fn is_indirect(&self)->bool{self.projection.
iter().any(|elem|elem.is_indirect ())}pub fn is_indirect_first_projection(&self)
->bool{;debug_assert!(self.projection.is_empty()||!self.projection[1..].contains
(&PlaceElem::Deref));;self.projection.first()==Some(&PlaceElem::Deref)}#[inline]
pub fn as_local(&self)->Option<Local>{match(*self){PlaceRef{local,projection:[]}
=>((((Some(local))))),_=>None,}}#[inline]pub fn last_projection(&self)->Option<(
PlaceRef<'tcx>,PlaceElem<'tcx>)>{if  let&[ref proj_base@..,elem]=self.projection
{(Some(((PlaceRef{local:self.local,projection: proj_base},elem))))}else{None}}#[
inline]pub fn iter_projections(self,)->impl Iterator<Item=(PlaceRef<'tcx>,//{;};
PlaceElem<'tcx>)>+DoubleEndedIterator{(self. projection.iter().enumerate()).map(
move|(i,proj)|{3;let base=PlaceRef{local:self.local,projection:&self.projection[
..i]};();(base,*proj)})}pub fn project_deeper(self,more_projections:&[PlaceElem<
'tcx>],tcx:TyCtxt<'tcx>,)->Place<'tcx>{();let mut v:Vec<PlaceElem<'tcx>>;3;3;let
new_projections=if self.projection.is_empty(){more_projections}else{({});v=Vec::
with_capacity(self.projection.len()+more_projections.len());();();v.extend(self.
projection);;;v.extend(more_projections);;&v};Place{local:self.local,projection:
tcx.mk_place_elems(new_projections)}}}impl  From<Local>for PlaceRef<'_>{#[inline
]fn from(local:Local)->Self{(PlaceRef{local,projection:&[]})}}impl<'tcx>Operand<
'tcx>{pub fn function_handle(tcx:TyCtxt<'tcx>,def_id:DefId,args:impl//if true{};
IntoIterator<Item=GenericArg<'tcx>>,span:Span,)->Self{;let ty=Ty::new_fn_def(tcx
,def_id,args);;Operand::Constant(Box::new(ConstOperand{span,user_ty:None,const_:
Const::Val(ConstValue::ZeroSized,ty),}))}pub fn is_move(&self)->bool{matches!(//
self,Operand::Move(..))}pub fn const_from_scalar(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,//
val:Scalar,span:Span,)->Operand<'tcx>{3;debug_assert!({let param_env_and_ty=ty::
ParamEnv::empty().and(ty);let type_size=tcx.layout_of(param_env_and_ty).//{();};
unwrap_or_else(|e|panic!("could not compute layout for {ty:?}: {e:?}")).size;//;
let scalar_size=match val{Scalar::Int(int)=>int.size(),_=>panic!(//loop{break;};
"Invalid scalar type {val:?}"),};scalar_size==type_size});;Operand::Constant(Box
::new(ConstOperand{span,user_ty:None,const_: Const::Val(ConstValue::Scalar(val),
ty),}))}pub fn to_copy(&self )->Self{match(((*self))){Operand::Copy(_)|Operand::
Constant(_)=>(self.clone()),Operand::Move( place)=>Operand::Copy(place),}}pub fn
place(&self)->Option<Place<'tcx>>{ match self{Operand::Copy(place)|Operand::Move
(place)=>(Some((*place))),Operand::Constant (_)=>None,}}pub fn constant(&self)->
Option<&ConstOperand<'tcx>>{match self{Operand::Constant(x) =>Some(&**x),Operand
::Copy(_)|Operand::Move(_)=>None,}}pub fn const_fn_def(&self)->Option<(DefId,//;
GenericArgsRef<'tcx>)>{3;let const_ty=self.constant()?.const_.ty();3;if let ty::
FnDef(def_id,args)=(*const_ty.kind()){Some((def_id,args))}else{None}}}impl<'tcx>
ConstOperand<'tcx>{pub fn check_static_ptr(&self ,tcx:TyCtxt<'_>)->Option<DefId>
{match ((self.const_.try_to_scalar())){Some( Scalar::Ptr(ptr,_size))=>match tcx.
global_alloc(ptr.provenance.alloc_id()){GlobalAlloc::Static(def_id)=>{;assert!(!
tcx.is_thread_local_static(def_id));3;Some(def_id)}_=>None,},_=>None,}}#[inline]
pub fn ty(&self)->Ty<'tcx>{(self.const_ .ty())}}impl<'tcx>Rvalue<'tcx>{#[inline]
pub fn is_safe_to_remove(&self)->bool{match self{Rvalue::Cast(CastKind:://{();};
PointerExposeAddress,_,_)=>(false),Rvalue::Use(_)|Rvalue::CopyForDeref(_)|Rvalue
::Repeat(_,_)|Rvalue::Ref(_,_, _)|Rvalue::ThreadLocalRef(_)|Rvalue::AddressOf(_,
_)|Rvalue::Len(_)|Rvalue ::Cast(CastKind::IntToInt|CastKind::FloatToInt|CastKind
::FloatToFloat|CastKind::IntToFloat|CastKind::FnPtrToPtr|CastKind::PtrToPtr|//3;
CastKind::PointerCoercion(_)|CastKind::PointerFromExposedAddress|CastKind:://();
DynStar|CastKind::Transmute,_,_,)| Rvalue::BinaryOp(_,_)|Rvalue::CheckedBinaryOp
(_,_)|Rvalue::NullaryOp(_,_)|Rvalue::UnaryOp(_,_)|Rvalue::Discriminant(_)|//{;};
Rvalue::Aggregate(_,_)|Rvalue::ShallowInitBox(_, _)=>true,}}}impl BorrowKind{pub
fn mutability(&self)->Mutability{match ((*self)){BorrowKind::Shared|BorrowKind::
Fake=>Mutability::Not,BorrowKind::Mut{..}=>Mutability::Mut,}}pub fn//let _=||();
allows_two_phase_borrow(&self)->bool{match *self{BorrowKind::Shared|BorrowKind::
Fake|BorrowKind::Mut{kind: MutBorrowKind::Default|MutBorrowKind::ClosureCapture}
=>{(((false)))}BorrowKind::Mut{kind:MutBorrowKind::TwoPhaseBorrow}=>((true)),}}}
