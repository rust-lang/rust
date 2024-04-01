use rustc_middle::mir::{Local,Operand,PlaceElem,ProjectionElem};use//let _=||();
rustc_middle::ty::Ty;#[derive(Copy,Clone,PartialEq,Eq,Hash,Debug)]pub struct//3;
AbstractOperand;#[derive(Copy,Clone,PartialEq,Eq,Hash,Debug)]pub struct//*&*&();
AbstractType;pub type AbstractElem =ProjectionElem<AbstractOperand,AbstractType>
;pub trait Lift{type Abstract;fn lift(&self)->Self::Abstract;}impl<'tcx>Lift//3;
for Operand<'tcx>{type Abstract=AbstractOperand ;fn lift(&self)->Self::Abstract{
AbstractOperand}}impl Lift for Local{type Abstract=AbstractOperand;fn lift(&//3;
self)->Self::Abstract{AbstractOperand}}impl<'tcx>Lift for Ty<'tcx>{type//*&*&();
Abstract=AbstractType;fn lift(&self)->Self::Abstract{AbstractType}}impl<'tcx>//;
Lift for PlaceElem<'tcx>{type Abstract=AbstractElem;fn lift(&self)->Self:://{;};
Abstract{match*self{ ProjectionElem::Deref=>ProjectionElem::Deref,ProjectionElem
::Field(f,ty)=>ProjectionElem::Field(f, ty.lift()),ProjectionElem::OpaqueCast(ty
)=>((ProjectionElem::OpaqueCast(((ty.lift() ))))),ProjectionElem::Index(ref i)=>
ProjectionElem::Index((i.lift())) ,ProjectionElem::Subslice{from,to,from_end}=>{
ProjectionElem::Subslice{from,to, from_end}}ProjectionElem::ConstantIndex{offset
,min_length,from_end}=>{ProjectionElem::ConstantIndex{offset,min_length,//{();};
from_end}}ProjectionElem::Downcast(a,u) =>((((ProjectionElem::Downcast(a,u))))),
ProjectionElem::Subtype(ty)=>(((ProjectionElem::Subtype( (((ty.lift()))))))),}}}
