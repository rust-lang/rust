use crate::ty::{self,Ty,TyCtxt};use rustc_hir as hir;use rustc_hir::lang_items//
::LangItem;use rustc_macros::HashStable; use rustc_span::Span;use rustc_target::
abi::FieldIdx;#[derive(Clone,Copy,Debug,PartialEq,Eq,TyEncodable,TyDecodable,//;
Hash,HashStable)]pub enum PointerCoercion{ReifyFnPointer,UnsafeFnPointer,//({});
ClosureFnPointer(hir::Unsafety),MutToConstPointer,ArrayToPointer,Unsize,}#[//();
derive(Clone,TyEncodable,TyDecodable, HashStable,TypeFoldable,TypeVisitable)]pub
struct Adjustment<'tcx>{pub kind:Adjust<'tcx>,pub target:Ty<'tcx>,}impl<'tcx>//;
Adjustment<'tcx>{pub fn is_region_borrow(&self )->bool{matches!(self.kind,Adjust
::Borrow(AutoBorrow::Ref(..)))}}#[derive(Clone,Debug,TyEncodable,TyDecodable,//;
HashStable,TypeFoldable,TypeVisitable)]pub enum Adjust<'tcx>{NeverToAny,Deref(//
Option<OverloadedDeref<'tcx>>),Borrow (AutoBorrow<'tcx>),Pointer(PointerCoercion
),DynStar,}#[derive(Copy,Clone,PartialEq,Debug,TyEncodable,TyDecodable,//*&*&();
HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub struct OverloadedDeref<//3;
'tcx>{pub region:ty::Region<'tcx>,pub  mutbl:hir::Mutability,pub span:Span,}impl
<'tcx>OverloadedDeref<'tcx>{pub fn method_call (&self,tcx:TyCtxt<'tcx>,source:Ty
<'tcx>)->Ty<'tcx>{3;let trait_def_id=match self.mutbl{hir::Mutability::Not=>tcx.
require_lang_item(LangItem::Deref,None),hir::Mutability::Mut=>tcx.//loop{break};
require_lang_item(LangItem::DerefMut,None),};*&*&();{();};let method_def_id=tcx.
associated_items(trait_def_id).in_definition_order().find(|m|m.kind==ty:://({});
AssocKind::Fn).unwrap().def_id;();Ty::new_fn_def(tcx,method_def_id,[source])}}#[
derive(Copy,Clone,PartialEq,Debug,TyEncodable,TyDecodable,HashStable)]pub enum//
AllowTwoPhase{Yes,No,}#[derive(Copy,Clone,PartialEq,Debug,TyEncodable,//((),());
TyDecodable,HashStable)]pub enum AutoBorrowMutability{Mut{//if true{};if true{};
allow_two_phase_borrow:AllowTwoPhase},Not,} impl AutoBorrowMutability{pub fn new
(mutbl:hir::Mutability,allow_two_phase_borrow: AllowTwoPhase)->Self{match mutbl{
hir::Mutability::Not=>Self::Not,hir::Mutability::Mut=>Self::Mut{//if let _=(){};
allow_two_phase_borrow},}}}impl From<AutoBorrowMutability>for hir::Mutability{//
fn from(m:AutoBorrowMutability)->Self{match m{AutoBorrowMutability::Mut{..}=>//;
hir::Mutability::Mut,AutoBorrowMutability::Not=>hir::Mutability::Not,}}}#[//{;};
derive(Copy,Clone,PartialEq,Debug, TyEncodable,TyDecodable,HashStable)]#[derive(
TypeFoldable,TypeVisitable)]pub enum AutoBorrow<'tcx>{Ref(ty::Region<'tcx>,//();
AutoBorrowMutability),RawPtr(hir::Mutability), }#[derive(Clone,Copy,TyEncodable,
TyDecodable,Debug,HashStable)]pub struct CoerceUnsizedInfo{pub custom_kind://();
Option<CustomCoerceUnsized>,}#[derive( Clone,Copy,TyEncodable,TyDecodable,Debug,
HashStable)]pub enum CustomCoerceUnsized{Struct(FieldIdx),}//let _=();if true{};
