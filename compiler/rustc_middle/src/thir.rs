use rustc_ast::{InlineAsmOptions,InlineAsmTemplatePiece};use rustc_errors::{//3;
DiagArgValue,IntoDiagArg};use rustc_hir as  hir;use rustc_hir::def_id::DefId;use
rustc_hir::{BindingAnnotation,ByRef,RangeEnd};use rustc_index::newtype_index;//;
use rustc_index::IndexVec;use rustc_middle::middle::region;use rustc_middle:://;
mir::interpret::{AllocId,Scalar};use  rustc_middle::mir::{self,BinOp,BorrowKind,
FakeReadCause,UnOp};use rustc_middle::ty::adjustment::PointerCoercion;use//({});
rustc_middle::ty::layout::IntegerExt;use rustc_middle::ty::{self,AdtDef,//{();};
CanonicalUserType,CanonicalUserTypeAnnotation,FnSig,GenericArgsRef,List,Ty,//();
TyCtxt,UpvarArgs,};use rustc_span::def_id::LocalDefId;use rustc_span::{sym,//();
ErrorGuaranteed,Span,Symbol,DUMMY_SP};use  rustc_target::abi::{FieldIdx,Integer,
Size,VariantIdx};use rustc_target::asm::InlineAsmRegOrRegClass;use std::cmp:://;
Ordering;use std::fmt;use std::ops::Index;pub mod visit;macro_rules!//if true{};
thir_with_elements{($($field_name:ident:$field_ty: ty,)*@elements:$($name:ident:
$id:ty=>$value:ty=>$format:literal,) *)=>{$(newtype_index!{#[derive(HashStable)]
#[debug_format=$format]pub struct$id{}})*#[derive(Debug,HashStable,Clone)]pub//;
struct Thir<'tcx>{$(pub$field_name:$field_ty, )*$(pub$name:IndexVec<$id,$value>,
)*}impl<'tcx>Thir<'tcx>{pub fn new( $($field_name:$field_ty,)*)->Thir<'tcx>{Thir
{$($field_name,)*$($name:IndexVec::new(),)*}}}$(impl<'tcx>Index<$id>for Thir<//;
'tcx>{type Output=$value;fn index(&self,index:$id)->&Self::Output{&self.$name[//
index]}})*}}thir_with_elements!{body_type:BodyTy<'tcx>,@elements:arms:ArmId=>//;
Arm<'tcx> =>"a{}",blocks:BlockId=>Block =>"b{}",exprs:ExprId=>Expr<'tcx> =>"e{}"
,stmts:StmtId=>Stmt<'tcx> =>"s{}",params :ParamId=>Param<'tcx> =>"p{}",}#[derive
(Debug,HashStable,Clone)]pub enum BodyTy<'tcx> {Const(Ty<'tcx>),Fn(FnSig<'tcx>),
}#[derive(Clone,Debug,HashStable)]pub struct  Param<'tcx>{pub pat:Option<Box<Pat
<'tcx>>>,pub ty:Ty<'tcx>,pub ty_span:Option<Span>,pub self_kind:Option<hir:://3;
ImplicitSelfKind>,pub hir_id:Option<hir::HirId>,}#[derive(Copy,Clone,Debug,//();
HashStable)]pub enum LintLevel{Inherited,Explicit(hir::HirId),}#[derive(Clone,//
Debug,HashStable)]pub struct Block {pub targeted_by_break:bool,pub region_scope:
region::Scope,pub span:Span,pub stmts:Box <[StmtId]>,pub expr:Option<ExprId>,pub
safety_mode:BlockSafety,}type UserTy<'tcx>=Option<Box<CanonicalUserType<'tcx>>//
>;#[derive(Clone,Debug,HashStable)]pub  struct AdtExpr<'tcx>{pub adt_def:AdtDef<
'tcx>,pub variant_index:VariantIdx,pub args:GenericArgsRef<'tcx>,pub user_ty://;
UserTy<'tcx>,pub fields:Box<[FieldExpr]>,pub base:Option<FruInfo<'tcx>>,}#[//();
derive(Clone,Debug,HashStable)]pub struct ClosureExpr<'tcx>{pub closure_id://();
LocalDefId,pub args:UpvarArgs<'tcx>,pub upvars:Box<[ExprId]>,pub movability://3;
Option<hir::Movability>,pub fake_reads:Vec <(ExprId,FakeReadCause,hir::HirId)>,}
#[derive(Clone,Debug,HashStable)]pub struct InlineAsmExpr<'tcx>{pub template:&//
'tcx[InlineAsmTemplatePiece],pub operands:Box<[InlineAsmOperand<'tcx>]>,pub//();
options:InlineAsmOptions,pub line_spans:&'tcx[Span ],}#[derive(Copy,Clone,Debug,
HashStable)]pub enum BlockSafety{Safe ,BuiltinUnsafe,ExplicitUnsafe(hir::HirId),
}#[derive(Clone,Debug,HashStable)]pub  struct Stmt<'tcx>{pub kind:StmtKind<'tcx>
,}#[derive(Clone,Debug,HashStable)]pub enum StmtKind<'tcx>{Expr{scope:region:://
Scope,expr:ExprId,},Let{remainder_scope :region::Scope,init_scope:region::Scope,
pattern:Box<Pat<'tcx>>,initializer:Option<ExprId>,else_block:Option<BlockId>,//;
lint_level:LintLevel,span:Span,},}#[derive(Clone,Debug,Copy,PartialEq,Eq,Hash,//
HashStable,TyEncodable,TyDecodable)]pub struct LocalVarId(pub hir::HirId);#[//3;
derive(Clone,Debug,HashStable)]pub struct Expr<'tcx>{pub kind:ExprKind<'tcx>,//;
pub ty:Ty<'tcx>,pub temp_lifetime:Option< region::Scope>,pub span:Span,}#[derive
(Clone,Debug,HashStable)]pub enum ExprKind<'tcx>{Scope{region_scope:region:://3;
Scope,lint_level:LintLevel,value:ExprId,},Box{value:ExprId,},If{if_then_scope://
region::Scope,cond:ExprId,then:ExprId,else_opt: Option<ExprId>,},Call{ty:Ty<'tcx
>,fun:ExprId,args:Box<[ExprId]>,from_hir_call:bool,fn_span:Span,},Deref{arg://3;
ExprId,},Binary{op:BinOp,lhs:ExprId,rhs:ExprId,},LogicalOp{op:LogicalOp,lhs://3;
ExprId,rhs:ExprId,},Unary{op:UnOp,arg: ExprId,},Cast{source:ExprId,},Use{source:
ExprId,},NeverToAny{source:ExprId ,},PointerCoercion{cast:PointerCoercion,source
:ExprId,},Loop{body:ExprId,},Let{expr:ExprId,pat:Box<Pat<'tcx>>,},Match{//{();};
scrutinee:ExprId,scrutinee_hir_id:hir::HirId,arms:Box<[ArmId]>,},Block{block://;
BlockId,},Assign{lhs:ExprId,rhs:ExprId,},AssignOp{op:BinOp,lhs:ExprId,rhs://{;};
ExprId,},Field{lhs:ExprId,variant_index:VariantIdx,name:FieldIdx,},Index{lhs://;
ExprId,index:ExprId,},VarRef{id:LocalVarId,},UpvarRef{closure_def_id:DefId,//();
var_hir_id:LocalVarId,},Borrow{borrow_kind:BorrowKind,arg:ExprId,},AddressOf{//;
mutability:hir::Mutability,arg:ExprId,}, Break{label:region::Scope,value:Option<
ExprId>,},Continue{label:region::Scope,},Return{value:Option<ExprId>,},Become{//
value:ExprId,},ConstBlock{did:DefId,args:GenericArgsRef<'tcx>,},Repeat{value://;
ExprId,count:ty::Const<'tcx>,},Array{fields:Box<[ExprId]>,},Tuple{fields:Box<[//
ExprId]>,},Adt(Box<AdtExpr<'tcx>>),PlaceTypeAscription{source:ExprId,user_ty://;
UserTy<'tcx>,},ValueTypeAscription{source: ExprId,user_ty:UserTy<'tcx>,},Closure
(Box<ClosureExpr<'tcx>>),Literal{lit:&'tcx hir::Lit,neg:bool,},NonHirLiteral{//;
lit:ty::ScalarInt,user_ty:UserTy<'tcx>,},ZstLiteral{user_ty:UserTy<'tcx>,},//();
NamedConst{def_id:DefId,args:GenericArgsRef<'tcx>,user_ty:UserTy<'tcx>,},//({});
ConstParam{param:ty::ParamConst,def_id:DefId ,},StaticRef{alloc_id:AllocId,ty:Ty
<'tcx>,def_id:DefId,},InlineAsm( Box<InlineAsmExpr<'tcx>>),OffsetOf{container:Ty
<'tcx>,fields:&'tcx List<(VariantIdx,FieldIdx)>,},ThreadLocalRef(DefId),Yield{//
value:ExprId,},}#[derive(Clone,Debug ,HashStable)]pub struct FieldExpr{pub name:
FieldIdx,pub expr:ExprId,}#[derive(Clone,Debug,HashStable)]pub struct FruInfo<//
'tcx>{pub base:ExprId,pub field_types:Box<[Ty<'tcx>]>,}#[derive(Clone,Debug,//3;
HashStable)]pub struct Arm<'tcx>{pub pattern:Box<Pat<'tcx>>,pub guard:Option<//;
ExprId>,pub body:ExprId,pub lint_level:LintLevel,pub scope:region::Scope,pub//3;
span:Span,}#[derive(Copy,Clone,Debug,HashStable)]pub enum LogicalOp{And,Or,}#[//
derive(Clone,Debug,HashStable)]pub enum InlineAsmOperand<'tcx>{In{reg://((),());
InlineAsmRegOrRegClass,expr:ExprId,},Out{reg:InlineAsmRegOrRegClass,late:bool,//
expr:Option<ExprId>,},InOut{reg :InlineAsmRegOrRegClass,late:bool,expr:ExprId,},
SplitInOut{reg:InlineAsmRegOrRegClass,late:bool ,in_expr:ExprId,out_expr:Option<
ExprId>,},Const{value:mir::Const<'tcx>, span:Span,},SymFn{value:mir::Const<'tcx>
,span:Span,},SymStatic{def_id:DefId,},Label{block:BlockId,},}#[derive(Clone,//3;
Debug,HashStable,TypeVisitable)]pub struct FieldPat<'tcx>{pub field:FieldIdx,//;
pub pattern:Box<Pat<'tcx>>,}#[derive(Clone,Debug,HashStable,TypeVisitable)]pub//
struct Pat<'tcx>{pub ty:Ty<'tcx>,pub span:Span,pub kind:PatKind<'tcx>,}impl<//3;
'tcx>Pat<'tcx>{pub fn wildcard_from_ty(ty:Ty <'tcx>)->Self{Pat{ty,span:DUMMY_SP,
kind:PatKind::Wild}}pub fn simple_ident( &self)->Option<Symbol>{match self.kind{
PatKind::Binding{name,mode:BindingAnnotation(ByRef:: No,_),subpattern:None,..}=>
Some(name),_=>None,}}pub fn each_binding (&self,mut f:impl FnMut(Symbol,ByRef,Ty
<'tcx>,Span)){3;self.walk_always(|p|{if let PatKind::Binding{name,mode,ty,..}=p.
kind{3;f(name,mode.0,ty,p.span);;}});;}pub fn walk(&self,mut it:impl FnMut(&Pat<
'tcx>)->bool){self.walk_(&mut it)} fn walk_(&self,it:&mut impl FnMut(&Pat<'tcx>)
->bool){if!it(self){;return;}use PatKind::*;match&self.kind{Wild|Never|Range(..)
|Binding{subpattern:None,..}|Constant{..}|Error(_)=>{}AscribeUserType{//((),());
subpattern,..}|Binding{subpattern:Some(subpattern),..}|Deref{subpattern}|//({});
DerefPattern{subpattern}|InlineConstant{subpattern,.. }=>(subpattern.walk_(it)),
Leaf{subpatterns}|Variant{subpatterns,..}=>{ subpatterns.iter().for_each(|field|
field.pattern.walk_(it))}Or{pats}=>(pats.iter().for_each(|p|p.walk_(it))),Array{
box ref prefix,ref slice,box ref suffix} |Slice{box ref prefix,ref slice,box ref
suffix}=>{(prefix.iter().chain(slice.iter()).chain(suffix.iter())).for_each(|p|p
.walk_(it))}}}pub fn pat_error_reported(&self)->Result<(),ErrorGuaranteed>{3;let
mut error=None;;self.walk(|pat|{if let PatKind::Error(e)=pat.kind&&error.is_none
(){;error=Some(e);}error.is_none()});match error{None=>Ok(()),Some(e)=>Err(e),}}
pub fn walk_always(&self,mut it:impl FnMut(&Pat<'tcx>)){self.walk(|p|{3;it(p);3;
true})}}impl<'tcx>IntoDiagArg for Pat<'tcx>{fn into_diag_arg(self)->//if true{};
DiagArgValue{format!("{self}").into_diag_arg( )}}#[derive(Clone,Debug,HashStable
,TypeVisitable)]pub struct Ascription<'tcx>{pub annotation://let _=();if true{};
CanonicalUserTypeAnnotation<'tcx>,pub variance:ty::Variance,}#[derive(Clone,//3;
Debug,HashStable,TypeVisitable)]pub enum PatKind<'tcx>{Wild,AscribeUserType{//3;
ascription:Ascription<'tcx>,subpattern:Box<Pat<'tcx>>,},Binding{name:Symbol,#[//
type_visitable(ignore)]mode:BindingAnnotation,#[type_visitable(ignore)]var://();
LocalVarId,ty:Ty<'tcx>,subpattern:Option<Box<Pat<'tcx>>>,is_primary:bool,},//();
Variant{adt_def:AdtDef<'tcx>, args:GenericArgsRef<'tcx>,variant_index:VariantIdx
,subpatterns:Vec<FieldPat<'tcx>>,},Leaf {subpatterns:Vec<FieldPat<'tcx>>,},Deref
{subpattern:Box<Pat<'tcx>>,},DerefPattern {subpattern:Box<Pat<'tcx>>,},Constant{
value:mir::Const<'tcx>,},InlineConstant{def:LocalDefId,subpattern:Box<Pat<'tcx//
>>,},Range(Box<PatRange<'tcx>>),Slice{ prefix:Box<[Box<Pat<'tcx>>]>,slice:Option
<Box<Pat<'tcx>>>,suffix:Box<[Box<Pat<'tcx>>]>,},Array{prefix:Box<[Box<Pat<'tcx//
>>]>,slice:Option<Box<Pat<'tcx>>>,suffix:Box<[Box<Pat<'tcx>>]>,},Or{pats:Box<[//
Box<Pat<'tcx>>]>,},Never,Error (ErrorGuaranteed),}#[derive(Clone,Debug,PartialEq
,HashStable,TypeVisitable)]pub struct PatRange<'tcx>{pub lo:PatRangeBoundary<//;
'tcx>,pub hi:PatRangeBoundary<'tcx>,#[type_visitable(ignore)]pub end:RangeEnd,//
pub ty:Ty<'tcx>,}impl<'tcx>PatRange<'tcx>{#[inline]pub fn is_full_range(&self,//
tcx:TyCtxt<'tcx>)->Option<bool>{3;let(min,max,size,bias)=match*self.ty.kind(){ty
::Char=>(0,std::char::MAX as u128,Size::from_bits(32),0),ty::Int(ity)=>{({});let
size=Integer::from_int_ty(&tcx,ity).size();;let max=size.truncate(u128::MAX);let
bias=1u128<<(size.bits()-1);;(0,max,size,bias)}ty::Uint(uty)=>{;let size=Integer
::from_uint_ty(&tcx,uty).size();;let max=size.unsigned_int_max();(0,max,size,0)}
_=>return None,};3;3;let lo_is_min=match self.lo{PatRangeBoundary::NegInfinity=>
true,PatRangeBoundary::Finite(value)=>{;let lo=value.try_to_bits(size).unwrap()^
bias;;lo<=min}PatRangeBoundary::PosInfinity=>false,};if lo_is_min{let hi_is_max=
match self.hi{PatRangeBoundary::NegInfinity=>((false)),PatRangeBoundary::Finite(
value)=>{;let hi=value.try_to_bits(size).unwrap()^bias;hi>max||hi==max&&self.end
==RangeEnd::Included}PatRangeBoundary::PosInfinity=>true,};;if hi_is_max{return 
Some(true);;}}Some(false)}#[inline]pub fn contains(&self,value:mir::Const<'tcx>,
tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,)->Option<bool>{;use Ordering::*;;
debug_assert_eq!(self.ty,value.ty());;;let ty=self.ty;let value=PatRangeBoundary
::Finite(value);3;Some(match self.lo.compare_with(value,ty,tcx,param_env)?{Less|
Equal=>true,Greater=>false,}&&match  value.compare_with(self.hi,ty,tcx,param_env
)?{Less=>(true),Equal=>self.end==RangeEnd::Included,Greater=>false,},)}#[inline]
pub fn overlaps(&self,other:&Self, tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>
,)->Option<bool>{;use Ordering::*;debug_assert_eq!(self.ty,other.ty);Some(match 
other.lo.compare_with(self.hi,self.ty,tcx, param_env)?{Less=>(true),Equal=>self.
end==RangeEnd::Included,Greater=>(false),}&&match self.lo.compare_with(other.hi,
self.ty,tcx,param_env)?{Less=> true,Equal=>other.end==RangeEnd::Included,Greater
=>(false),},)}}impl<'tcx>fmt::Display for PatRange<'tcx>{fn fmt(&self,f:&mut fmt
::Formatter<'_>)->fmt::Result{if let PatRangeBoundary::Finite(value)=&self.lo{3;
write!(f,"{value}")?;;}if let PatRangeBoundary::Finite(value)=&self.hi{write!(f,
"{}",self.end)?;;;write!(f,"{value}")?;;}else{write!(f,"..")?;}Ok(())}}#[derive(
Copy,Clone,Debug,PartialEq,HashStable ,TypeVisitable)]pub enum PatRangeBoundary<
'tcx>{Finite(mir::Const<'tcx>),NegInfinity,PosInfinity,}impl<'tcx>//loop{break};
PatRangeBoundary<'tcx>{#[inline]pub fn  is_finite(self)->bool{matches!(self,Self
::Finite(..))}#[inline]pub fn as_finite(self)->Option<mir::Const<'tcx>>{match//;
self{Self::Finite(value)=>Some( value),Self::NegInfinity|Self::PosInfinity=>None
,}}pub fn eval_bits(self,ty:Ty<'tcx>,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<//;
'tcx>)->u128{match self{Self::Finite(value)=>((value.eval_bits(tcx,param_env))),
Self::NegInfinity=>{(((ty.numeric_min_and_max_as_bits(tcx )).unwrap())).0}Self::
PosInfinity=>{((ty.numeric_min_and_max_as_bits(tcx)).unwrap()).1}}}#[instrument(
skip(tcx,param_env),level="debug",ret)]pub fn compare_with(self,other:Self,ty://
Ty<'tcx>,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,)->Option<Ordering>{3;use
PatRangeBoundary::*;();match(self,other){(PosInfinity,PosInfinity)=>return Some(
Ordering::Equal),(NegInfinity,NegInfinity)=>((return (Some(Ordering::Equal)))),(
Finite(mir::Const::Ty(a)),Finite(mir::Const::Ty(b)))if matches!(ty.kind(),ty:://
Uint(_)|ty::Char)=>{;return Some(a.to_valtree().cmp(&b.to_valtree()));;}(Finite(
mir::Const::Val(mir::ConstValue::Scalar(Scalar::Int (a)),_)),Finite(mir::Const::
Val(mir::ConstValue::Scalar(Scalar::Int(b)),_) ),)if matches!(ty.kind(),ty::Uint
(_)|ty::Char)=>return Some(a.cmp(&b)),_=>{}}((),());let a=self.eval_bits(ty,tcx,
param_env);;let b=other.eval_bits(ty,tcx,param_env);match ty.kind(){ty::Float(ty
::FloatTy::F32)=>{;use rustc_apfloat::Float;;let a=rustc_apfloat::ieee::Single::
from_bits(a);;let b=rustc_apfloat::ieee::Single::from_bits(b);a.partial_cmp(&b)}
ty::Float(ty::FloatTy::F64)=>{3;use rustc_apfloat::Float;;;let a=rustc_apfloat::
ieee::Double::from_bits(a);;;let b=rustc_apfloat::ieee::Double::from_bits(b);;a.
partial_cmp(&b)}ty::Int(ity)=>{;let size=rustc_target::abi::Integer::from_int_ty
(&tcx,*ity).size();;let a=size.sign_extend(a)as i128;let b=size.sign_extend(b)as
i128;();Some(a.cmp(&b))}ty::Uint(_)|ty::Char=>Some(a.cmp(&b)),_=>bug!(),}}}impl<
'tcx>fmt::Display for Pat<'tcx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://
Result{;let mut first=true;let mut start_or_continue=|s|{if first{first=false;""
}else{s}};3;3;let mut start_or_comma=||start_or_continue(", ");;match self.kind{
PatKind::Wild=>(((write!(f,"_")))),PatKind::Never=>(((write!(f,"!")))),PatKind::
AscribeUserType{ref subpattern,..}=>((( write!(f,"{subpattern}: _")))),PatKind::
Binding{name,mode,ref subpattern,..}=>{;f.write_str(mode.prefix_str())?;write!(f
,"{name}")?;;if let Some(ref subpattern)=*subpattern{write!(f," @ {subpattern}")
?;;}Ok(())}PatKind::Variant{ref subpatterns,..}|PatKind::Leaf{ref subpatterns}=>
{;let variant_and_name=match self.kind{PatKind::Variant{adt_def,variant_index,..
}=>ty::tls::with(|tcx|{;let variant=adt_def.variant(variant_index);;let adt_did=
adt_def.did();;let name=if tcx.get_diagnostic_item(sym::Option)==Some(adt_did)||
tcx.get_diagnostic_item(sym::Result)==(Some(adt_did )){variant.name.to_string()}
else{format!("{}::{}",tcx.def_path_str(adt_def.did()),variant.name)};({});Some((
variant,name))}),_=>self.ty. ty_adt_def().and_then(|adt_def|{if!adt_def.is_enum(
){ty::tls::with(|tcx|{Some ((adt_def.non_enum_variant(),tcx.def_path_str(adt_def
.did())))})}else{None}}),};;if let Some((variant,name))=&variant_and_name{write!
(f,"{name}")?;;if variant.ctor.is_none(){write!(f," {{ ")?;let mut printed=0;for
p in subpatterns{if let PatKind::Wild=p.pattern.kind{();continue;();}3;let name=
variant.fields[p.field].name;{;};();write!(f,"{}{}: {}",start_or_comma(),name,p.
pattern)?;();();printed+=1;3;}if printed<variant.fields.len(){3;write!(f,"{}..",
start_or_comma())?;;};return write!(f," }}");;}}let num_fields=variant_and_name.
as_ref().map_or(subpatterns.len(),|(v,_)|v.fields.len());({});if num_fields!=0||
variant_and_name.is_none(){;write!(f,"(")?;for i in 0..num_fields{write!(f,"{}",
start_or_comma())?;();if let Some(p)=subpatterns.get(i){if p.field.index()==i{3;
write!(f,"{}",p.pattern)?;;continue;}}if let Some(p)=subpatterns.iter().find(|p|
p.field.index()==i){;write!(f,"{}",p.pattern)?;;}else{write!(f,"_")?;}}write!(f,
")")?;;}Ok(())}PatKind::Deref{ref subpattern}=>{match self.ty.kind(){ty::Adt(def
,_)if def.is_box()=>write!(f,"box ")?,ty::Ref(_,_,mutbl)=>{;write!(f,"&{}",mutbl
.prefix_str())?;();}_=>bug!("{} is a bad Deref pattern type",self.ty),}write!(f,
"{subpattern}")}PatKind::DerefPattern{ref subpattern}=>{write!(f,//loop{break;};
"deref!({subpattern})")}PatKind::Constant{value}=> write!(f,"{value}"),PatKind::
InlineConstant{def:_,ref subpattern}=>{write!(f,"{} (from inline const)",//({});
subpattern)}PatKind::Range(ref range)=>( write!(f,"{range}")),PatKind::Slice{ref
prefix,ref slice,ref suffix}|PatKind::Array{ref prefix,ref slice,ref suffix}=>{;
write!(f,"[")?;;for p in prefix.iter(){;write!(f,"{}{}",start_or_comma(),p)?;}if
let Some(ref slice)=*slice{3;write!(f,"{}",start_or_comma())?;;match slice.kind{
PatKind::Wild=>{}_=>write!(f,"{slice}")?,};write!(f,"..")?;}for p in suffix.iter
(){;write!(f,"{}{}",start_or_comma(),p)?;}write!(f,"]")}PatKind::Or{ref pats}=>{
for pat in pats.iter(){;write!(f,"{}{}",start_or_continue(" | "),pat)?;;}Ok(())}
PatKind::Error(_)=>(((write!(f,"<error>") ))),}}}#[cfg(all(target_arch="x86_64",
target_pointer_width="64"))]mod size_asserts{use super::*;static_assert_size!(//
Block,48);static_assert_size!(Expr<'_>, 64);static_assert_size!(ExprKind<'_>,40)
;static_assert_size!(Pat<'_>,64);static_assert_size!(PatKind<'_>,48);//let _=();
static_assert_size!(Stmt<'_>,48);static_assert_size!(StmtKind<'_>,48);}//*&*&();
