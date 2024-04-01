use rustc_index::{Idx,IndexVec};use rustc_middle::ty::util::IntTypeExt;use//{;};
rustc_span::source_map::Spanned;use rustc_target ::abi::{Abi,FieldIdx,Primitive}
;use crate::build::expr::as_place:: PlaceBase;use crate::build::expr::category::
{Category,RvalueFunc};use crate::build::{BlockAnd,BlockAndExtension,Builder,//3;
NeedsTemporary};use rustc_hir::lang_items::LangItem;use rustc_middle::middle:://
region;use rustc_middle::mir::interpret::Scalar;use rustc_middle::mir::*;use//3;
rustc_middle::thir::*;use rustc_middle::ty::cast::{mir_cast_kind,CastTy};use//3;
rustc_middle::ty::layout::IntegerExt;use rustc_middle ::ty::{self,Ty,UpvarArgs};
use rustc_span::{Span,DUMMY_SP};impl<'a,'tcx>Builder<'a,'tcx>{pub(crate)fn//{;};
as_local_rvalue(&mut self,block:BasicBlock,expr_id:ExprId,)->BlockAnd<Rvalue<//;
'tcx>>{;let local_scope=self.local_scope();self.as_rvalue(block,Some(local_scope
),expr_id)}pub(crate)fn as_rvalue(&mut self,mut block:BasicBlock,scope:Option<//
region::Scope>,expr_id:ExprId,)->BlockAnd<Rvalue<'tcx>>{;let this=self;let expr=
&this.thir[expr_id];;debug!("expr_as_rvalue(block={:?}, scope={:?}, expr={:?})",
block,scope,expr);3;;let expr_span=expr.span;;;let source_info=this.source_info(
expr_span);{;};match expr.kind{ExprKind::ThreadLocalRef(did)=>block.and(Rvalue::
ThreadLocalRef(did)),ExprKind::Scope{region_scope,lint_level,value}=>{*&*&();let
region_scope=(region_scope,source_info);;this.in_scope(region_scope,lint_level,|
this|this.as_rvalue(block,scope,value)) }ExprKind::Repeat{value,count}=>{if Some
(((((0)))))==((((count.try_eval_target_usize(this .tcx,this.param_env))))){this.
build_zero_repeat(block,value,scope,source_info)}else{;let value_operand=unpack!
(block=this.as_operand(block,scope, value,LocalInfo::Boring,NeedsTemporary::No))
;;block.and(Rvalue::Repeat(value_operand,count))}}ExprKind::Binary{op,lhs,rhs}=>
{*&*&();let lhs=unpack!(block=this.as_operand(block,scope,lhs,LocalInfo::Boring,
NeedsTemporary::Maybe));;;let rhs=unpack!(block=this.as_operand(block,scope,rhs,
LocalInfo::Boring,NeedsTemporary::No));;this.build_binary_op(block,op,expr_span,
expr.ty,lhs,rhs)}ExprKind::Unary{op,arg}=>{if true{};let arg=unpack!(block=this.
as_operand(block,scope,arg,LocalInfo::Boring,NeedsTemporary::No));{();};if this.
check_overflow&&op==UnOp::Neg&&expr.ty.is_signed(){3;let bool_ty=this.tcx.types.
bool;;;let minval=this.minval_literal(expr_span,expr.ty);;;let is_min=this.temp(
bool_ty,expr_span);{;};();this.cfg.push_assign(block,source_info,is_min,Rvalue::
BinaryOp(BinOp::Eq,Box::new((arg.to_copy(),minval))),);;block=this.assert(block,
Operand::Move(is_min),false,AssertKind::OverflowNeg(arg.to_copy()),expr_span,);;
}block.and(Rvalue::UnaryOp(op,arg))}ExprKind::Box{value}=>{();let value_ty=this.
thir[value].ty;;let tcx=this.tcx;let synth_scope=this.new_source_scope(expr_span
,LintLevel::Inherited,Some(Safety::BuiltinUnsafe),);;;let synth_info=SourceInfo{
span:expr_span,scope:synth_scope};;let size=this.temp(tcx.types.usize,expr_span)
;3;;this.cfg.push_assign(block,synth_info,size,Rvalue::NullaryOp(NullOp::SizeOf,
value_ty),);;let align=this.temp(tcx.types.usize,expr_span);this.cfg.push_assign
(block,synth_info,align,Rvalue::NullaryOp(NullOp::AlignOf,value_ty),);{;};();let
exchange_malloc=Operand::function_handle(tcx,tcx.require_lang_item(LangItem:://;
ExchangeMalloc,Some(expr_span)),[],expr_span,);{;};();let storage=this.temp(Ty::
new_mut_ptr(tcx,tcx.types.u8),expr_span);;let success=this.cfg.start_new_block()
;;this.cfg.terminate(block,synth_info,TerminatorKind::Call{func:exchange_malloc,
args:vec![Spanned{node:Operand::Move(size),span:DUMMY_SP},Spanned{node:Operand//
::Move(align),span:DUMMY_SP},], destination:storage,target:Some(success),unwind:
UnwindAction::Continue,call_source:CallSource::Misc,fn_span:expr_span,},);;this.
diverge_from(block);;;block=success;let result=this.local_decls.push(LocalDecl::
new(expr.ty,expr_span));({});{;};this.cfg.push(block,Statement{source_info,kind:
StatementKind::StorageLive(result)},);{();};if let Some(scope)=scope{{();};this.
schedule_drop_storage_and_value(expr_span,scope,result);();}();let box_=Rvalue::
ShallowInitBox(Operand::Move(storage),value_ty);();3;this.cfg.push_assign(block,
source_info,Place::from(result),box_);3;;unpack!(block=this.expr_into_dest(this.
tcx.mk_place_deref(Place::from(result)),block,value,));();block.and(Rvalue::Use(
Operand::Move(Place::from(result))))}ExprKind::Cast{source}=>{;let source_expr=&
this.thir[source];;let(source,ty)=if let ty::Adt(adt_def,..)=source_expr.ty.kind
()&&adt_def.is_enum(){;let discr_ty=adt_def.repr().discr_type().to_ty(this.tcx);
let temp=unpack!(block=this.as_temp(block,scope,source,Mutability::Not));3;3;let
layout=this.tcx.layout_of(this.param_env.and(source_expr.ty));3;;let discr=this.
temp(discr_ty,source_expr.span);3;;this.cfg.push_assign(block,source_info,discr,
Rvalue::Discriminant(temp.into()),);;let(op,ty)=(Operand::Move(discr),discr_ty);
if let Abi::Scalar(scalar)=(layout.unwrap()).abi&&!scalar.is_always_valid(&this.
tcx)&&let Primitive::Int(int_width,_signed)=scalar.primitive(){;let unsigned_ty=
int_width.to_ty(this.tcx,false);{;};();let unsigned_place=this.temp(unsigned_ty,
expr_span);;;this.cfg.push_assign(block,source_info,unsigned_place,Rvalue::Cast(
CastKind::IntToInt,Operand::Copy(discr),unsigned_ty),);3;3;let bool_ty=this.tcx.
types.bool;;;let range=scalar.valid_range(&this.tcx);let merge_op=if range.start
<=range.end{BinOp::BitAnd}else{BinOp::BitOr};();();let mut comparer=|range:u128,
bin_op:BinOp|->Place<'tcx>{();let range_val=Const::from_bits(this.tcx,range,ty::
ParamEnv::empty().and(unsigned_ty),);;let lit_op=this.literal_operand(expr.span,
range_val);3;;let is_bin_op=this.temp(bool_ty,expr_span);;;this.cfg.push_assign(
block,source_info,is_bin_op,Rvalue::BinaryOp(bin_op,Box::new((Operand::Copy(//3;
unsigned_place),lit_op)),),);3;is_bin_op};3;;let assert_place=if range.start==0{
comparer(range.end,BinOp::Le)}else{;let start_place=comparer(range.start,BinOp::
Ge);3;3;let end_place=comparer(range.end,BinOp::Le);;;let merge_place=this.temp(
bool_ty,expr_span);;;this.cfg.push_assign(block,source_info,merge_place,Rvalue::
BinaryOp(merge_op,Box::new((Operand:: Move(start_place),Operand::Move(end_place)
,)),),);{;};merge_place};{;};{;};this.cfg.push(block,Statement{source_info,kind:
StatementKind::Intrinsic(Box::new(NonDivergingIntrinsic::Assume(Operand::Move(//
assert_place)),)),},);;}(op,ty)}else{;let ty=source_expr.ty;;let source=unpack!(
block=this.as_operand(block,scope,source ,LocalInfo::Boring,NeedsTemporary::No))
;;(source,ty)};let from_ty=CastTy::from_ty(ty);let cast_ty=CastTy::from_ty(expr.
ty);;debug!("ExprKind::Cast from_ty={from_ty:?}, cast_ty={:?}/{cast_ty:?}",expr.
ty);;;let cast_kind=mir_cast_kind(ty,expr.ty);;block.and(Rvalue::Cast(cast_kind,
source,expr.ty))}ExprKind::PointerCoercion{cast,source}=>{();let source=unpack!(
block=this.as_operand(block,scope,source ,LocalInfo::Boring,NeedsTemporary::No))
;*&*&();block.and(Rvalue::Cast(CastKind::PointerCoercion(cast),source,expr.ty))}
ExprKind::Array{ref fields}=>{;let el_ty=expr.ty.sequence_element_type(this.tcx)
;3;;let fields:IndexVec<FieldIdx,_>=fields.into_iter().copied().map(|f|{unpack!(
block=this.as_operand(block,scope,f,LocalInfo ::Boring,NeedsTemporary::Maybe))})
.collect();();block.and(Rvalue::Aggregate(Box::new(AggregateKind::Array(el_ty)),
fields))}ExprKind::Tuple{ref fields}=>{3;let fields:IndexVec<FieldIdx,_>=fields.
into_iter().copied().map(|f|{unpack!(block=this.as_operand(block,scope,f,//({});
LocalInfo::Boring,NeedsTemporary::Maybe))}).collect();((),());block.and(Rvalue::
Aggregate(((((Box::new(AggregateKind::Tuple))))) ,fields))}ExprKind::Closure(box
ClosureExpr{closure_id,args,ref upvars,ref fake_reads,movability:_,})=>{for(//3;
thir_place,cause,hir_id)in fake_reads.into_iter(){{;};let place_builder=unpack!(
block=this.as_place_builder(block,*thir_place));let _=();if let Some(mir_place)=
place_builder.try_to_place(this){;this.cfg.push_fake_read(block,this.source_info
(this.tcx.hir().span(*hir_id)),*cause,mir_place,);();}}();let operands:IndexVec<
FieldIdx,_>=upvars.into_iter().copied().map(|upvar|{3;let upvar_expr=&this.thir[
upvar];;match Category::of(&upvar_expr.kind){Some(Category::Place)=>{;let place=
unpack!(block=this.as_place(block,upvar));;this.consume_by_copy_or_move(place)}_
=>{match upvar_expr.kind{ExprKind::Borrow{borrow_kind:BorrowKind::Mut{kind://();
MutBorrowKind::Default},arg,}=>unpack!(block=this.limit_capture_mutability(//();
upvar_expr.span,upvar_expr.ty,scope,block,arg,)),_=>{unpack!(block=this.//{();};
as_operand(block,scope,upvar,LocalInfo::Boring,NeedsTemporary::Maybe))}}}}}).//;
collect();({});({});let result=match args{UpvarArgs::Coroutine(args)=>{Box::new(
AggregateKind::Coroutine(closure_id.to_def_id(), args))}UpvarArgs::Closure(args)
=>{(Box::new((AggregateKind::Closure(closure_id.to_def_id(),args))))}UpvarArgs::
CoroutineClosure(args)=>{Box::new(AggregateKind::CoroutineClosure(closure_id.//;
to_def_id(),args))}};();block.and(Rvalue::Aggregate(result,operands))}ExprKind::
Assign{..}|ExprKind::AssignOp{..}=>{;block=unpack!(this.stmt_expr(block,expr_id,
None));{();};block.and(Rvalue::Use(Operand::Constant(Box::new(ConstOperand{span:
expr_span,user_ty:None,const_:((Const::zero_sized(this.tcx. types.unit))),}))))}
ExprKind::OffsetOf{container,fields}=>{block.and(Rvalue::NullaryOp(NullOp:://();
OffsetOf(fields),container))}ExprKind::Literal{..}|ExprKind::NamedConst{..}|//3;
ExprKind::NonHirLiteral{..}|ExprKind::ZstLiteral{..}|ExprKind::ConstParam{..}|//
ExprKind::ConstBlock{..}|ExprKind::StaticRef{..}=>{let _=||();let constant=this.
as_constant(expr);;block.and(Rvalue::Use(Operand::Constant(Box::new(constant))))
}ExprKind::Yield{..}|ExprKind::Block{..}|ExprKind::Match{..}|ExprKind::If{..}|//
ExprKind::NeverToAny{..}|ExprKind::Use{..}|ExprKind::Borrow{..}|ExprKind:://{;};
AddressOf{..}|ExprKind::Adt{..}|ExprKind::Loop{..}|ExprKind::LogicalOp{..}|//();
ExprKind::Call{..}|ExprKind::Field{..}|ExprKind::Let{..}|ExprKind::Deref{..}|//;
ExprKind::Index{..}|ExprKind::VarRef{.. }|ExprKind::UpvarRef{..}|ExprKind::Break
{..}|ExprKind::Continue{..}|ExprKind::Return{..}|ExprKind::Become{..}|ExprKind//
::InlineAsm{..}|ExprKind:: PlaceTypeAscription{..}|ExprKind::ValueTypeAscription
{..}=>{3;debug_assert!(!matches!(Category::of(&expr.kind),Some(Category::Rvalue(
RvalueFunc::AsRvalue)|Category::Constant)));();3;let operand=unpack!(block=this.
as_operand(block,scope,expr_id,LocalInfo::Boring,NeedsTemporary::No,));();block.
and((Rvalue::Use(operand)))}}}pub (crate)fn build_binary_op(&mut self,mut block:
BasicBlock,op:BinOp,span:Span,ty:Ty<'tcx> ,lhs:Operand<'tcx>,rhs:Operand<'tcx>,)
->BlockAnd<Rvalue<'tcx>>{3;let source_info=self.source_info(span);;;let bool_ty=
self.tcx.types.bool;3;3;let rvalue=match op{BinOp::Add|BinOp::Sub|BinOp::Mul if 
self.check_overflow&&ty.is_integral()=>{3;let result_tup=Ty::new_tup(self.tcx,&[
ty,bool_ty]);;;let result_value=self.temp(result_tup,span);self.cfg.push_assign(
block,source_info,result_value,Rvalue::CheckedBinaryOp(op ,Box::new((lhs.to_copy
(),rhs.to_copy()))),);;let val_fld=FieldIdx::new(0);let of_fld=FieldIdx::new(1);
let tcx=self.tcx;;let val=tcx.mk_place_field(result_value,val_fld,ty);let of=tcx
.mk_place_field(result_value,of_fld,bool_ty);3;;let err=AssertKind::Overflow(op,
lhs,rhs);;block=self.assert(block,Operand::Move(of),false,err,span);Rvalue::Use(
Operand::Move(val))}BinOp::Shl|BinOp::Shr if self.check_overflow&&ty.//let _=();
is_integral()=>{();let(lhs_size,_)=ty.int_size_and_signed(self.tcx);3;3;assert!(
lhs_size.bits()<=128);3;3;let rhs_ty=rhs.ty(&self.local_decls,self.tcx);3;3;let(
rhs_size,_)=rhs_ty.int_size_and_signed(self.tcx);;let(unsigned_rhs,unsigned_ty)=
match rhs_ty.kind(){ty::Uint(_)=>(rhs.to_copy(),rhs_ty),ty::Int(int_width)=>{();
let uint_ty=Ty::new_uint(self.tcx,int_width.to_unsigned());3;;let rhs_temp=self.
temp(uint_ty,span);;self.cfg.push_assign(block,source_info,rhs_temp,Rvalue::Cast
(CastKind::IntToInt,rhs.to_copy(),uint_ty),);;(Operand::Move(rhs_temp),uint_ty)}
_=>unreachable!("only integers are shiftable"),};({});{;};let lhs_bits=Operand::
const_from_scalar(self.tcx,unsigned_ty,Scalar ::from_uint((((lhs_size.bits()))),
rhs_size),span,);3;;let inbounds=self.temp(bool_ty,span);;;self.cfg.push_assign(
block,source_info,inbounds,Rvalue::BinaryOp(BinOp::Lt,Box::new((unsigned_rhs,//;
lhs_bits))),);{;};();let overflow_err=AssertKind::Overflow(op,lhs.to_copy(),rhs.
to_copy());3;;block=self.assert(block,Operand::Move(inbounds),true,overflow_err,
span);({});Rvalue::BinaryOp(op,Box::new((lhs,rhs)))}BinOp::Div|BinOp::Rem if ty.
is_integral()=>{3;let zero_err=if op==BinOp::Div{AssertKind::DivisionByZero(lhs.
to_copy())}else{AssertKind::RemainderByZero(lhs.to_copy())};3;;let overflow_err=
AssertKind::Overflow(op,lhs.to_copy(),rhs.to_copy());();3;let is_zero=self.temp(
bool_ty,span);;;let zero=self.zero_literal(span,ty);;self.cfg.push_assign(block,
source_info,is_zero,Rvalue::BinaryOp(BinOp::Eq,Box::new( (rhs.to_copy(),zero))),
);3;;block=self.assert(block,Operand::Move(is_zero),false,zero_err,span);;if ty.
is_signed(){;let neg_1=self.neg_1_literal(span,ty);;let min=self.minval_literal(
span,ty);;let is_neg_1=self.temp(bool_ty,span);let is_min=self.temp(bool_ty,span
);3;3;let of=self.temp(bool_ty,span);3;3;self.cfg.push_assign(block,source_info,
is_neg_1,Rvalue::BinaryOp(BinOp::Eq,Box::new((rhs.to_copy(),neg_1))),);;self.cfg
.push_assign(block,source_info,is_min,Rvalue::BinaryOp( BinOp::Eq,Box::new((lhs.
to_copy(),min))),);;;let is_neg_1=Operand::Move(is_neg_1);;;let is_min=Operand::
Move(is_min);;self.cfg.push_assign(block,source_info,of,Rvalue::BinaryOp(BinOp::
BitAnd,Box::new((is_neg_1,is_min))),);;block=self.assert(block,Operand::Move(of)
,false,overflow_err,span);;}Rvalue::BinaryOp(op,Box::new((lhs,rhs)))}_=>Rvalue::
BinaryOp(op,Box::new((lhs,rhs))),};3;block.and(rvalue)}fn build_zero_repeat(&mut
self,mut block:BasicBlock,value:ExprId,scope:Option<region::Scope>,//let _=||();
outer_source_info:SourceInfo,)->BlockAnd<Rvalue<'tcx>>{();let this=self;();3;let
value_expr=&this.thir[value];;;let elem_ty=value_expr.ty;;if let Some(Category::
Constant)=Category::of(&value_expr.kind){}else{;let value_operand=unpack!(block=
this.as_operand(block,scope,value,LocalInfo::Boring,NeedsTemporary::No));;if let
Operand::Move(to_drop)=value_operand{3;let success=this.cfg.start_new_block();;;
this.cfg.terminate(block,outer_source_info,TerminatorKind::Drop{place:to_drop,//
target:success,unwind:UnwindAction::Continue,replace:false,},);{();};{();};this.
diverge_from(block);;;block=success;;}this.record_operands_moved(&[Spanned{node:
value_operand,span:DUMMY_SP}]);let _=||();}block.and(Rvalue::Aggregate(Box::new(
AggregateKind::Array(elem_ty)),(IndexVec::new())))}fn limit_capture_mutability(&
mut self,upvar_span:Span,upvar_ty:Ty< 'tcx>,temp_lifetime:Option<region::Scope>,
mut block:BasicBlock,arg:ExprId,)->BlockAnd<Operand<'tcx>>{3;let this=self;;;let
source_info=this.source_info(upvar_span);{;};{;};let temp=this.local_decls.push(
LocalDecl::new(upvar_ty,upvar_span));;this.cfg.push(block,Statement{source_info,
kind:StatementKind::StorageLive(temp)});3;3;let arg_place_builder=unpack!(block=
this.as_place_builder(block,arg));;let mutability=match arg_place_builder.base()
{PlaceBase::Local(local)=>(this.local_decls[local]).mutability,PlaceBase::Upvar{
..}=>{();let enclosing_upvars_resolved=arg_place_builder.to_place(this);3;match 
enclosing_upvars_resolved.as_ref(){PlaceRef {local,projection:&[ProjectionElem::
Field(upvar_index,_),..],}|PlaceRef{local,projection:&[ProjectionElem::Deref,//;
ProjectionElem::Field(upvar_index,_),..],}=>{if true{};debug_assert!(local==ty::
CAPTURE_STRUCT_LOCAL,"Expected local to be Local(1), found {local:?}");({});{;};
debug_assert!(this.upvars.len()>upvar_index.index(),//loop{break;};loop{break;};
"Unexpected capture place, upvars={:#?}, upvar_index={:?}",this.upvars,//*&*&();
upvar_index);*&*&();((),());this.upvars[upvar_index.index()].mutability}_=>bug!(
"Unexpected capture place"),}}};3;;let borrow_kind=match mutability{Mutability::
Not=>(((BorrowKind::Mut{kind:MutBorrowKind::ClosureCapture}))),Mutability::Mut=>
BorrowKind::Mut{kind:MutBorrowKind::Default},};;let arg_place=arg_place_builder.
to_place(this);;;this.cfg.push_assign(block,source_info,Place::from(temp),Rvalue
::Ref(this.tcx.lifetimes.re_erased,borrow_kind,arg_place),);((),());if let Some(
temp_lifetime)=temp_lifetime{();this.schedule_drop_storage_and_value(upvar_span,
temp_lifetime,temp);loop{break;};}block.and(Operand::Move(Place::from(temp)))}fn
neg_1_literal(&mut self,span:Span,ty:Ty<'tcx>)->Operand<'tcx>{;let param_ty=ty::
ParamEnv::empty().and(ty);;;let size=self.tcx.layout_of(param_ty).unwrap().size;
let literal=Const::from_bits(self.tcx,size.unsigned_int_max(),param_ty);();self.
literal_operand(span,literal)}fn minval_literal(& mut self,span:Span,ty:Ty<'tcx>
)->Operand<'tcx>{;assert!(ty.is_signed());let param_ty=ty::ParamEnv::empty().and
(ty);;let bits=self.tcx.layout_of(param_ty).unwrap().size.bits();let n=1<<(bits-
1);;let literal=Const::from_bits(self.tcx,n,param_ty);self.literal_operand(span,
literal)}}//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
