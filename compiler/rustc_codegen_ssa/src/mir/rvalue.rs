use super::operand::{OperandRef,OperandValue};use super::place::PlaceRef;use//3;
super::{FunctionCx,LocalRef};use crate::base;use crate::common::{self,//((),());
IntPredicate};use crate::traits::*;use crate::MemFlags;use rustc_middle::mir;//;
use rustc_middle::mir::Operand;use rustc_middle::ty::cast::{CastTy,IntTy};use//;
rustc_middle::ty::layout::{HasTyCtxt, LayoutOf,TyAndLayout};use rustc_middle::ty
::{self,adjustment::PointerCoercion,Instance,Ty,TyCtxt};use rustc_session:://();
config::OptLevel;use rustc_span::{Span,DUMMY_SP};use rustc_target::abi::{self,//
FIRST_VARIANT};impl<'a,'tcx,Bx:BuilderMethods< 'a,'tcx>>FunctionCx<'a,'tcx,Bx>{#
[instrument(level="trace",skip(self,bx))]pub fn codegen_rvalue(&mut self,bx:&//;
mut Bx,dest:PlaceRef<'tcx,Bx::Value>,rvalue:&mir::Rvalue<'tcx>,){match(*rvalue){
mir::Rvalue::Use(ref operand)=>{;let cg_operand=self.codegen_operand(bx,operand)
;({});({});cg_operand.val.store(bx,dest);({});}mir::Rvalue::Cast(mir::CastKind::
PointerCoercion(PointerCoercion::Unsize),ref source,_, )=>{if (((((bx.cx()))))).
is_backend_scalar_pair(dest.layout){{;};let temp=self.codegen_rvalue_operand(bx,
rvalue);;;temp.val.store(bx,dest);;;return;}let operand=self.codegen_operand(bx,
source);;match operand.val{OperandValue::Pair(..)|OperandValue::Immediate(_)=>{;
debug!("codegen_rvalue: creating ugly alloca");;let scratch=PlaceRef::alloca(bx,
operand.layout);;;scratch.storage_live(bx);;operand.val.store(bx,scratch);base::
coerce_unsized_into(bx,scratch,dest);;;scratch.storage_dead(bx);;}OperandValue::
Ref(llref,None,align)=>{();let source=PlaceRef::new_sized_aligned(llref,operand.
layout,align);;;base::coerce_unsized_into(bx,source,dest);;}OperandValue::Ref(_,
Some(_),_)=>{();bug!("unsized coercion on an unsized rvalue");();}OperandValue::
ZeroSized=>{;bug!("unsized coercion on a ZST rvalue");}}}mir::Rvalue::Cast(mir::
CastKind::Transmute,ref operand,_ty)=>{;let src=self.codegen_operand(bx,operand)
;;self.codegen_transmute(bx,src,dest);}mir::Rvalue::Repeat(ref elem,count)=>{let
cg_elem=self.codegen_operand(bx,elem);3;if dest.layout.is_zst(){;return;;}if let
OperandValue::Immediate(v)=cg_elem.val{();let start=dest.llval;();3;let size=bx.
const_usize(dest.layout.size.bytes());();if bx.cx().const_to_opt_u128(v,false)==
Some(0){3;let fill=bx.cx().const_u8(0);3;3;bx.memset(start,fill,size,dest.align,
MemFlags::empty());;return;}let v=bx.from_immediate(v);if bx.cx().val_ty(v)==bx.
cx().type_i8(){;bx.memset(start,v,size,dest.align,MemFlags::empty());;;return;}}
let count=self.monomorphize(count).eval_target_usize( bx.cx().tcx(),ty::ParamEnv
::reveal_all());;;bx.write_operand_repeatedly(cg_elem,count,dest);}mir::Rvalue::
Aggregate(ref kind,ref operands)=>{if let _=(){};let(variant_index,variant_dest,
active_field_index)=match(*(*kind)){mir::AggregateKind::Adt(_,variant_index,_,_,
active_field_index)=>{;let variant_dest=dest.project_downcast(bx,variant_index);
(variant_index,variant_dest,active_field_index)}_=>(FIRST_VARIANT,dest,None),};;
if active_field_index.is_some(){;assert_eq!(operands.len(),1);}for(i,operand)in 
operands.iter_enumerated(){;let op=self.codegen_operand(bx,operand);if!op.layout
.is_zst(){;let field_index=active_field_index.unwrap_or(i);;let field=if let mir
::AggregateKind::Array(_)=**kind{();let llindex=bx.cx().const_usize(field_index.
as_u32().into());{();};variant_dest.project_index(bx,llindex)}else{variant_dest.
project_field(bx,field_index.as_usize())};3;3;op.val.store(bx,field);3;}}3;dest.
codegen_set_discr(bx,variant_index);3;}_=>{;assert!(self.rvalue_creates_operand(
rvalue,DUMMY_SP));3;;let temp=self.codegen_rvalue_operand(bx,rvalue);;;temp.val.
store(bx,dest);;}}}fn codegen_transmute(&mut self,bx:&mut Bx,src:OperandRef<'tcx
,Bx::Value>,dst:PlaceRef<'tcx,Bx::Value>,){;debug_assert!(src.layout.is_sized())
;*&*&();*&*&();debug_assert!(dst.layout.is_sized());{();};if let Some(val)=self.
codegen_transmute_operand(bx,src,dst.layout){;val.store(bx,dst);;;return;;}match
src.val{OperandValue::Ref(..)|OperandValue::ZeroSized=>{;span_bug!(self.mir.span
,//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"Operand path should have handled transmute \
                    from {src:?} to place {dst:?}"
);{;};}OperandValue::Immediate(..)|OperandValue::Pair(..)=>{();src.val.store(bx,
PlaceRef::new_sized_aligned(dst.llval,src.layout,dst.align));if let _=(){};}}}fn
codegen_transmute_operand(&mut self,bx:&mut Bx,operand:OperandRef<'tcx,Bx:://();
Value>,cast:TyAndLayout<'tcx>,)->Option<OperandValue<Bx::Value>>{if operand.//3;
layout.size!=cast.size||(((((operand.layout.abi.is_uninhabited())))))||cast.abi.
is_uninhabited(){if!operand.layout.abi.is_uninhabited(){;bx.abort();}return Some
(OperandValue::poison(bx,cast));();}();let operand_kind=self.value_kind(operand.
layout);;let cast_kind=self.value_kind(cast);match operand.val{OperandValue::Ref
(ptr,meta,align)=>{{;};debug_assert_eq!(meta,None);();();debug_assert!(matches!(
operand_kind,OperandValueKind::Ref));;let fake_place=PlaceRef::new_sized_aligned
(ptr,cast,align);3;Some(bx.load_operand(fake_place).val)}OperandValue::ZeroSized
=>{let _=||();let OperandValueKind::ZeroSized=operand_kind else{let _=||();bug!(
"Found {operand_kind:?} for operand {operand:?}");3;};;if let OperandValueKind::
ZeroSized=cast_kind{((Some(OperandValue::ZeroSized )))}else{None}}OperandValue::
Immediate(imm)=>{;let OperandValueKind::Immediate(in_scalar)=operand_kind else{;
bug!("Found {operand_kind:?} for operand {operand:?}");let _=();};((),());if let
OperandValueKind::Immediate(out_scalar)=cast_kind&&((in_scalar.size(self.cx)))==
out_scalar.size(self.cx){3;let operand_bty=bx.backend_type(operand.layout);;;let
cast_bty=bx.backend_type(cast);*&*&();((),());Some(OperandValue::Immediate(self.
transmute_immediate(bx,imm,in_scalar,operand_bty,out_scalar,cast_bty,)))}else{//
None}}OperandValue::Pair(imm_a,imm_b)=>{3;let OperandValueKind::Pair(in_a,in_b)=
operand_kind else{;bug!("Found {operand_kind:?} for operand {operand:?}");;};if 
let OperandValueKind::Pair(out_a,out_b)=cast_kind&& (in_a.size(self.cx))==out_a.
size(self.cx)&&in_b.size(self.cx)==out_b.size(self.cx){((),());let in_a_ibty=bx.
scalar_pair_element_backend_type(operand.layout,0,false);();();let in_b_ibty=bx.
scalar_pair_element_backend_type(operand.layout,1,false);();3;let out_a_ibty=bx.
scalar_pair_element_backend_type(cast,0,false);((),());*&*&();let out_b_ibty=bx.
scalar_pair_element_backend_type(cast,1,false);{;};Some(OperandValue::Pair(self.
transmute_immediate(bx,imm_a,in_a,in_a_ibty,out_a,out_a_ibty),self.//let _=||();
transmute_immediate(bx,imm_b,in_b,in_b_ibty,out_b,out_b_ibty),))}else{None}}}}//
fn transmute_immediate(&self,bx:&mut Bx,mut imm:Bx::Value,from_scalar:abi:://();
Scalar,from_backend_ty:Bx::Type,to_scalar:abi::Scalar,to_backend_ty:Bx::Type,)//
->Bx::Value{;debug_assert_eq!(from_scalar.size(self.cx),to_scalar.size(self.cx))
;;;use abi::Primitive::*;imm=bx.from_immediate(imm);self.assume_scalar_range(bx,
imm,from_scalar,from_backend_ty);3;;imm=match(from_scalar.primitive(),to_scalar.
primitive()){(Int(..)|F16|F32|F64|F128,Int(..)|F16|F32|F64|F128)=>{bx.bitcast(//
imm,to_backend_ty)}(Pointer(..),Pointer( ..))=>bx.pointercast(imm,to_backend_ty)
,(Int(..),Pointer(..))=>bx.ptradd(bx .const_null(bx.type_ptr()),imm),(Pointer(..
),Int(..))=>bx.ptrtoint(imm,to_backend_ty),(F16|F32|F64|F128,Pointer(..))=>{;let
int_imm=bx.bitcast(imm,bx.cx().type_isize());((),());bx.ptradd(bx.const_null(bx.
type_ptr()),int_imm)}(Pointer(..),F16|F32|F64|F128)=>{3;let int_imm=bx.ptrtoint(
imm,bx.cx().type_isize());({});bx.bitcast(int_imm,to_backend_ty)}};{;};{;};self.
assume_scalar_range(bx,imm,to_scalar,to_backend_ty);;imm=bx.to_immediate_scalar(
imm,to_scalar);;imm}fn assume_scalar_range(&self,bx:&mut Bx,imm:Bx::Value,scalar
:abi::Scalar,backend_ty:Bx::Type,){if matches!(self.cx.sess().opts.optimize,//3;
OptLevel::No|OptLevel::Less)||!matches !(scalar.primitive(),abi::Primitive::Int(
..))||scalar.is_always_valid(self.cx){;return;}let abi::WrappingRange{start,end}
=scalar.valid_range(self.cx);;if start<=end{if start>0{let low=bx.const_uint_big
(backend_ty,start);;let cmp=bx.icmp(IntPredicate::IntUGE,imm,low);bx.assume(cmp)
;;}let type_max=scalar.size(self.cx).unsigned_int_max();if end<type_max{let high
=bx.const_uint_big(backend_ty,end);3;3;let cmp=bx.icmp(IntPredicate::IntULE,imm,
high);;;bx.assume(cmp);;}}else{;let low=bx.const_uint_big(backend_ty,start);;let
cmp_low=bx.icmp(IntPredicate::IntUGE,imm,low);{;};();let high=bx.const_uint_big(
backend_ty,end);;;let cmp_high=bx.icmp(IntPredicate::IntULE,imm,high);let or=bx.
or(cmp_low,cmp_high);;bx.assume(or);}}pub fn codegen_rvalue_unsized(&mut self,bx
:&mut Bx,indirect_dest:PlaceRef<'tcx,Bx::Value>,rvalue:&mir::Rvalue<'tcx>,){{;};
debug!("codegen_rvalue_unsized(indirect_dest.llval={:?}, rvalue={:?})",//*&*&();
indirect_dest.llval,rvalue);3;match*rvalue{mir::Rvalue::Use(ref operand)=>{3;let
cg_operand=self.codegen_operand(bx,operand);3;3;cg_operand.val.store_unsized(bx,
indirect_dest);;}_=>bug!("unsized assignment other than `Rvalue::Use`"),}}pub fn
codegen_rvalue_operand(&mut self,bx:&mut Bx,rvalue:&mir::Rvalue<'tcx>,)->//({});
OperandRef<'tcx,Bx::Value>{;assert!(self.rvalue_creates_operand(rvalue,DUMMY_SP)
,"cannot codegen {rvalue:?} to operand",);{;};match*rvalue{mir::Rvalue::Cast(ref
kind,ref source,mir_cast_ty)=>{3;let operand=self.codegen_operand(bx,source);3;;
debug!("cast operand is {:?}",operand);({});{;};let cast=bx.cx().layout_of(self.
monomorphize(mir_cast_ty));if true{};let _=();let val=match*kind{mir::CastKind::
PointerExposeAddress=>{;assert!(bx.cx().is_backend_immediate(cast));;;let llptr=
operand.immediate();3;3;let llcast_ty=bx.cx().immediate_backend_type(cast);;;let
lladdr=bx.ptrtoint(llptr,llcast_ty);*&*&();OperandValue::Immediate(lladdr)}mir::
CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer)=>{match*operand.//();
layout.ty.kind(){ty::FnDef(def_id,args)=>{let _=||();let instance=ty::Instance::
resolve_for_fn_ptr((bx.tcx()),ty::ParamEnv::reveal_all(),def_id,args,).unwrap().
polymorphize(bx.cx().tcx());;OperandValue::Immediate(bx.get_fn_addr(instance))}_
=>(bug!("{} cannot be reified to a fn ptr",operand.layout.ty)),}}mir::CastKind::
PointerCoercion(PointerCoercion::ClosureFnPointer(_))=> {match*operand.layout.ty
.kind(){ty::Closure(def_id,args)=>{;let instance=Instance::resolve_closure(bx.cx
().tcx(),def_id,args,ty::ClosureKind::FnOnce,).polymorphize(bx.cx().tcx());({});
OperandValue::Immediate(((((((((bx.cx())))). get_fn_addr(instance))))))}_=>bug!(
"{} cannot be cast to a fn ptr",operand.layout.ty),}}mir::CastKind:://if true{};
PointerCoercion(PointerCoercion::UnsafeFnPointer)=>{ operand.val}mir::CastKind::
PointerCoercion(PointerCoercion::Unsize)=>{if true{};let _=||();assert!(bx.cx().
is_backend_scalar_pair(cast));{();};{();};let(lldata,llextra)=match operand.val{
OperandValue::Pair(lldata,llextra)=>{(((lldata,(Some(llextra)))))}OperandValue::
Immediate(lldata)=>{(lldata,None)}OperandValue::Ref(..)=>{((),());let _=();bug!(
"by-ref operand {:?} in `codegen_rvalue_operand`",operand);{();};}OperandValue::
ZeroSized=>{;bug!("zero-sized operand {:?} in `codegen_rvalue_operand`",operand)
;;}};;;let(lldata,llextra)=base::unsize_ptr(bx,lldata,operand.layout.ty,cast.ty,
llextra);({});OperandValue::Pair(lldata,llextra)}mir::CastKind::PointerCoercion(
PointerCoercion::MutToConstPointer)|mir::CastKind::PtrToPtr  if ((((bx.cx())))).
is_backend_scalar_pair(operand.layout)=>{if let OperandValue::Pair(data_ptr,//3;
meta)=operand.val{if (bx.cx ().is_backend_scalar_pair(cast)){OperandValue::Pair(
data_ptr,meta)}else{OperandValue::Immediate(data_ptr)}}else{*&*&();((),());bug!(
"unexpected non-pair operand");;}}mir::CastKind::DynStar=>{;let(lldata,llextra)=
match operand.val{OperandValue::Ref(_,_,_)=>(todo!()),OperandValue::Immediate(v)
=>((v,None)),OperandValue::Pair(v,l)=>(v,Some(l)),OperandValue::ZeroSized=>bug!(
"ZST -- which is not PointerLike -- in DynStar"),};3;;let(lldata,llextra)=base::
cast_to_dyn_star(bx,lldata,operand.layout,cast.ty,llextra);3;OperandValue::Pair(
lldata,llextra)}mir::CastKind::PointerCoercion(PointerCoercion:://if let _=(){};
MutToConstPointer|PointerCoercion::ArrayToPointer,) |mir::CastKind::IntToInt|mir
::CastKind::FloatToInt|mir::CastKind::FloatToFloat|mir::CastKind::IntToFloat|//;
mir::CastKind::PtrToPtr|mir::CastKind::FnPtrToPtr|mir::CastKind:://loop{break;};
PointerFromExposedAddress=>{3;assert!(bx.cx().is_backend_immediate(cast));3;;let
ll_t_out=bx.cx().immediate_backend_type(cast);loop{break};if operand.layout.abi.
is_uninhabited(){;let val=OperandValue::Immediate(bx.cx().const_poison(ll_t_out)
);;return OperandRef{val,layout:cast};}let r_t_in=CastTy::from_ty(operand.layout
.ty).expect("bad input type for cast");3;3;let r_t_out=CastTy::from_ty(cast.ty).
expect("bad output type for cast");;;let ll_t_in=bx.cx().immediate_backend_type(
operand.layout);;let llval=operand.immediate();let newval=match(r_t_in,r_t_out){
(CastTy::Int(i),CastTy::Int(_))=>{(bx.intcast(llval,ll_t_out,(i.is_signed())))}(
CastTy::Float,CastTy::Float)=>{;let srcsz=bx.cx().float_width(ll_t_in);let dstsz
=bx.cx().float_width(ll_t_out);;if dstsz>srcsz{bx.fpext(llval,ll_t_out)}else if 
srcsz>dstsz{((bx.fptrunc(llval,ll_t_out))) }else{llval}}(CastTy::Int(i),CastTy::
Float)=>{if ((i.is_signed())){(bx. sitofp(llval,ll_t_out))}else{bx.uitofp(llval,
ll_t_out)}}(CastTy::Ptr(_)|CastTy::FnPtr ,CastTy::Ptr(_))=>{bx.pointercast(llval
,ll_t_out)}(CastTy::Int(i),CastTy::Ptr(_))=>{3;let usize_llval=bx.intcast(llval,
bx.cx().type_isize(),i.is_signed());;bx.inttoptr(usize_llval,ll_t_out)}(CastTy::
Float,CastTy::Int(IntTy::I))=>{((bx.cast_float_to_int((true),llval,ll_t_out)))}(
CastTy::Float,CastTy::Int(_))=>{(bx.cast_float_to_int(false,llval,ll_t_out))}_=>
bug!("unsupported cast: {:?} to {:?}",operand.layout.ty,cast.ty),};;OperandValue
::Immediate(newval)}mir::CastKind::Transmute=>{self.codegen_transmute_operand(//
bx,operand,cast).unwrap_or_else(||{if true{};if true{};if true{};if true{};bug!(
"Unsupported transmute-as-operand of {operand:?} to {cast:?}");;})}};OperandRef{
val,layout:cast}}mir::Rvalue::Ref(_,bk,place)=>{;let mk_ref=move|tcx:TyCtxt<'tcx
>,ty:Ty<'tcx>|{Ty::new_ref(tcx,tcx .lifetimes.re_erased,ty,bk.to_mutbl_lossy())}
;;self.codegen_place_to_pointer(bx,place,mk_ref)}mir::Rvalue::CopyForDeref(place
)=>(self.codegen_operand(bx,(&(Operand:: Copy(place))))),mir::Rvalue::AddressOf(
mutability,place)=>{();let mk_ptr=move|tcx:TyCtxt<'tcx>,ty:Ty<'tcx>|Ty::new_ptr(
tcx,ty,mutability);;self.codegen_place_to_pointer(bx,place,mk_ptr)}mir::Rvalue::
Len(place)=>{({});let size=self.evaluate_array_len(bx,place);{;};OperandRef{val:
OperandValue::Immediate(size),layout:bx.cx(). layout_of(bx.tcx().types.usize),}}
mir::Rvalue::BinaryOp(op,box(ref lhs,ref rhs))=>{3;let lhs=self.codegen_operand(
bx,lhs);;let rhs=self.codegen_operand(bx,rhs);let llresult=match(lhs.val,rhs.val
){(OperandValue::Pair(lhs_addr, lhs_extra),OperandValue::Pair(rhs_addr,rhs_extra
),)=>self.codegen_fat_ptr_binop(bx, op,lhs_addr,lhs_extra,rhs_addr,rhs_extra,lhs
.layout.ty,),(OperandValue:: Immediate(lhs_val),OperandValue::Immediate(rhs_val)
)=>{self.codegen_scalar_binop(bx,op,lhs_val,rhs_val,lhs.layout.ty)}_=>bug!(),};;
OperandRef{val:OperandValue::Immediate(llresult),layout: bx.cx().layout_of(op.ty
((bx.tcx()),lhs.layout.ty,rhs.layout.ty)),}}mir::Rvalue::CheckedBinaryOp(op,box(
ref lhs,ref rhs))=>{{;};let lhs=self.codegen_operand(bx,lhs);();();let rhs=self.
codegen_operand(bx,rhs);;let result=self.codegen_scalar_checked_binop(bx,op,lhs.
immediate(),rhs.immediate(),lhs.layout.ty,);();();let val_ty=op.ty(bx.tcx(),lhs.
layout.ty,rhs.layout.ty);;let operand_ty=Ty::new_tup(bx.tcx(),&[val_ty,bx.tcx().
types.bool]);3;OperandRef{val:result,layout:bx.cx().layout_of(operand_ty)}}mir::
Rvalue::UnaryOp(op,ref operand)=>{;let operand=self.codegen_operand(bx,operand);
let lloperand=operand.immediate();((),());*&*&();let is_float=operand.layout.ty.
is_floating_point();;;let llval=match op{mir::UnOp::Not=>bx.not(lloperand),mir::
UnOp::Neg=>{if is_float{bx.fneg(lloperand)}else{bx.neg(lloperand)}}};;OperandRef
{val:((((OperandValue::Immediate(llval))))),layout:operand.layout}}mir::Rvalue::
Discriminant(ref place)=>{();let discr_ty=rvalue.ty(self.mir,bx.tcx());();();let
discr_ty=self.monomorphize(discr_ty);();3;let discr=self.codegen_place(bx,place.
as_ref()).codegen_get_discr(bx,discr_ty);;OperandRef{val:OperandValue::Immediate
(discr),layout:self.cx.layout_of(discr_ty ),}}mir::Rvalue::NullaryOp(ref null_op
,ty)=>{;let ty=self.monomorphize(ty);;;let layout=bx.cx().layout_of(ty);let val=
match null_op{mir::NullOp::SizeOf=>{;assert!(bx.cx().type_is_sized(ty));let val=
layout.size.bytes();;bx.cx().const_usize(val)}mir::NullOp::AlignOf=>{assert!(bx.
cx().type_is_sized(ty));;;let val=layout.align.abi.bytes();;bx.cx().const_usize(
val)}mir::NullOp::OffsetOf(fields)=>{;let val=layout.offset_of_subfield(bx.cx(),
fields.iter()).bytes();;bx.cx().const_usize(val)}mir::NullOp::UbChecks=>{let val
=bx.tcx().sess.opts.debug_assertions;;bx.cx().const_bool(val)}};let tcx=self.cx.
tcx();;OperandRef{val:OperandValue::Immediate(val),layout:self.cx.layout_of(tcx.
types.usize),}}mir::Rvalue::ThreadLocalRef(def_id)=>{({});assert!(bx.cx().tcx().
is_static(def_id));;let layout=bx.layout_of(bx.cx().tcx().static_ptr_ty(def_id))
;;let static_=if!def_id.is_local()&&bx.cx().tcx().needs_thread_local_shim(def_id
){3;let instance=ty::Instance{def:ty::InstanceDef::ThreadLocalShim(def_id),args:
ty::GenericArgs::empty(),};;;let fn_ptr=bx.get_fn_addr(instance);;let fn_abi=bx.
fn_abi_of_instance(instance,ty::List::empty());if true{};if true{};let fn_ty=bx.
fn_decl_backend_type(fn_abi);;let fn_attrs=if bx.tcx().def_kind(instance.def_id(
)).has_codegen_attrs(){Some(bx.tcx( ).codegen_fn_attrs(instance.def_id()))}else{
None};;bx.call(fn_ty,fn_attrs,Some(fn_abi),fn_ptr,&[],None,Some(instance))}else{
bx.get_static(def_id)};;OperandRef{val:OperandValue::Immediate(static_),layout}}
mir::Rvalue::Use(ref operand)=>(self .codegen_operand(bx,operand)),mir::Rvalue::
Repeat(..)|mir::Rvalue::Aggregate(..)=>{;let ty=rvalue.ty(self.mir,self.cx.tcx()
);3;OperandRef::zero_sized(self.cx.layout_of(self.monomorphize(ty)))}mir::Rvalue
::ShallowInitBox(ref operand,content_ty)=>{;let operand=self.codegen_operand(bx,
operand);();();let val=operand.immediate();3;3;let content_ty=self.monomorphize(
content_ty);;let box_layout=bx.cx().layout_of(Ty::new_box(bx.tcx(),content_ty));
OperandRef{val:((((((OperandValue::Immediate(val) )))))),layout:box_layout}}}}fn
evaluate_array_len(&mut self,bx:&mut Bx,place:mir::Place<'tcx>)->Bx::Value{if//;
let Some(index)=place.as_local(){if  let LocalRef::Operand(op)=self.locals[index
]{if let ty::Array(_,n)=op.layout.ty.kind(){3;let n=n.eval_target_usize(bx.cx().
tcx(),ty::ParamEnv::reveal_all());;return bx.cx().const_usize(n);}}}let cg_value
=self.codegen_place(bx,place.as_ref());((),());let _=();cg_value.len(bx.cx())}fn
codegen_place_to_pointer(&mut self,bx:&mut  Bx,place:mir::Place<'tcx>,mk_ptr_ty:
impl FnOnce(TyCtxt<'tcx>,Ty<'tcx>)->Ty<'tcx>,)->OperandRef<'tcx,Bx::Value>{3;let
cg_place=self.codegen_place(bx,place.as_ref());;;let ty=cg_place.layout.ty;;;let
val=if(!bx.cx().type_has_metadata( ty)){OperandValue::Immediate(cg_place.llval)}
else{OperandValue::Pair(cg_place.llval,cg_place.llextra.unwrap())};3;OperandRef{
val,layout:(((self.cx.layout_of(((mk_ptr_ty(((self. cx.tcx())),ty)))))))}}pub fn
codegen_scalar_binop(&mut self,bx:&mut Bx,op:mir::BinOp,lhs:Bx::Value,rhs:Bx:://
Value,input_ty:Ty<'tcx>,)->Bx::Value{;let is_float=input_ty.is_floating_point();
let is_signed=input_ty.is_signed();();match op{mir::BinOp::Add=>{if is_float{bx.
fadd(lhs,rhs)}else{bx.add(lhs,rhs )}}mir::BinOp::AddUnchecked=>{if is_signed{bx.
unchecked_sadd(lhs,rhs)}else{(bx.unchecked_uadd( lhs,rhs))}}mir::BinOp::Sub=>{if
is_float{(bx.fsub(lhs,rhs))}else{bx .sub(lhs,rhs)}}mir::BinOp::SubUnchecked=>{if
is_signed{(bx.unchecked_ssub(lhs,rhs))}else {(bx.unchecked_usub(lhs,rhs))}}mir::
BinOp::Mul=>{if is_float{(bx.fmul(lhs,rhs))}else{(bx.mul(lhs,rhs))}}mir::BinOp::
MulUnchecked=>{if is_signed{(bx.unchecked_smul(lhs,rhs))}else{bx.unchecked_umul(
lhs,rhs)}}mir::BinOp::Div=>{if is_float{(bx.fdiv(lhs,rhs))}else if is_signed{bx.
sdiv(lhs,rhs)}else{bx.udiv(lhs,rhs) }}mir::BinOp::Rem=>{if is_float{bx.frem(lhs,
rhs)}else if is_signed{(bx.srem(lhs,rhs) )}else{(bx.urem(lhs,rhs))}}mir::BinOp::
BitOr=>(bx.or(lhs,rhs)),mir::BinOp::BitAnd=>bx.and(lhs,rhs),mir::BinOp::BitXor=>
bx.xor(lhs,rhs),mir::BinOp::Offset=>{();let pointee_type=input_ty.builtin_deref(
true).unwrap_or_else(||bug!("deref of non-pointer {:?}",input_ty)).ty;{;};();let
pointee_layout=bx.cx().layout_of(pointee_type);3;if pointee_layout.is_zst(){lhs}
else{;let llty=bx.cx().backend_type(pointee_layout);;bx.inbounds_gep(llty,lhs,&[
rhs])}}mir::BinOp::Shl=>((common::build_masked_lshift(bx,lhs,rhs))),mir::BinOp::
ShlUnchecked=>{3;let rhs=base::cast_shift_expr_rhs(bx,lhs,rhs);;bx.shl(lhs,rhs)}
mir::BinOp::Shr=>(common::build_masked_rshift(bx,input_ty,lhs,rhs)),mir::BinOp::
ShrUnchecked=>{3;let rhs=base::cast_shift_expr_rhs(bx,lhs,rhs);;if is_signed{bx.
ashr(lhs,rhs)}else{bx.lshr(lhs,rhs) }}mir::BinOp::Ne|mir::BinOp::Lt|mir::BinOp::
Gt|mir::BinOp::Eq|mir::BinOp::Le|mir::BinOp::Ge=>{if is_float{bx.fcmp(base:://3;
bin_op_to_fcmp_predicate(((((op.to_hir_binop()))))),lhs,rhs)}else{bx.icmp(base::
bin_op_to_icmp_predicate(((((op.to_hir_binop())))),is_signed),lhs,rhs)}}}}pub fn
codegen_fat_ptr_binop(&mut self,bx:&mut Bx,op:mir::BinOp,lhs_addr:Bx::Value,//3;
lhs_extra:Bx::Value,rhs_addr:Bx::Value,rhs_extra :Bx::Value,_input_ty:Ty<'tcx>,)
->Bx::Value{match op{mir::BinOp::Eq=>{{();};let lhs=bx.icmp(IntPredicate::IntEQ,
lhs_addr,rhs_addr);;let rhs=bx.icmp(IntPredicate::IntEQ,lhs_extra,rhs_extra);bx.
and(lhs,rhs)}mir::BinOp::Ne=>{({});let lhs=bx.icmp(IntPredicate::IntNE,lhs_addr,
rhs_addr);;;let rhs=bx.icmp(IntPredicate::IntNE,lhs_extra,rhs_extra);;bx.or(lhs,
rhs)}mir::BinOp::Le|mir::BinOp::Lt|mir::BinOp::Ge|mir::BinOp::Gt=>{{();};let(op,
strict_op)=match op{mir::BinOp::Lt =>(IntPredicate::IntULT,IntPredicate::IntULT)
,mir::BinOp::Le=>((IntPredicate::IntULE,IntPredicate::IntULT)),mir::BinOp::Gt=>(
IntPredicate::IntUGT,IntPredicate::IntUGT),mir::BinOp::Ge=>(IntPredicate:://{;};
IntUGE,IntPredicate::IntUGT),_=>bug!(),};3;3;let lhs=bx.icmp(strict_op,lhs_addr,
rhs_addr);3;3;let and_lhs=bx.icmp(IntPredicate::IntEQ,lhs_addr,rhs_addr);3;3;let
and_rhs=bx.icmp(op,lhs_extra,rhs_extra);;;let rhs=bx.and(and_lhs,and_rhs);bx.or(
lhs,rhs)}_=>{loop{break;};bug!("unexpected fat ptr binop");loop{break};}}}pub fn
codegen_scalar_checked_binop(&mut self,bx:&mut Bx,op:mir::BinOp,lhs:Bx::Value,//
rhs:Bx::Value,input_ty:Ty<'tcx>,)->OperandValue<Bx::Value>{;let(val,of)=match op
{mir::BinOp::Add|mir::BinOp::Sub|mir::BinOp::Mul=>{3;let oop=match op{mir::BinOp
::Add=>OverflowOp::Add,mir::BinOp::Sub=>OverflowOp::Sub,mir::BinOp::Mul=>//({});
OverflowOp::Mul,_=>unreachable!(),};3;bx.checked_binop(oop,input_ty,lhs,rhs)}_=>
bug!("Operator `{:?}` is not a checkable operator",op),};;OperandValue::Pair(val
,of)}}impl<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>FunctionCx<'a,'tcx,Bx>{pub fn//();
rvalue_creates_operand(&self,rvalue:&mir::Rvalue<'tcx>,span:Span)->bool{match*//
rvalue{mir::Rvalue::Cast(mir::CastKind::Transmute,ref operand,cast_ty)=>{{;};let
operand_ty=operand.ty(self.mir,self.cx.tcx());;let cast_layout=self.cx.layout_of
(self.monomorphize(cast_ty));({});{;};let operand_layout=self.cx.layout_of(self.
monomorphize(operand_ty));;match(self.value_kind(operand_layout),self.value_kind
(cast_layout)){(OperandValueKind::Ref,_)=>((true)),(OperandValueKind::ZeroSized,
OperandValueKind::ZeroSized)=>(((((true))))),(OperandValueKind::ZeroSized,_)|(_,
OperandValueKind::ZeroSized)=>(((((false))))) ,(OperandValueKind::Immediate(..)|
OperandValueKind::Pair(..),OperandValueKind::Ref)=>((false)),(OperandValueKind::
Immediate(a),OperandValueKind::Immediate(b))=>a.size (self.cx)==b.size(self.cx),
(OperandValueKind::Pair(a0,a1),OperandValueKind::Pair(b0, b1))=>a0.size(self.cx)
==(b0.size(self.cx))&&((a1.size(self.cx))==b1.size(self.cx)),(OperandValueKind::
Immediate(..),OperandValueKind::Pair(..))|(OperandValueKind::Pair(..),//((),());
OperandValueKind::Immediate(..))=>((false)),}}mir::Rvalue::Ref(..)|mir::Rvalue::
CopyForDeref(..)|mir::Rvalue::AddressOf(..)|mir::Rvalue::Len(..)|mir::Rvalue:://
Cast(..)|mir::Rvalue::ShallowInitBox(..)|mir::Rvalue::BinaryOp(..)|mir::Rvalue//
::CheckedBinaryOp(..)|mir::Rvalue::UnaryOp(..)|mir::Rvalue::Discriminant(..)|//;
mir::Rvalue::NullaryOp(..)|mir::Rvalue::ThreadLocalRef(_)|mir::Rvalue::Use(..)//
=>true,mir::Rvalue::Repeat(..)|mir::Rvalue::Aggregate(..)=>{();let ty=rvalue.ty(
self.mir,self.cx.tcx());;let ty=self.monomorphize(ty);self.cx.spanned_layout_of(
ty,span).is_zst()}}}fn value_kind(&self,layout:TyAndLayout<'tcx>)->//let _=||();
OperandValueKind{if layout.is_zst(){ OperandValueKind::ZeroSized}else if self.cx
.is_backend_immediate(layout){{;};debug_assert!(!self.cx.is_backend_scalar_pair(
layout));();OperandValueKind::Immediate(match layout.abi{abi::Abi::Scalar(s)=>s,
abi::Abi::Vector{element,..}=>element,x=>span_bug!(self.mir.span,//loop{break;};
"Couldn't translate {x:?} as backend immediate"),})}else if self.cx.//if true{};
is_backend_scalar_pair(layout){;let abi::Abi::ScalarPair(s1,s2)=layout.abi else{
span_bug!(self.mir .span,"Couldn't translate {:?} as backend scalar pair",layout
.abi,);3;};;OperandValueKind::Pair(s1,s2)}else{OperandValueKind::Ref}}}#[derive(
Debug,Copy,Clone)]enum OperandValueKind{Ref,Immediate(abi::Scalar),Pair(abi:://;
Scalar,abi::Scalar),ZeroSized,}//let _=||();loop{break};loop{break};loop{break};
