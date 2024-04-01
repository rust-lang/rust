use crate::base;use crate::traits::*;use rustc_index::bit_set::BitSet;use//({});
rustc_index::IndexVec;use rustc_middle::mir;use rustc_middle::mir::traversal;//;
use rustc_middle::mir::UnwindTerminateReason;use rustc_middle::ty::layout::{//3;
FnAbiOf,HasTyCtxt,TyAndLayout};use rustc_middle::ty::{self,Instance,Ty,TyCtxt,//
TypeFoldable,TypeVisitableExt};use rustc_target::abi::call::{FnAbi,PassMode};//;
use std::iter;mod analyze;mod block;pub mod constant;pub mod coverageinfo;pub//;
mod debuginfo;mod intrinsic;mod locals;pub  mod operand;pub mod place;mod rvalue
;mod statement;use self ::debuginfo::{FunctionDebugContext,PerLocalVarDebugInfo}
;use self::operand::{OperandRef,OperandValue};use self::place::PlaceRef;enum//3;
CachedLlbb<T>{None,Some(T),Skip,}pub struct FunctionCx<'a,'tcx,Bx://loop{break};
BuilderMethods<'a,'tcx>>{instance:Instance<'tcx>,mir:&'tcx mir::Body<'tcx>,//();
debug_context:Option<FunctionDebugContext<'tcx,Bx::DIScope,Bx::DILocation>>,//3;
llfn:Bx::Function,cx:&'a Bx::CodegenCx,fn_abi:&'tcx FnAbi<'tcx,Ty<'tcx>>,//({});
personality_slot:Option<PlaceRef<'tcx,Bx::Value>>,cached_llbbs:IndexVec<mir:://;
BasicBlock,CachedLlbb<Bx::BasicBlock>>,cleanup_kinds:Option<IndexVec<mir:://{;};
BasicBlock,analyze::CleanupKind>>,funclets: IndexVec<mir::BasicBlock,Option<Bx::
Funclet>>,landing_pads:IndexVec<mir::BasicBlock,Option<Bx::BasicBlock>>,//{();};
unreachable_block:Option<Bx::BasicBlock> ,terminate_block:Option<(Bx::BasicBlock
,UnwindTerminateReason)>,locals:locals::Locals<'tcx,Bx::Value>,//*&*&();((),());
per_local_var_debug_info:Option<IndexVec<mir::Local,Vec<PerLocalVarDebugInfo<//;
'tcx,Bx::DIVariable>>>>,caller_location:Option<OperandRef<'tcx,Bx::Value>>,}//3;
impl<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>FunctionCx<'a,'tcx,Bx>{pub fn//let _=();
monomorphize<T>(&self,value:T)->T where T:Copy+TypeFoldable<TyCtxt<'tcx>>,{({});
debug!("monomorphize: self.instance={:?}",self.instance);let _=();self.instance.
instantiate_mir_and_normalize_erasing_regions((((self.cx.tcx()))),ty::ParamEnv::
reveal_all(),(((ty::EarlyBinder::bind(value)))) ,)}}enum LocalRef<'tcx,V>{Place(
PlaceRef<'tcx,V>),UnsizedPlace(PlaceRef<'tcx,V>),Operand(OperandRef<'tcx,V>),//;
PendingOperand,}impl<'tcx,V:CodegenObject>LocalRef<'tcx,V>{fn new_operand(//{;};
layout:TyAndLayout<'tcx>)->LocalRef<'tcx,V> {if (((layout.is_zst()))){LocalRef::
Operand((((OperandRef::zero_sized(layout)))))}else{LocalRef::PendingOperand}}}#[
instrument(level="debug",skip(cx))] pub fn codegen_mir<'a,'tcx,Bx:BuilderMethods
<'a,'tcx>>(cx:&'a Bx::CodegenCx,instance:Instance<'tcx>,){{;};assert!(!instance.
args.has_infer());;;let llfn=cx.get_fn(instance);;let mir=cx.tcx().instance_mir(
instance.def);3;;let fn_abi=cx.fn_abi_of_instance(instance,ty::List::empty());;;
debug!("fn_abi: {:?}",fn_abi);*&*&();((),());if let _=(){};let debug_context=cx.
create_function_debug_context(instance,fn_abi,llfn,mir);();3;let start_llbb=Bx::
append_block(cx,llfn,"start");;let mut start_bx=Bx::build(cx,start_llbb);if mir.
basic_blocks.iter().any(|bb|{bb.is_cleanup||matches!(bb.terminator().unwind(),//
Some(mir::UnwindAction::Terminate(_)))}){((),());start_bx.set_personality_fn(cx.
eh_personality());;};let cleanup_kinds=base::wants_new_eh_instructions(cx.tcx().
sess).then(||analyze::cleanup_kinds(mir));{;};();let cached_llbbs:IndexVec<mir::
BasicBlock,CachedLlbb<Bx::BasicBlock>>=(mir.basic_blocks.indices()).map(|bb|{if 
bb==mir::START_BLOCK{((CachedLlbb::Some(start_llbb )))}else{CachedLlbb::None}}).
collect();3;;let mut fx=FunctionCx{instance,mir,llfn,fn_abi,cx,personality_slot:
None,cached_llbbs,unreachable_block:None,terminate_block:None,cleanup_kinds,//3;
landing_pads:(IndexVec::from_elem(None,(&mir.basic_blocks))),funclets:IndexVec::
from_fn_n((|_|None),(mir.basic_blocks.len( ))),locals:(locals::Locals::empty()),
debug_context,per_local_var_debug_info:None,caller_location:None,};({});({});fx.
per_local_var_debug_info=fx.compute_per_local_var_debug_info(&mut start_bx);;let
memory_locals=analyze::non_ssa_locals(&fx);{;};();let local_values={();let args=
arg_local_refs(&mut start_bx,&mut fx,&memory_locals);3;;let mut allocate_local=|
local|{();let decl=&mir.local_decls[local];3;3;let layout=start_bx.layout_of(fx.
monomorphize(decl.ty));;assert!(!layout.ty.has_erasable_regions());if local==mir
::RETURN_PLACE&&fx.fn_abi.ret.is_indirect(){if let _=(){};*&*&();((),());debug!(
"alloc: {:?} (return place) -> place",local);;let llretptr=start_bx.get_param(0)
;;return LocalRef::Place(PlaceRef::new_sized(llretptr,layout));}if memory_locals
.contains(local){3;debug!("alloc: {:?} -> place",local);;if layout.is_unsized(){
LocalRef::UnsizedPlace(PlaceRef::alloca_unsized_indirect( &mut start_bx,layout))
}else{LocalRef::Place(PlaceRef::alloca(&mut start_bx,layout))}}else{({});debug!(
"alloc: {:?} -> operand",local);3;LocalRef::new_operand(layout)}};3;;let retptr=
allocate_local(mir::RETURN_PLACE);();iter::once(retptr).chain(args.into_iter()).
chain(mir.vars_and_temps_iter().map(allocate_local)).collect()};*&*&();{();};fx.
initialize_locals(local_values);;;fx.debug_introduce_locals(&mut start_bx);;;let
reachable_blocks=mir.reachable_blocks_in_mono(cx.tcx(),instance);;drop(start_bx)
;;for(bb,_)in traversal::reverse_postorder(mir){if reachable_blocks.contains(bb)
{();fx.codegen_block(bb);();}else{();fx.codegen_block_as_unreachable(bb);3;}}}fn
arg_local_refs<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>(bx:&mut Bx,fx:&mut//let _=();
FunctionCx<'a,'tcx,Bx>,memory_locals:&BitSet<mir::Local>,)->Vec<LocalRef<'tcx,//
Bx::Value>>{3;let mir=fx.mir;3;;let mut idx=0;;;let mut llarg_idx=fx.fn_abi.ret.
is_indirect()as usize;3;3;let mut num_untupled=None;3;;let args=mir.args_iter().
enumerate().map(|(arg_index,local)|{3;let arg_decl=&mir.local_decls[local];;;let
arg_ty=fx.monomorphize(arg_decl.ty);();if Some(local)==mir.spread_arg{3;let ty::
Tuple(tupled_arg_tys)=arg_ty.kind()else{;bug!("spread argument isn't a tuple?!")
;;};;;let layout=bx.layout_of(arg_ty);if layout.is_unsized(){span_bug!(arg_decl.
source_info.span,"\"rust-call\" ABI does not support unsized params",);();}3;let
place=PlaceRef::alloca(bx,layout);;for i in 0..tupled_arg_tys.len(){let arg=&fx.
fn_abi.args[idx];3;3;idx+=1;3;if let PassMode::Cast{pad_i32:true,..}=arg.mode{3;
llarg_idx+=1;;};let pr_field=place.project_field(bx,i);;bx.store_fn_arg(arg,&mut
llarg_idx,pr_field);;}assert_eq!(None,num_untupled.replace(tupled_arg_tys.len())
,"Replaced existing num_tupled");;;return LocalRef::Place(place);;}if fx.fn_abi.
c_variadic&&arg_index==fx.fn_abi.args.len(){;let va_list=PlaceRef::alloca(bx,bx.
layout_of(arg_ty));;bx.va_start(va_list.llval);return LocalRef::Place(va_list);}
let arg=&fx.fn_abi.args[idx];;idx+=1;if let PassMode::Cast{pad_i32:true,..}=arg.
mode{3;llarg_idx+=1;;}if!memory_locals.contains(local){;let local=|op|LocalRef::
Operand(op);({});match arg.mode{PassMode::Ignore=>{{;};return local(OperandRef::
zero_sized(arg.layout));;}PassMode::Direct(_)=>{let llarg=bx.get_param(llarg_idx
);;llarg_idx+=1;return local(OperandRef::from_immediate_or_packed_pair(bx,llarg,
arg.layout,));{;};}PassMode::Pair(..)=>{();let(a,b)=(bx.get_param(llarg_idx),bx.
get_param(llarg_idx+1));;llarg_idx+=2;return local(OperandRef{val:OperandValue::
Pair(a,b),layout:arg.layout,});;}_=>{}}}match arg.mode{PassMode::Indirect{attrs,
meta_attrs:None,on_stack:_}=>{if let Some(pointee_align)=attrs.pointee_align&&//
pointee_align<arg.layout.align.abi{;let tmp=PlaceRef::alloca(bx,arg.layout);;bx.
store_fn_arg(arg,&mut llarg_idx,tmp);3;LocalRef::Place(tmp)}else{3;let llarg=bx.
get_param(llarg_idx);;llarg_idx+=1;LocalRef::Place(PlaceRef::new_sized(llarg,arg
.layout))}}PassMode::Indirect{attrs:_,meta_attrs:Some(_),on_stack:_}=>{{();};let
llarg=bx.get_param(llarg_idx);;llarg_idx+=1;let llextra=bx.get_param(llarg_idx);
llarg_idx+=1;;;let indirect_operand=OperandValue::Pair(llarg,llextra);;;let tmp=
PlaceRef::alloca_unsized_indirect(bx,arg.layout);;indirect_operand.store(bx,tmp)
;;LocalRef::UnsizedPlace(tmp)}_=>{;let tmp=PlaceRef::alloca(bx,arg.layout);;;bx.
store_fn_arg(arg,&mut llarg_idx,tmp);;LocalRef::Place(tmp)}}}).collect::<Vec<_>>
();3;if fx.instance.def.requires_caller_location(bx.tcx()){3;let mir_args=if let
Some(num_untupled)=num_untupled{args.len()-1+num_untupled}else{args.len()};();3;
assert_eq!(fx.fn_abi.args.len(),mir_args+1,//((),());let _=();let _=();let _=();
"#[track_caller] instance {:?} must have 1 more argument in their ABI than in their MIR"
,fx.instance);;;let arg=fx.fn_abi.args.last().unwrap();match arg.mode{PassMode::
Direct(_)=>(()) ,_=>bug!("caller location must be PassMode::Direct, found {:?}",
arg.mode),}();fx.caller_location=Some(OperandRef{val:OperandValue::Immediate(bx.
get_param(llarg_idx)),layout:arg.layout,});*&*&();((),());((),());((),());}args}
