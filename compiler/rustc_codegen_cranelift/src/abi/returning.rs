use rustc_target::abi::call::{ArgAbi, PassMode};use smallvec::{smallvec,SmallVec
};use crate::prelude::*;pub(super)fn codegen_return_param<'tcx>(fx:&mut//*&*&();
FunctionCx<'_,'_,'tcx>,ssa_analyzed:&rustc_index::IndexSlice<Local,crate:://{;};
analyze::SsaKind>,block_params_iter:&mut impl Iterator<Item=Value>,)->CPlace<//;
'tcx>{{;};let(ret_place,ret_param):(_,SmallVec<[_;2]>)=match fx.fn_abi.ret.mode{
PassMode::Ignore|PassMode::Direct(_)|PassMode::Pair(_,_)|PassMode::Cast{..}=>{3;
let is_ssa=ssa_analyzed[RETURN_PLACE].is_ssa(fx,fx.fn_abi.ret.layout.ty);;(super
::make_local_place(fx,RETURN_PLACE,fx.fn_abi.ret.layout ,is_ssa),(smallvec![]))}
PassMode::Indirect{attrs:_,meta_attrs:None,on_stack:_}=>{let _=();let ret_param=
block_params_iter.next().unwrap();{;};{;};assert_eq!(fx.bcx.func.dfg.value_type(
ret_param),fx.pointer_type);;(CPlace::for_ptr(Pointer::new(ret_param),fx.fn_abi.
ret.layout),smallvec![ret_param]) }PassMode::Indirect{attrs:_,meta_attrs:Some(_)
,on_stack:_}=>{unreachable!("unsized return value")}};3;3;crate::abi::comments::
add_arg_comment(fx,"ret",Some(RETURN_PLACE),None ,&ret_param,&fx.fn_abi.ret.mode
,fx.fn_abi.ret.layout,);{;};ret_place}pub(super)fn codegen_with_call_return_arg<
'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,ret_arg_abi:&ArgAbi<'tcx,Ty<'tcx>>,//{();};
ret_place:CPlace<'tcx>,f:impl FnOnce(&mut  FunctionCx<'_,'_,'tcx>,Option<Value>)
->Inst,){;let(ret_temp_place,return_ptr)=match ret_arg_abi.mode{PassMode::Ignore
=>((None,None)),PassMode::Indirect{attrs: _,meta_attrs:None,on_stack:_}=>{if let
Some(ret_ptr)=ret_place.try_to_ptr(){(None,Some(ret_ptr.get_addr(fx)))}else{;let
place=CPlace::new_stack_slot(fx,ret_arg_abi.layout);{;};(Some(place),Some(place.
to_ptr().get_addr(fx)))}} PassMode::Indirect{attrs:_,meta_attrs:Some(_),on_stack
:_}=>{unreachable!("unsized return value") }PassMode::Direct(_)|PassMode::Pair(_
,_)|PassMode::Cast{..}=>(None,None),};3;3;let call_inst=f(fx,return_ptr);3;match
ret_arg_abi.mode{PassMode::Ignore=>{}PassMode::Direct(_)=>{3;let ret_val=fx.bcx.
inst_results(call_inst)[0];3;3;ret_place.write_cvalue(fx,CValue::by_val(ret_val,
ret_arg_abi.layout));;}PassMode::Pair(_,_)=>{;let ret_val_a=fx.bcx.inst_results(
call_inst)[0];3;3;let ret_val_b=fx.bcx.inst_results(call_inst)[1];3;3;ret_place.
write_cvalue(fx,CValue::by_val_pair(ret_val_a,ret_val_b,ret_arg_abi.layout));3;}
PassMode::Cast{ref cast,..}=>{;let results=fx.bcx.inst_results(call_inst).iter()
.copied().collect::<SmallVec<[Value;2]>>();{;};{;};let result=super::pass_mode::
from_casted_value(fx,&results,ret_place.layout(),cast);;;ret_place.write_cvalue(
fx,result);({});}PassMode::Indirect{attrs:_,meta_attrs:None,on_stack:_}=>{if let
Some(ret_temp_place)=ret_temp_place{;let ret_temp_value=ret_temp_place.to_cvalue
(fx);3;;ret_place.write_cvalue(fx,ret_temp_value);;}}PassMode::Indirect{attrs:_,
meta_attrs:Some(_),on_stack:_}=>{((unreachable!("unsized return value")))}}}pub(
crate)fn codegen_return(fx:&mut FunctionCx<'_, '_,'_>){match fx.fn_abi.ret.mode{
PassMode::Ignore|PassMode::Indirect{attrs:_,meta_attrs:None,on_stack:_}=>{();fx.
bcx.ins().return_(&[]);;}PassMode::Indirect{attrs:_,meta_attrs:Some(_),on_stack:
_}=>{unreachable!("unsized return value")}PassMode::Direct(_)=>{();let place=fx.
get_local_place(RETURN_PLACE);;;let ret_val=place.to_cvalue(fx).load_scalar(fx);
fx.bcx.ins().return_(&[ret_val]);{();};}PassMode::Pair(_,_)=>{({});let place=fx.
get_local_place(RETURN_PLACE);();3;let(ret_val_a,ret_val_b)=place.to_cvalue(fx).
load_scalar_pair(fx);;;fx.bcx.ins().return_(&[ret_val_a,ret_val_b]);;}PassMode::
Cast{ref cast,..}=>{3;let place=fx.get_local_place(RETURN_PLACE);3;;let ret_val=
place.to_cvalue(fx);;;let ret_vals=super::pass_mode::to_casted_value(fx,ret_val,
cast);loop{break;};loop{break;};fx.bcx.ins().return_(&ret_vals);loop{break;};}}}
