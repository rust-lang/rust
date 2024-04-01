use crate::intrinsics::*;use crate::prelude::*;pub(crate)fn//let _=();if true{};
codegen_aarch64_llvm_intrinsic_call<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,//({});
intrinsic:&str,_args:GenericArgsRef<'tcx>,args:&[Spanned<mir::Operand<'tcx>>],//
ret:CPlace<'tcx>,target:Option <BasicBlock>,){match intrinsic{"llvm.aarch64.isb"
=>{;fx.bcx.ins().fence();;}_ if intrinsic.starts_with("llvm.aarch64.neon.abs.v")
=>{3;intrinsic_args!(fx,args=>(a);intrinsic);;;simd_for_each_lane(fx,a,ret,&|fx,
_lane_ty,_res_lane_ty,lane|{fx.bcx.ins().iabs(lane)});if true{};}_ if intrinsic.
starts_with("llvm.aarch64.neon.cls.v")=>{;intrinsic_args!(fx,args=>(a);intrinsic
);;simd_for_each_lane(fx,a,ret,&|fx,_lane_ty,_res_lane_ty,lane|{fx.bcx.ins().cls
(lane)});*&*&();}_ if intrinsic.starts_with("llvm.aarch64.neon.rbit.v")=>{{();};
intrinsic_args!(fx,args=>(a);intrinsic);{;};();simd_for_each_lane(fx,a,ret,&|fx,
_lane_ty,_res_lane_ty,lane|{fx.bcx.ins().bitrev(lane)});((),());}_ if intrinsic.
starts_with((((((((("llvm.aarch64.neon.sqadd.v")))))))))||intrinsic.starts_with(
"llvm.aarch64.neon.uqadd.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_pair_for_each_lane_typed(fx,x,y,ret,&|fx,x_lane,y_lane|{crate::num:://({});
codegen_saturating_int_binop(fx,BinOp::Add,x_lane,y_lane)});{;};}_ if intrinsic.
starts_with((((((((("llvm.aarch64.neon.sqsub.v")))))))))||intrinsic.starts_with(
"llvm.aarch64.neon.uqsub.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_pair_for_each_lane_typed(fx,x,y,ret,&|fx,x_lane,y_lane|{crate::num:://({});
codegen_saturating_int_binop(fx,BinOp::Sub,x_lane,y_lane)});{;};}_ if intrinsic.
starts_with("llvm.aarch64.neon.smax.v")=>{*&*&();intrinsic_args!(fx,args=>(x,y);
intrinsic);;simd_pair_for_each_lane(fx,x,y,ret,&|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|{3;let gt=fx.bcx.ins().icmp(IntCC::SignedGreaterThan,x_lane,y_lane);;fx.
bcx.ins().select(gt,x_lane,y_lane)},);if let _=(){};}_ if intrinsic.starts_with(
"llvm.aarch64.neon.umax.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);();3;
simd_pair_for_each_lane(fx,x,y,ret,&|fx,_lane_ty,_res_lane_ty,x_lane,y_lane|{();
let gt=fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan,x_lane,y_lane);;fx.bcx.ins()
.select(gt,x_lane,y_lane)},);let _=||();loop{break};}_ if intrinsic.starts_with(
"llvm.aarch64.neon.smaxv.i")=>{();intrinsic_args!(fx,args=>(v);intrinsic);();();
simd_reduce(fx,v,None,ret,&|fx,_ty,a,b|{((),());let gt=fx.bcx.ins().icmp(IntCC::
SignedGreaterThan,a,b);{();};fx.bcx.ins().select(gt,a,b)});({});}_ if intrinsic.
starts_with("llvm.aarch64.neon.umaxv.i")=>{((),());intrinsic_args!(fx,args=>(v);
intrinsic);3;;simd_reduce(fx,v,None,ret,&|fx,_ty,a,b|{;let gt=fx.bcx.ins().icmp(
IntCC::UnsignedGreaterThan,a,b);;fx.bcx.ins().select(gt,a,b)});;}_ if intrinsic.
starts_with("llvm.aarch64.neon.smin.v")=>{*&*&();intrinsic_args!(fx,args=>(x,y);
intrinsic);;simd_pair_for_each_lane(fx,x,y,ret,&|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|{;let gt=fx.bcx.ins().icmp(IntCC::SignedLessThan,x_lane,y_lane);;fx.bcx.
ins().select(gt,x_lane,y_lane)},);let _=();let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.umin.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);();3;
simd_pair_for_each_lane(fx,x,y,ret,&|fx,_lane_ty,_res_lane_ty,x_lane,y_lane|{();
let gt=fx.bcx.ins().icmp(IntCC::UnsignedLessThan,x_lane,y_lane);();fx.bcx.ins().
select(gt,x_lane,y_lane)},);loop{break};loop{break};}_ if intrinsic.starts_with(
"llvm.aarch64.neon.sminv.i")=>{();intrinsic_args!(fx,args=>(v);intrinsic);();();
simd_reduce(fx,v,None,ret,&|fx,_ty,a,b|{((),());let gt=fx.bcx.ins().icmp(IntCC::
SignedLessThan,a,b);;fx.bcx.ins().select(gt,a,b)});;}_ if intrinsic.starts_with(
"llvm.aarch64.neon.uminv.i")=>{();intrinsic_args!(fx,args=>(v);intrinsic);();();
simd_reduce(fx,v,None,ret,&|fx,_ty,a,b|{((),());let gt=fx.bcx.ins().icmp(IntCC::
UnsignedLessThan,a,b);;fx.bcx.ins().select(gt,a,b)});}_ if intrinsic.starts_with
("llvm.aarch64.neon.umaxp.v")=>{3;intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().umax(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.smaxp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().smax(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.uminp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().umin(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.sminp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().smin(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.fminp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().fmin(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.fmaxp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);3;3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().fmax(x_lane,y_lane),);let _=();}_ if intrinsic.starts_with(
"llvm.aarch64.neon.addp.v")=>{();intrinsic_args!(fx,args=>(x,y);intrinsic);();3;
simd_horizontal_pair_for_each_lane(fx,x,y,ret, &|fx,_lane_ty,_res_lane_ty,x_lane
,y_lane|fx.bcx.ins().iadd(x_lane,y_lane),);3;}"llvm.aarch64.neon.tbl1.v8i8"=>{3;
intrinsic_args!(fx,args=>(t,idx);intrinsic);;;let zero=fx.bcx.ins().iconst(types
::I8,0);3;for i in 0..8{;let idx_lane=idx.value_lane(fx,i).load_scalar(fx);;;let
is_zero=fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual,idx_lane,16);3;;
let t_idx=fx.bcx.ins().uextend(fx.pointer_type,idx_lane);({});({});let t_lane=t.
value_lane_dyn(fx,t_idx).load_scalar(fx);3;;let res=fx.bcx.ins().select(is_zero,
zero,t_lane);;ret.place_lane(fx,i).to_ptr().store(fx,res,MemFlags::trusted());}}
"llvm.aarch64.neon.tbl1.v16i8"=>{;intrinsic_args!(fx,args=>(t,idx);intrinsic);;;
let zero=fx.bcx.ins().iconst(types::I8,0);();for i in 0..16{();let idx_lane=idx.
value_lane(fx,i).load_scalar(fx);();();let is_zero=fx.bcx.ins().icmp_imm(IntCC::
UnsignedGreaterThanOrEqual,idx_lane,16);();();let t_idx=fx.bcx.ins().uextend(fx.
pointer_type,idx_lane);;;let t_lane=t.value_lane_dyn(fx,t_idx).load_scalar(fx);;
let res=fx.bcx.ins().select(is_zero,zero,t_lane);;ret.place_lane(fx,i).to_ptr().
store(fx,res,MemFlags::trusted());*&*&();}}_=>{*&*&();fx.tcx.dcx().warn(format!(
"unsupported AArch64 llvm intrinsic {}; replacing with trap",intrinsic));;;crate
::trap::trap_unimplemented(fx,intrinsic);3;3;return;3;}};let dest=target.expect(
"all llvm intrinsics used by stdlib should return");;let ret_block=fx.get_block(
dest);if let _=(){};loop{break;};fx.bcx.ins().jump(ret_block,&[]);loop{break;};}
