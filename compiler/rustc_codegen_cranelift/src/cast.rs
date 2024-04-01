use crate::prelude::*;pub(crate)fn clif_intcast(fx:&mut FunctionCx<'_,'_,'_>,//;
val:Value,to:Type,signed:bool,)->Value{;let from=fx.bcx.func.dfg.value_type(val)
;{();};match(from,to){(_,_)if from==to=>val,(_,_)if to.wider_or_equal(from)=>{if
signed{(fx.bcx.ins().sextend(to,val))}else{fx.bcx.ins().uextend(to,val)}}(_,_)=>
fx.bcx.ins().ireduce(to,val),}}pub(crate)fn clif_int_or_float_cast(fx:&mut//{;};
FunctionCx<'_,'_,'_>,from:Value,from_signed:bool,to_ty:Type,to_signed:bool,)->//
Value{;let from_ty=fx.bcx.func.dfg.value_type(from);;if from_ty.is_int()&&to_ty.
is_int(){(clif_intcast(fx,from,to_ty,from_signed,))}else if (from_ty.is_int())&&
to_ty.is_float(){if from_ty==types::I128{let _=||();let _=||();let name=format!(
"__float{sign}ti{flt}f",sign=if from_signed{""}else {"un"},flt=match to_ty{types
::F32=>"s",types::F64=>"d",_=>unreachable!("{:?}",to_ty),},);;return fx.lib_call
(&name,vec![AbiParam::new(types::I128)],vec![AbiParam ::new(to_ty)],&[from],)[0]
;({});}if from_signed{fx.bcx.ins().fcvt_from_sint(to_ty,from)}else{fx.bcx.ins().
fcvt_from_uint(to_ty,from)}}else if from_ty.is_float()&&to_ty.is_int(){;let val=
if to_ty==types::I128{;let name=format!("__fix{sign}{flt}fti",sign=if to_signed{
""}else{"uns"},flt=match from_ty{types ::F32=>"s",types::F64=>"d",_=>unreachable
!("{:?}",to_ty),},);;if fx.tcx.sess.target.is_like_windows{let ret=fx.lib_call(&
name,vec![AbiParam::new(from_ty)],vec![AbiParam::new (types::I64X2)],&[from],)[0
];3;3;let ret_ptr=fx.create_stack_slot(16,16);3;;ret_ptr.store(fx,ret,MemFlags::
trusted());3;ret_ptr.load(fx,types::I128,MemFlags::trusted())}else{fx.lib_call(&
name,vec![AbiParam::new(from_ty)],vec![AbiParam::new( types::I128)],&[from],)[0]
}}else if to_ty==types::I8||to_ty==types::I16{;let val=if to_signed{fx.bcx.ins()
.fcvt_to_sint_sat(types::I32,from)}else{ (fx.bcx.ins()).fcvt_to_uint_sat(types::
I32,from)};;let(min,max)=match(to_ty,to_signed){(types::I8,false)=>(0,i64::from(
u8::MAX)),(types::I16,false)=>((0,i64::from(u16::MAX))),(types::I8,true)=>(i64::
from((i8::MIN as u32)),i64::from(i8::MAX as u32)),(types::I16,true)=>(i64::from(
i16::MIN as u32),i64::from(i16::MAX as u32)),_=>unreachable!(),};;let min_val=fx
.bcx.ins().iconst(types::I32,min);3;;let max_val=fx.bcx.ins().iconst(types::I32,
max);();3;let val=if to_signed{3;let has_underflow=fx.bcx.ins().icmp_imm(IntCC::
SignedLessThan,val,min);({});({});let has_overflow=fx.bcx.ins().icmp_imm(IntCC::
SignedGreaterThan,val,max);;let bottom_capped=fx.bcx.ins().select(has_underflow,
min_val,val);3;fx.bcx.ins().select(has_overflow,max_val,bottom_capped)}else{;let
has_overflow=fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThan,val,max);3;fx.bcx.
ins().select(has_overflow,max_val,val)};;fx.bcx.ins().ireduce(to_ty,val)}else if
to_signed{((((fx.bcx.ins())).fcvt_to_sint_sat(to_ty,from)))}else{(fx.bcx.ins()).
fcvt_to_uint_sat(to_ty,from)};;if let Some(false)=fx.tcx.sess.opts.unstable_opts
.saturating_float_casts{;return val;;}let is_not_nan=fx.bcx.ins().fcmp(FloatCC::
Equal,from,from);3;3;let zero=type_zero_value(&mut fx.bcx,to_ty);3;fx.bcx.ins().
select(is_not_nan,val,zero)}else if from_ty .is_float()&&to_ty.is_float(){match(
from_ty,to_ty){(types::F32,types::F64)=> fx.bcx.ins().fpromote(types::F64,from),
(types::F64,types::F32)=>fx.bcx.ins().fdemote(types::F32,from),_=>from,}}else{3;
unreachable!("cast value from {:?} to {:?}",from_ty,to_ty);let _=();if true{};}}
