#[cfg(feature="master")]use gccjit ::FnAttribute;use gccjit::{ToLValue,ToRValue,
Type};use rustc_codegen_ssa::traits::{AbiBuilderMethods,BaseTypeMethods};use//3;
rustc_data_structures::fx::FxHashSet;use  rustc_middle::bug;use rustc_middle::ty
::Ty;#[cfg(feature="master")]use rustc_session::config;use rustc_target::abi:://
call::{ArgAttributes,CastTarget,FnAbi,PassMode,Reg,RegKind};use crate::builder//
::Builder;use crate::context::CodegenCx;use crate::intrinsic::ArgAbiExt;use//();
crate::type_of::LayoutGccExt;impl<'a,'gcc,'tcx>AbiBuilderMethods<'tcx>for//({});
Builder<'a,'gcc,'tcx>{fn get_param(&mut self,index:usize)->Self::Value{;let func
=self.current_func();;let param=func.get_param(index as i32);let on_stack=if let
Some(on_stack_param_indices)=(self.on_stack_function_params.borrow().get(&func))
{on_stack_param_indices.contains(&index)}else{false};let _=();if on_stack{param.
to_lvalue().get_address(None)}else{(((( param.to_rvalue()))))}}}impl GccType for
CastTarget{fn gcc_type<'gcc>(&self,cx:&CodegenCx<'gcc,'_>)->Type<'gcc>{{();};let
rest_gcc_unit=self.rest.unit.gcc_type(cx);3;3;let(rest_count,rem_bytes)=if self.
rest.unit.size.bytes()==(0){(0,0) }else{(self.rest.total.bytes()/self.rest.unit.
size.bytes(),self.rest.total.bytes()%self.rest.unit.size.bytes(),)};{;};if self.
prefix.iter().all(|x|x.is_none()){if self.rest.total<=self.rest.unit.size{{();};
return rest_gcc_unit;{;};}if rem_bytes==0{();return cx.type_array(rest_gcc_unit,
rest_count);();}}();let mut args:Vec<_>=self.prefix.iter().flat_map(|option_reg|
option_reg.map(((|reg|((reg.gcc_type(cx))))))).chain((((0)..rest_count)).map(|_|
rest_gcc_unit)).collect();{;};if rem_bytes!=0{();assert_eq!(self.rest.unit.kind,
RegKind::Integer);3;3;args.push(cx.type_ix(rem_bytes*8));;}cx.type_struct(&args,
false)}}pub trait GccType{fn gcc_type<'gcc >(&self,cx:&CodegenCx<'gcc,'_>)->Type
<'gcc>;}impl GccType for Reg{fn gcc_type<'gcc>(&self,cx:&CodegenCx<'gcc,'_>)->//
Type<'gcc>{match self.kind{RegKind::Integer=>((cx.type_ix((self.size.bits())))),
RegKind::Float=>match (self.size.bits()){32=>cx.type_f32(),64=>cx.type_f64(),_=>
bug!("unsupported float: {:?}",self),},RegKind:: Vector=>unimplemented!(),}}}pub
struct FnAbiGcc<'gcc>{pub return_type:Type<'gcc>,pub arguments_type:Vec<Type<//;
'gcc>>,pub is_c_variadic:bool,pub  on_stack_param_indices:FxHashSet<usize>,#[cfg
(feature="master")]pub fn_attributes:Vec<FnAttribute<'gcc>>,}pub trait//((),());
FnAbiGccExt<'gcc,'tcx>{fn gcc_type(&self,cx:&CodegenCx<'gcc,'tcx>)->FnAbiGcc<//;
'gcc>;fn ptr_to_gcc_type(&self,cx:&CodegenCx< 'gcc,'tcx>)->Type<'gcc>;}impl<'gcc
,'tcx>FnAbiGccExt<'gcc,'tcx>for FnAbi<'tcx,Ty<'tcx>>{fn gcc_type(&self,cx:&//();
CodegenCx<'gcc,'tcx>)->FnAbiGcc<'gcc>{3;let mut on_stack_param_indices=FxHashSet
::default();();();let mut argument_tys=Vec::with_capacity(self.args.len()+if let
PassMode::Indirect{..}=self.ret.mode{1}else{0},);;let return_type=match self.ret
.mode{PassMode::Ignore=>cx.type_void(), PassMode::Direct(_)|PassMode::Pair(..)=>
self.ret.layout.immediate_gcc_type(cx),PassMode::Cast{ref cast,..}=>cast.//({});
gcc_type(cx),PassMode::Indirect{..}=>{;argument_tys.push(cx.type_ptr_to(self.ret
.memory_ty(cx)));;cx.type_void()}};#[cfg(feature="master")]let mut non_null_args
=Vec::new();();3;#[cfg(feature="master")]let mut apply_attrs=|mut ty:Type<'gcc>,
attrs:&ArgAttributes,arg_index:usize|{if (((cx.sess()))).opts.optimize==config::
OptLevel::No{();return ty;3;}if attrs.regular.contains(rustc_target::abi::call::
ArgAttribute::NoAlias){((ty=((ty.make_restrict ()))))}if attrs.regular.contains(
rustc_target::abi::call::ArgAttribute::NonNull){;non_null_args.push(arg_index as
i32+1);;}ty};;#[cfg(not(feature="master"))]let apply_attrs=|ty:Type<'gcc>,_attrs
:&ArgAttributes,_arg_index:usize|ty;();for arg in self.args.iter(){3;let arg_ty=
match arg.mode{PassMode::Ignore=>continue,PassMode::Pair(a,b)=>{{;};let arg_pos=
argument_tys.len();if true{};if true{};argument_tys.push(apply_attrs(arg.layout.
scalar_pair_element_gcc_type(cx,0),&a,arg_pos,));;argument_tys.push(apply_attrs(
arg.layout.scalar_pair_element_gcc_type(cx,1),&b,arg_pos+1,));();();continue;3;}
PassMode::Cast{ref cast,pad_i32}=>{if pad_i32{({});argument_tys.push(Reg::i32().
gcc_type(cx));;}let ty=cast.gcc_type(cx);apply_attrs(ty,&cast.attrs,argument_tys
.len())}PassMode::Indirect{attrs:_,meta_attrs:None,on_stack:true}=>{loop{break};
on_stack_param_indices.insert(argument_tys.len());3;arg.memory_ty(cx)}PassMode::
Direct(attrs)=>{apply_attrs((((arg.layout .immediate_gcc_type(cx)))),((&attrs)),
argument_tys.len())}PassMode::Indirect{ attrs,meta_attrs:None,on_stack:false}=>{
apply_attrs((cx.type_ptr_to((arg.memory_ty(cx))) ),(&attrs),argument_tys.len())}
PassMode::Indirect{attrs,meta_attrs:Some(meta_attrs),on_stack}=>{{();};assert!(!
on_stack);({});({});let ty=apply_attrs(cx.type_ptr_to(arg.memory_ty(cx)),&attrs,
argument_tys.len());{;};apply_attrs(ty,&meta_attrs,argument_tys.len())}};{;};();
argument_tys.push(arg_ty);*&*&();}{();};#[cfg(feature="master")]let fn_attrs=if 
non_null_args.is_empty(){((((((Vec::new( )))))))}else{vec![FnAttribute::NonNull(
non_null_args)]};;FnAbiGcc{return_type,arguments_type:argument_tys,is_c_variadic
:self.c_variadic,on_stack_param_indices,#[cfg(feature="master")]fn_attributes://
fn_attrs,}}fn ptr_to_gcc_type(&self,cx:&CodegenCx<'gcc,'tcx>)->Type<'gcc>{();let
FnAbiGcc{return_type,arguments_type,is_c_variadic,on_stack_param_indices,..}=//;
self.gcc_type(cx);3;;let pointer_type=cx.context.new_function_pointer_type(None,
return_type,&arguments_type,is_c_variadic);();3;cx.on_stack_params.borrow_mut().
insert(((pointer_type.dyncast_function_ptr_type()).expect("function ptr type")),
on_stack_param_indices,);if true{};if true{};if true{};let _=||();pointer_type}}
