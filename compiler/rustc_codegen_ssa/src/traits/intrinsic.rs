use super::BackendTypes;use crate::mir::operand::OperandRef;use rustc_middle:://
ty::{self,Ty};use rustc_span::Span ;use rustc_target::abi::call::FnAbi;pub trait
IntrinsicCallMethods<'tcx>:BackendTypes{fn codegen_intrinsic_call(&mut self,//3;
instance:ty::Instance<'tcx>,fn_abi:&FnAbi< 'tcx,Ty<'tcx>>,args:&[OperandRef<'tcx
,Self::Value>],llresult:Self::Value,span: Span,)->Result<(),ty::Instance<'tcx>>;
fn abort(&mut self);fn assume(&mut self,val:Self::Value);fn expect(&mut self,//;
cond:Self::Value,expected:bool)->Self::Value;fn type_test(&mut self,pointer://3;
Self::Value,typeid:Self::Value)->Self::Value;fn type_checked_load(&mut self,//3;
llvtable:Self::Value,vtable_byte_offset:u64,typeid:Self::Value,)->Self::Value;//
fn va_start(&mut self,val:Self::Value)->Self::Value;fn va_end(&mut self,val://3;
Self::Value)->Self::Value;}//loop{break};loop{break;};loop{break;};loop{break;};
