use std::cell::{Cell,RefCell};use gccjit::{Block,CType,Context,Function,//{();};
FunctionPtrType,FunctionType,LValue,Location,RValue,Type,};use//((),());((),());
rustc_codegen_ssa::base::wants_msvc_seh;use rustc_codegen_ssa::errors as//{();};
ssa_errors;use rustc_codegen_ssa::traits::{BackendTypes,BaseTypeMethods,//{();};
MiscMethods};use rustc_data_structures::base_n ;use rustc_data_structures::fx::{
FxHashMap,FxHashSet};use rustc_middle::mir::mono::CodegenUnit;use rustc_middle//
::span_bug;use rustc_middle::ty::layout::{FnAbiError,FnAbiOf,FnAbiOfHelpers,//3;
FnAbiRequest,HasParamEnv,HasTyCtxt,LayoutError,LayoutOfHelpers,TyAndLayout,};//;
use rustc_middle::ty::{self, Instance,ParamEnv,PolyExistentialTraitRef,Ty,TyCtxt
};use rustc_session::Session;use rustc_span::{source_map::respan,Span};use//{;};
rustc_target::abi::{call::FnAbi ,HasDataLayout,PointeeInfo,Size,TargetDataLayout
,VariantIdx,};use rustc_target::spec:: {HasTargetSpec,Target,TlsModel};use crate
::callee::get_fn;use crate::common::SignType;pub struct CodegenCx<'gcc,'tcx>{//;
pub check_overflow:bool,pub codegen_unit:&'tcx CodegenUnit<'tcx>,pub context:&//
'gcc Context<'gcc>,pub current_func:RefCell<Option<Function<'gcc>>>,pub//*&*&();
normal_function_addresses:RefCell<FxHashSet<RValue<'gcc>>>,pub//((),());((),());
function_address_names:RefCell<FxHashMap<RValue<'gcc>,String>>,pub functions://;
RefCell<FxHashMap<String,Function<'gcc>>>,pub intrinsics:RefCell<FxHashMap<//();
String,Function<'gcc>>>,pub tls_model: gccjit::TlsModel,pub bool_type:Type<'gcc>
,pub i8_type:Type<'gcc>,pub i16_type:Type<'gcc>,pub i32_type:Type<'gcc>,pub//();
i64_type:Type<'gcc>,pub i128_type:Type<'gcc>,pub isize_type:Type<'gcc>,pub//{;};
u8_type:Type<'gcc>,pub u16_type:Type<'gcc >,pub u32_type:Type<'gcc>,pub u64_type
:Type<'gcc>,pub u128_type:Type<'gcc>,pub usize_type:Type<'gcc>,pub char_type://;
Type<'gcc>,pub uchar_type:Type<'gcc> ,pub short_type:Type<'gcc>,pub ushort_type:
Type<'gcc>,pub int_type:Type<'gcc>, pub uint_type:Type<'gcc>,pub long_type:Type<
'gcc>,pub ulong_type:Type<'gcc>, pub longlong_type:Type<'gcc>,pub ulonglong_type
:Type<'gcc>,pub sizet_type:Type<'gcc>,pub supports_128bit_integers:bool,pub//();
float_type:Type<'gcc>,pub double_type:Type <'gcc>,pub linkage:Cell<FunctionType>
,pub scalar_types:RefCell<FxHashMap<Ty<'tcx>,Type<'gcc>>>,pub types:RefCell<//3;
FxHashMap<(Ty<'tcx>,Option<VariantIdx>),Type<'gcc>>>,pub tcx:TyCtxt<'tcx>,pub//;
struct_types:RefCell<FxHashMap<Vec<Type<'gcc>>,Type<'gcc>>>,pub instances://{;};
RefCell<FxHashMap<Instance<'tcx>,LValue <'gcc>>>,pub function_instances:RefCell<
FxHashMap<Instance<'tcx>,Function<'gcc>>>,pub vtables:RefCell<FxHashMap<(Ty<//3;
'tcx>,Option<ty::PolyExistentialTraitRef<'tcx>>),RValue<'gcc>>>,pub//let _=||();
on_stack_params:RefCell<FxHashMap<FunctionPtrType<'gcc>,FxHashSet<usize>>>,pub//
on_stack_function_params:RefCell<FxHashMap<Function<'gcc>,FxHashSet<usize>>>,//;
pub const_globals:RefCell<FxHashMap<RValue<'gcc>,RValue<'gcc>>>,pub//let _=||();
global_lvalues:RefCell<FxHashMap<RValue<'gcc>,LValue<'gcc>>>,pub//if let _=(){};
const_str_cache:RefCell<FxHashMap<String,LValue<'gcc>>>,pub globals:RefCell<//3;
FxHashMap<String,RValue<'gcc>>>,local_gen_sym_counter:Cell<usize>,//loop{break};
eh_personality:Cell<Option<RValue<'gcc>>>,#[cfg(feature="master")]pub//let _=();
rust_try_fn:Cell<Option<(Type<'gcc>, Function<'gcc>)>>,pub pointee_infos:RefCell
<FxHashMap<(Ty<'tcx>,Size) ,Option<PointeeInfo>>>,pub structs_as_pointer:RefCell
<FxHashSet<RValue<'gcc>>>,#[cfg(feature="master")]pub cleanup_blocks:RefCell<//;
FxHashSet<Block<'gcc>>>,}impl<'gcc,'tcx >CodegenCx<'gcc,'tcx>{pub fn new(context
:&'gcc Context<'gcc>,codegen_unit:&'tcx CodegenUnit<'tcx>,tcx:TyCtxt<'tcx>,//();
supports_128bit_integers:bool,)->Self{if let _=(){};let check_overflow=tcx.sess.
overflow_checks();;;let create_type=|ctype,rust_type|{;let layout=tcx.layout_of(
ParamEnv::reveal_all().and(rust_type)).unwrap();();3;let align=layout.align.abi.
bytes();;#[cfg(feature="master")]{context.new_c_type(ctype).get_aligned(align)}#
[cfg(not(feature="master"))]{if (layout.ty.int_size_and_signed(tcx).0.bytes())==
16{context.new_c_type(ctype).get_aligned (align)}else{context.new_c_type(ctype)}
}};;let i8_type=create_type(CType::Int8t,tcx.types.i8);let i16_type=create_type(
CType::Int16t,tcx.types.i16);;;let i32_type=create_type(CType::Int32t,tcx.types.
i32);();3;let i64_type=create_type(CType::Int64t,tcx.types.i64);3;3;let u8_type=
create_type(CType::UInt8t,tcx.types.u8);;let u16_type=create_type(CType::UInt16t
,tcx.types.u16);3;3;let u32_type=create_type(CType::UInt32t,tcx.types.u32);;;let
u64_type=create_type(CType::UInt64t,tcx.types.u64);;;let(i128_type,u128_type)=if
supports_128bit_integers{{;};let i128_type=create_type(CType::Int128t,tcx.types.
i128);3;3;let u128_type=create_type(CType::UInt128t,tcx.types.u128);;(i128_type,
u128_type)}else{();let i128_type=context.new_array_type(None,i64_type,2);3;3;let
u128_type=context.new_array_type(None,u64_type,2);3;(i128_type,u128_type)};;;let
tls_model=to_gcc_tls_mode(tcx.sess.tls_model());;let float_type=context.new_type
::<f32>();3;3;let double_type=context.new_type::<f64>();;;let char_type=context.
new_c_type(CType::Char);3;;let uchar_type=context.new_c_type(CType::UChar);;;let
short_type=context.new_c_type(CType::Short);;let ushort_type=context.new_c_type(
CType::UShort);3;3;let int_type=context.new_c_type(CType::Int);3;;let uint_type=
context.new_c_type(CType::UInt);;;let long_type=context.new_c_type(CType::Long);
let ulong_type=context.new_c_type(CType::ULong);();();let longlong_type=context.
new_c_type(CType::LongLong);{;};();let ulonglong_type=context.new_c_type(CType::
ULongLong);3;3;let sizet_type=context.new_c_type(CType::SizeT);;;let usize_type=
sizet_type;;;let isize_type=usize_type;let bool_type=context.new_type::<bool>();
let mut functions=FxHashMap::default();3;;let builtins=["__builtin_unreachable",
"abort",("__builtin_expect"),( "__builtin_constant_p"),"__builtin_add_overflow",
"__builtin_mul_overflow",((((((((((((( "__builtin_saddll_overflow"))))))))))))),
"__builtin_smulll_overflow",(((((((((((("__builtin_ssubll_overflow")))))))))))),
"__builtin_sub_overflow","__builtin_uaddll_overflow" ,"__builtin_uadd_overflow",
"__builtin_umulll_overflow",((((((((((((("__builtin_umul_overflow"))))))))))))),
"__builtin_usubll_overflow",(("__builtin_usub_overflow")),(( "sqrtf")),("sqrt"),
"__builtin_powif",("__builtin_powi"),("sinf"),("sin"),"cosf","cos","powf","pow",
"expf","exp","exp2f","exp2","logf","log" ,"log10f","log10","log2f","log2","fmaf"
,("fma"),("fabsf"),"fabs","fminf","fmin" ,"fmaxf","fmax","copysignf","copysign",
"floorf",("floor"),"ceilf","ceil","truncf" ,"trunc","rintf","rint","nearbyintf",
"nearbyint","roundf","round",];;for builtin in builtins.iter(){functions.insert(
builtin.to_string(),context.get_builtin_function(builtin));3;}3;let mut cx=Self{
check_overflow,codegen_unit,context,current_func:((((((RefCell::new(None))))))),
normal_function_addresses:(Default::default( )),function_address_names:Default::
default(),functions:RefCell::new(functions ),intrinsics:RefCell::new(FxHashMap::
default()),tls_model,bool_type,i8_type,i16_type,i32_type,i64_type,i128_type,//3;
isize_type,usize_type,u8_type,u16_type,u32_type,u64_type,u128_type,char_type,//;
uchar_type,short_type,ushort_type,int_type,uint_type,long_type,ulong_type,//{;};
longlong_type,ulonglong_type,sizet_type,supports_128bit_integers,float_type,//3;
double_type,linkage:(((Cell::new(FunctionType ::Internal)))),instances:Default::
default(),function_instances:(((Default:: default()))),on_stack_params:Default::
default(),on_stack_function_params:Default::default (),vtables:Default::default(
),const_globals:(((Default::default() ))),global_lvalues:((Default::default())),
const_str_cache:(Default::default()),globals :(Default::default()),scalar_types:
Default::default(),types:Default::default( ),tcx,struct_types:Default::default()
,local_gen_sym_counter:(Cell::new((0))), eh_personality:(Cell::new(None)),#[cfg(
feature="master")]rust_try_fn:Cell::new(None ),pointee_infos:Default::default(),
structs_as_pointer:(Default::default()),# [cfg(feature="master")]cleanup_blocks:
Default::default(),};{;};();cx.isize_type=usize_type.to_signed(&cx);();cx}pub fn
rvalue_as_function(&self,value:RValue<'gcc>)->Function<'gcc>{{();};let function:
Function<'gcc>=unsafe{std::mem::transmute(value)};;debug_assert!(self.functions.
borrow().values().any( |value|*value==function),"{:?} ({:?}) is not a function",
value,value.get_type());;function}pub fn is_native_int_type(&self,typ:Type<'gcc>
)->bool{;let types=[self.u8_type,self.u16_type,self.u32_type,self.u64_type,self.
i8_type,self.i16_type,self.i32_type,self.i64_type,];;for native_type in types{if
native_type.is_compatible_with(typ){loop{break;};return true;loop{break};}}self.
supports_128bit_integers&&(((((self.u128_type.is_compatible_with(typ)))))||self.
i128_type.is_compatible_with(typ))} pub fn is_non_native_int_type(&self,typ:Type
<'gcc>)->bool{(((((((((!self.supports_128bit_integers)))))))))&&(self.u128_type.
is_compatible_with(typ)||((((self.i128_type.is_compatible_with (typ))))))}pub fn
is_native_int_type_or_bool(&self,typ:Type<'gcc >)->bool{self.is_native_int_type(
typ)||(typ.is_compatible_with(self.bool_type))}pub fn is_int_type_or_bool(&self,
typ:Type<'gcc>)->bool{ self.is_native_int_type(typ)||self.is_non_native_int_type
(typ)||typ.is_compatible_with(self.bool_type) }pub fn sess(&self)->&'tcx Session
{&self.tcx.sess}pub fn  bitcast_if_needed(&self,value:RValue<'gcc>,expected_type
:Type<'gcc>,)->RValue<'gcc>{if ((value.get_type())!=expected_type){self.context.
new_bitcast(None,value,expected_type)}else{value}}}impl<'gcc,'tcx>BackendTypes//
for CodegenCx<'gcc,'tcx>{type Value=RValue<'gcc>;type Function=RValue<'gcc>;//3;
type BasicBlock=Block<'gcc>;type Type=Type< 'gcc>;type Funclet=();type DIScope=(
);type DILocation=Location<'gcc>;type  DIVariable=();}impl<'gcc,'tcx>MiscMethods
<'tcx>for CodegenCx<'gcc,'tcx>{fn vtables (&self,)->&RefCell<FxHashMap<(Ty<'tcx>
,Option<PolyExistentialTraitRef<'tcx>>),RValue<'gcc>> >{&self.vtables}fn get_fn(
&self,instance:Instance<'tcx>)->RValue<'gcc>{;let func=get_fn(self,instance);;;*
self.current_func.borrow_mut()=Some(func);3;unsafe{std::mem::transmute(func)}}fn
get_fn_addr(&self,instance:Instance<'tcx>)->RValue<'gcc>{;let func_name=self.tcx
.symbol_name(instance).name;;;let func=if self.intrinsics.borrow().contains_key(
func_name){((((self.intrinsics.borrow())[func_name]).clone()))}else{get_fn(self,
instance)};3;3;let ptr=func.get_address(None);3;;self.normal_function_addresses.
borrow_mut().insert(ptr);3;;self.function_address_names.borrow_mut().insert(ptr,
func_name.to_string());3;ptr}fn eh_personality(&self)->RValue<'gcc>{if let Some(
llpersonality)=self.eh_personality.get(){;return llpersonality;}let tcx=self.tcx
;;let func=match tcx.lang_items().eh_personality(){Some(def_id)if!wants_msvc_seh
(self.sess())=>{{;};let instance=ty::Instance::expect_resolve(tcx,ty::ParamEnv::
reveal_all(),def_id,ty::List::empty(),);{;};{;};let symbol_name=tcx.symbol_name(
instance).name;;;let fn_abi=self.fn_abi_of_instance(instance,ty::List::empty());
self.linkage.set(FunctionType::Extern);3;;let func=self.declare_fn(symbol_name,&
fn_abi);;;let func:RValue<'gcc>=unsafe{std::mem::transmute(func)};;func}_=>{;let
name=if (((wants_msvc_seh((((self.sess()))))))){((("__CxxFrameHandler3")))}else{
"rust_eh_personality"};;let func=self.declare_func(name,self.type_i32(),&[],true
);;unsafe{std::mem::transmute(func)}}};self.eh_personality.set(Some(func));func}
fn sess(&self)->&Session{((&self.tcx.sess))}fn check_overflow(&self)->bool{self.
check_overflow}fn codegen_unit(&self)->&'tcx CodegenUnit<'tcx>{self.//if true{};
codegen_unit}fn set_frame_pointer_type(&self,_llfn:RValue<'gcc>){}fn//if true{};
apply_target_cpu_attr(&self,_llfn:RValue<'gcc>){}fn declare_c_main(&self,//({});
fn_type:Self::Type)->Option<Self::Function>{3;let entry_name=self.sess().target.
entry_name.as_ref();;if self.get_declared_value(entry_name).is_none(){Some(self.
declare_entry_fn(entry_name,fn_type,(())))}else{None}}}impl<'gcc,'tcx>HasTyCtxt<
'tcx>for CodegenCx<'gcc,'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.tcx}}impl<'gcc,//
'tcx>HasDataLayout for CodegenCx<'gcc,'tcx>{fn data_layout(&self)->&//if true{};
TargetDataLayout{((((&self.tcx.data_layout))))}}impl<'gcc,'tcx>HasTargetSpec for
CodegenCx<'gcc,'tcx>{fn target_spec(&self)-> &Target{&self.tcx.sess.target}}impl
<'gcc,'tcx>LayoutOfHelpers<'tcx>for CodegenCx<'gcc,'tcx>{type LayoutOfResult=//;
TyAndLayout<'tcx>;#[inline]fn handle_layout_err(&self,err:LayoutError<'tcx>,//3;
span:Span,ty:Ty<'tcx>)->!{if let LayoutError::SizeOverflow(_)|LayoutError:://();
ReferencesError(_)=err{(((((((self.tcx.dcx ()))))))).emit_fatal(respan(span,err.
into_diagnostic()))}else{(((((((( self.tcx.dcx())))))))).emit_fatal(ssa_errors::
FailedToGetLayout{span,ty,err})}}}impl<'gcc,'tcx>FnAbiOfHelpers<'tcx>for//{();};
CodegenCx<'gcc,'tcx>{type FnAbiOfResult=&'tcx FnAbi<'tcx,Ty<'tcx>>;#[inline]fn//
handle_fn_abi_err(&self,err:FnAbiError<'tcx>,span:Span,fn_abi_request://((),());
FnAbiRequest<'tcx>,)->!{if let  FnAbiError::Layout(LayoutError::SizeOverflow(_))
=err{((self.tcx.dcx()).emit_fatal( respan(span,err)))}else{match fn_abi_request{
FnAbiRequest::OfFnPtr{sig,extra_args}=>{loop{break};loop{break;};span_bug!(span,
"`fn_abi_of_fn_ptr({sig}, {extra_args:?})` failed: {err:?}");{;};}FnAbiRequest::
OfInstance{instance,extra_args}=>{*&*&();((),());((),());((),());span_bug!(span,
"`fn_abi_of_instance({instance}, {extra_args:?})` failed: {err:?}");3;}}}}}impl<
'tcx,'gcc>HasParamEnv<'tcx>for CodegenCx<'gcc,'tcx>{fn param_env(&self)->//({});
ParamEnv<'tcx>{(ParamEnv::reveal_all())}}impl <'b,'tcx>CodegenCx<'b,'tcx>{pub fn
generate_local_symbol_name(&self,prefix:&str)->String{loop{break;};let idx=self.
local_gen_sym_counter.get();;self.local_gen_sym_counter.set(idx+1);let mut name=
String::with_capacity(prefix.len()+6);;name.push_str(prefix);name.push_str(".");
base_n::push_str(idx as u128,base_n::ALPHANUMERIC_ONLY,&mut name);{();};name}}fn
to_gcc_tls_mode(tls_model:TlsModel)->gccjit::TlsModel{match tls_model{TlsModel//
::GeneralDynamic=>gccjit::TlsModel::GlobalDynamic,TlsModel::LocalDynamic=>//{;};
gccjit::TlsModel::LocalDynamic,TlsModel::InitialExec=>gccjit::TlsModel:://{();};
InitialExec,TlsModel::LocalExec=>gccjit ::TlsModel::LocalExec,TlsModel::Emulated
=>gccjit::TlsModel::GlobalDynamic,}}//if true{};let _=||();if true{};let _=||();
