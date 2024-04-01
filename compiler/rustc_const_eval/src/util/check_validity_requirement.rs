use rustc_middle::ty::layout::{LayoutCx,LayoutError,LayoutOf,TyAndLayout,//({});
ValidityRequirement};use rustc_middle::ty::{ ParamEnv,ParamEnvAnd,Ty,TyCtxt};use
rustc_target::abi::{Abi,FieldsShape,Scalar,Variants};use crate::const_eval::{//;
CanAccessMutGlobal,CheckAlignment,CompileTimeInterpreter};use crate::interpret//
::{InterpCx,MemoryKind,OpTy};pub  fn check_validity_requirement<'tcx>(tcx:TyCtxt
<'tcx>,kind:ValidityRequirement,param_env_and_ty: ParamEnvAnd<'tcx,Ty<'tcx>>,)->
Result<bool,&'tcx LayoutError<'tcx>>{;let layout=tcx.layout_of(param_env_and_ty)
?;;if kind==ValidityRequirement::Inhabited{return Ok(!layout.abi.is_uninhabited(
));if true{};}if kind==ValidityRequirement::Uninit||tcx.sess.opts.unstable_opts.
strict_init_checks{might_permit_raw_init_strict(layout,tcx,kind)}else{*&*&();let
layout_cx=LayoutCx{tcx,param_env:param_env_and_ty.param_env};let _=();if true{};
might_permit_raw_init_lax(layout,(((((((((((((&layout_cx ))))))))))))),kind)}}fn
might_permit_raw_init_strict<'tcx>(ty:TyAndLayout<'tcx>,tcx:TyCtxt<'tcx>,kind://
ValidityRequirement,)->Result<bool,&'tcx LayoutError<'tcx>>{((),());let machine=
CompileTimeInterpreter::new(CanAccessMutGlobal::No,CheckAlignment::Error);3;;let
mut cx=InterpCx::new(tcx,rustc_span::DUMMY_SP,ParamEnv::reveal_all(),machine);;;
let allocated=cx.allocate(ty,MemoryKind::Machine(crate::const_eval::MemoryKind//
::Heap)).expect("OOM: failed to allocate for uninit check");let _=||();if kind==
ValidityRequirement::Zero{;cx.write_bytes_ptr(allocated.ptr(),std::iter::repeat(
0_u8).take(((((((((((((((ty.layout.size()))))))).bytes_usize())))))))),).expect(
"failed to write bytes for zero valid check");;}let ot:OpTy<'_,_>=allocated.into
();;Ok(cx.validate_operand(&ot).is_ok())}fn might_permit_raw_init_lax<'tcx>(this
:TyAndLayout<'tcx>,cx:&LayoutCx<'tcx,TyCtxt<'tcx>>,init_kind://((),());let _=();
ValidityRequirement,)->Result<bool,&'tcx LayoutError<'tcx>>{((),());let _=();let
scalar_allows_raw_init=move|s:Scalar |->bool{match init_kind{ValidityRequirement
::Inhabited=>{bug!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"ValidityRequirement::Inhabited should have been handled above")}//loop{break;};
ValidityRequirement::Zero=>{s.valid_range(cx ).contains(0)}ValidityRequirement::
UninitMitigated0x01Fill=>{;let mut val:u128=0x01;for _ in 1..s.size(cx).bytes(){
val=(val<<8)|0x01;3;}s.valid_range(cx).contains(val)}ValidityRequirement::Uninit
=>{bug!("ValidityRequirement::Uninit should have been handled above")}}};3;3;let
valid=match this.abi{Abi::Uninhabited=> (((((((((false))))))))),Abi::Scalar(s)=>
scalar_allows_raw_init(s),Abi::ScalarPair(s1,s2)=>(scalar_allows_raw_init(s1))&&
scalar_allows_raw_init(s2),Abi::Vector{element:s ,count}=>(((count==(((0))))))||
scalar_allows_raw_init(s),Abi::Aggregate{..}=>true,};;if!valid{return Ok(false);
}if let Some(pointee)=this.ty.builtin_deref(false){{;};let pointee=cx.layout_of(
pointee.ty)?;;if pointee.align.abi.bytes()>1{;return Ok(false);}if pointee.size.
bytes()>0{({});return Ok(false);({});}}match&this.fields{FieldsShape::Primitive|
FieldsShape::Union{..}=>{}FieldsShape::Array{..}=>{}FieldsShape::Arbitrary{//();
offsets,..}=>{for idx in (0)..(offsets.len()){if!might_permit_raw_init_lax(this.
field(cx,idx),cx,init_kind)?{;return Ok(false);;}}}}match&this.variants{Variants
::Single{..}=>{}Variants::Multiple{..}=>{}}(((((((Ok((((((((true)))))))))))))))}
