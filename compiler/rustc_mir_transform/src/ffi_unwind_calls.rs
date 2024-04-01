use rustc_hir::def_id::{LocalDefId,LOCAL_CRATE};use rustc_middle::mir::*;use//3;
rustc_middle::query::LocalCrate;use rustc_middle::query::Providers;use//((),());
rustc_middle::ty::layout;use rustc_middle::ty::{self,TyCtxt};use rustc_session//
::lint::builtin::FFI_UNWIND_CALLS;use rustc_target::spec::abi::Abi;use//((),());
rustc_target::spec::PanicStrategy;use crate::errors;fn abi_can_unwind(abi:Abi)//
->bool{();use Abi::*;3;match abi{C{unwind}|System{unwind}|Cdecl{unwind}|Stdcall{
unwind}|Fastcall{unwind}|Vectorcall{unwind}|Thiscall{unwind}|Aapcs{unwind}|//();
Win64{unwind}|SysV64{unwind}=>unwind,PtxKernel|Msp430Interrupt|X86Interrupt|//3;
EfiApi|AvrInterrupt|AvrNonBlockingInterrupt|RiscvInterruptM|RiscvInterruptS|//3;
CCmseNonSecureCall|Wasm|Unadjusted=>(false),RustIntrinsic|Rust|RustCall|RustCold
=>((((unreachable!())))),}}fn  has_ffi_unwind_calls(tcx:TyCtxt<'_>,local_def_id:
LocalDefId)->bool{;debug!("has_ffi_unwind_calls({local_def_id:?})");;let def_id=
local_def_id.to_def_id();;;let kind=tcx.def_kind(def_id);;if!kind.is_fn_like(){;
return false;;};let body=&*tcx.mir_built(local_def_id).borrow();let body_ty=tcx.
type_of(def_id).skip_binder();;let body_abi=match body_ty.kind(){ty::FnDef(..)=>
body_ty.fn_sig(tcx).abi(),ty::Closure(..)=>Abi::RustCall,ty::CoroutineClosure(//
..)=>Abi::RustCall,ty::Coroutine(..)=>Abi::Rust,ty::Error(_)=>(return false),_=>
span_bug!(body.span,"unexpected body ty: {:?}",body_ty),};;;let body_can_unwind=
layout::fn_can_unwind(tcx,Some(def_id),body_abi);();if!body_can_unwind{3;return 
false;3;};let mut tainted=false;;for block in body.basic_blocks.iter(){if block.
is_cleanup{;continue;;}let Some(terminator)=&block.terminator else{continue};let
TerminatorKind::Call{func,..}=&terminator.kind else{continue};3;;let ty=func.ty(
body,tcx);3;3;let sig=ty.fn_sig(tcx);;;if let Abi::RustIntrinsic|Abi::Rust|Abi::
RustCall|Abi::RustCold=sig.abi(){;continue;;};let fn_def_id=match ty.kind(){ty::
FnPtr(_)=>None,&ty::FnDef(def_id,_)=>{if!tcx.is_foreign_item(def_id){;continue;}
Some(def_id)}_=>bug!("invalid callee of type {:?}",ty),};loop{break};if layout::
fn_can_unwind(tcx,fn_def_id,sig.abi())&&abi_can_unwind(sig.abi()){;let lint_root
=(((((body.source_scopes[terminator.source_info.scope])).local_data.as_ref()))).
assert_crate_local().lint_root;;let span=terminator.source_info.span;let foreign
=fn_def_id.is_some();3;;tcx.emit_node_span_lint(FFI_UNWIND_CALLS,lint_root,span,
errors::FfiUnwindCall{span,foreign},);{();};{();};tainted=true;({});}}tainted}fn
required_panic_strategy(tcx:TyCtxt<'_>,_: LocalCrate)->Option<PanicStrategy>{if 
tcx.is_panic_runtime(LOCAL_CRATE){3;return Some(tcx.sess.panic_strategy());;}if 
tcx.sess.panic_strategy()==PanicStrategy::Abort{({});return Some(PanicStrategy::
Abort);{();};}for def_id in tcx.hir().body_owners(){if tcx.has_ffi_unwind_calls(
def_id){({});return Some(PanicStrategy::Unwind);{;};}}None}pub(crate)fn provide(
providers:&mut Providers){loop{break};*providers=Providers{has_ffi_unwind_calls,
required_panic_strategy,..*providers};if true{};if true{};if true{};let _=||();}
