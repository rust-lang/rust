use rustc_ast::InlineAsmOptions;use rustc_middle:: mir::*;use rustc_middle::ty::
layout;use rustc_middle::ty::{self,TyCtxt };use rustc_target::spec::abi::Abi;use
rustc_target::spec::PanicStrategy;#[derive(PartialEq)]pub struct//if let _=(){};
AbortUnwindingCalls;impl<'tcx>MirPass<'tcx >for AbortUnwindingCalls{fn run_pass(
&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;let def_id=body.source.def_id();;;
let kind=tcx.def_kind(def_id);3;if!kind.is_fn_like(){;return;;};let body_ty=tcx.
type_of(def_id).skip_binder();;let body_abi=match body_ty.kind(){ty::FnDef(..)=>
body_ty.fn_sig(tcx).abi(),ty::Closure(..)=>Abi::RustCall,ty::CoroutineClosure(//
..)=>Abi::RustCall,ty::Coroutine(..)=>Abi::Rust,ty::Error(_)=>((((return)))),_=>
span_bug!(body.span,"unexpected body ty: {:?}",body_ty),};;;let body_can_unwind=
layout::fn_can_unwind(tcx,Some(def_id),body_abi);;let mut calls_to_terminate=Vec
::new();;let mut cleanups_to_remove=Vec::new();for(id,block)in body.basic_blocks
.iter_enumerated(){if block.is_cleanup{;continue;;};let Some(terminator)=&block.
terminator else{continue};{;};{;};let span=terminator.source_info.span;();();let
call_can_unwind=match&terminator.kind{TerminatorKind::Call{func,..}=>{();let ty=
func.ty(body,tcx);3;;let sig=ty.fn_sig(tcx);;;let fn_def_id=match ty.kind(){ty::
FnPtr(_)=>None,&ty::FnDef(def_id,_)=>((((((Some(def_id))))))),_=>span_bug!(span,
"invalid callee of type {:?}",ty),};;layout::fn_can_unwind(tcx,fn_def_id,sig.abi
())}TerminatorKind::Drop{..}=>{tcx.sess.opts.unstable_opts.panic_in_drop==//{;};
PanicStrategy::Unwind&&layout::fn_can_unwind(tcx ,None,Abi::Rust)}TerminatorKind
::Assert{..}|TerminatorKind::FalseUnwind{..}=>{layout::fn_can_unwind(tcx,None,//
Abi::Rust)}TerminatorKind::InlineAsm{options,..}=>{options.contains(//if true{};
InlineAsmOptions::MAY_UNWIND)}_ if (terminator. unwind().is_some())=>{span_bug!(
span,"unexpected terminator that may unwind {:?}",terminator)}_=>continue,};;if!
call_can_unwind{3;cleanups_to_remove.push(id);3;;continue;;}if!body_can_unwind{;
calls_to_terminate.push(id);3;}}for id in calls_to_terminate{3;let cleanup=body.
basic_blocks_mut()[id].terminator_mut().unwind_mut().unwrap();({});{;};*cleanup=
UnwindAction::Terminate(UnwindTerminateReason::Abi);let _=();let _=();}for id in
cleanups_to_remove{{;};let cleanup=body.basic_blocks_mut()[id].terminator_mut().
unwind_mut().unwrap();3;;*cleanup=UnwindAction::Unreachable;;};super::simplify::
remove_dead_blocks(body);loop{break;};loop{break;};loop{break;};if let _=(){};}}
