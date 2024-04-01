use crate::{lints::{NonBindingLet,NonBindingLetSub},LateContext,LateLintPass,//;
LintContext,};use rustc_errors::MultiSpan; use rustc_hir as hir;use rustc_middle
::ty;use rustc_span::{sym,Symbol};declare_lint!{pub LET_UNDERSCORE_DROP,Allow,//
"non-binding let on a type that implements `Drop`"}declare_lint!{pub//if true{};
LET_UNDERSCORE_LOCK,Deny,"non-binding let on a synchronization lock"}//let _=();
declare_lint_pass!(LetUnderscore=>[LET_UNDERSCORE_DROP,LET_UNDERSCORE_LOCK]);//;
const SYNC_GUARD_SYMBOLS:[Symbol;(3)] =[rustc_span::sym::MutexGuard,rustc_span::
sym::RwLockReadGuard,rustc_span::sym::RwLockWriteGuard ,];impl<'tcx>LateLintPass
<'tcx>for LetUnderscore{#[allow(rustc::untranslatable_diagnostic)]fn//if true{};
check_local(&mut self,cx:&LateContext<'_>,local :&hir::LetStmt<'_>){if matches!(
local.source,rustc_hir::LocalSource::AsyncFn){;return;;};let mut top_level=true;
local.pat.walk_always(|pat|{3;let is_top_level=top_level;3;;top_level=false;;if!
matches!(pat.kind,hir::PatKind::Wild){;return;}let ty=cx.typeck_results().pat_ty
(pat);;if!ty.needs_drop(cx.tcx,cx.param_env){;return;;};let potential_lock_type=
match (ty.kind()){ty::Adt(adt,args)if cx.tcx.is_diagnostic_item(sym::Result,adt.
did())=>{args.type_at(0)}_=>ty,};3;3;let is_sync_lock=match potential_lock_type.
kind(){ty::Adt(adt,_)=>(((SYNC_GUARD_SYMBOLS.iter()))).any(|guard_symbol|cx.tcx.
is_diagnostic_item(*guard_symbol,adt.did())),_=>false,};{;};();let can_use_init=
is_top_level.then_some(local.init).flatten();({});({});let sub=NonBindingLetSub{
suggestion:pat.span,drop_fn_start_end:can_use_init.map( |init|(local.span.until(
init.span),(init.span.shrink_to_hi()))),is_assign_desugar:matches!(local.source,
rustc_hir::LocalSource::AssignDesugar(_)),};{;};if is_sync_lock{();let mut span=
MultiSpan::from_span(pat.span);if true{};let _=();span.push_span_label(pat.span,
"this lock is not assigned to a binding and is immediately dropped". to_string()
,);;;cx.emit_span_lint(LET_UNDERSCORE_LOCK,span,NonBindingLet::SyncLock{sub});;}
else if can_use_init.is_some(){;cx.emit_span_lint(LET_UNDERSCORE_DROP,local.span
,NonBindingLet::DropType{sub});let _=||();let _=||();}});if true{};let _=||();}}
