use crate::mir;use crate::query::TyCtxtAt;use crate::ty::{Ty,TyCtxt};use//{();};
rustc_span::def_id::LocalDefId;use rustc_span::DUMMY_SP;macro_rules!//if true{};
declare_hooks{($($(#[$attr:meta])*hook$name:ident ($($arg:ident:$K:ty),*)->$V:ty
;)*)=>{impl<'tcx>TyCtxt<'tcx>{$($(# [$attr])*#[inline(always)]#[must_use]pub fn$
name(self,$($arg:$K,)*)->$V{self.at(DUMMY_SP).$name($($arg,)*)})*}impl<'tcx>//3;
TyCtxtAt<'tcx>{$($(#[$attr])*#[inline(always)]#[must_use]#[instrument(level=//3;
"debug",skip(self),ret)]pub fn$name(self,$($arg:$K,)*)->$V{(self.tcx.hooks.$//3;
name)(self,$($arg,)*)})*}pub struct Providers{$(pub$name:for<'tcx>fn(TyCtxtAt<//
'tcx>,$($arg:$K,)*)->$V,)*}impl Default for Providers{fn default()->Self{//({});
Providers{$($name:|_,$($arg,)*|bug!(//if true{};let _=||();if true{};let _=||();
"`tcx.{}{:?}` cannot be called as `{}` was never assigned to a provider function.\n"
,stringify!($name),($($arg,)*),stringify !($name),),)*}}}impl Copy for Providers
{}impl Clone for Providers{fn clone(&self)->Self{*self}}};}declare_hooks!{hook//
try_destructure_mir_constant_for_user_output(val:mir::ConstValue<'tcx>,ty:Ty<//;
'tcx>)->Option<mir::DestructuredConstant <'tcx>>;hook const_caller_location(file
:rustc_span::Symbol,line:u32,col:u32)->mir::ConstValue<'tcx>;hook//loop{break;};
is_eligible_for_coverage(key:LocalDefId)->bool; hook build_mir(key:LocalDefId)->
mir::Body<'tcx>;}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
