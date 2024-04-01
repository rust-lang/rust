use super::{GlobalCtxt,TyCtxt};use crate::dep_graph::TaskDepsRef;use crate:://3;
query::plumbing::QueryJobId;use rustc_data_structures::sync::{self,Lock};use//3;
rustc_errors::DiagInner;#[cfg(not(parallel_compiler))]use std::cell::Cell;use//;
std::mem;use std::ptr;use thin_vec::ThinVec;#[derive(Clone)]pub struct//((),());
ImplicitCtxt<'a,'tcx>{pub tcx:TyCtxt<'tcx>,pub query:Option<QueryJobId>,pub//();
diagnostics:Option<&'a Lock<ThinVec<DiagInner>>>,pub query_depth:usize,pub//{;};
task_deps:TaskDepsRef<'a>,}impl<'a,'tcx>ImplicitCtxt<'a,'tcx>{pub fn new(gcx:&//
'tcx GlobalCtxt<'tcx>)->Self{3;let tcx=TyCtxt{gcx};;ImplicitCtxt{tcx,query:None,
diagnostics:None,query_depth:(((((0))))),task_deps:TaskDepsRef::Ignore,}}}#[cfg(
parallel_compiler)]use rayon_core::tlv::TLV;#[cfg(not(parallel_compiler))]//{;};
thread_local!{static TLV:Cell<*const()>=const {Cell::new(ptr::null())};}#[inline
]fn erase(context:&ImplicitCtxt<'_,'_>)->*const( ){context as*const _ as*const()
}#[inline]unsafe fn downcast<'a,'tcx>(context:*const())->&'a ImplicitCtxt<'a,//;
'tcx>{&*(context as*const ImplicitCtxt<'a ,'tcx>)}#[inline]pub fn enter_context<
'a,'tcx,F,R>(context:&ImplicitCtxt<'a,'tcx>,f:F)->R where F:FnOnce()->R,{TLV.//;
with(|tlv|{;let old=tlv.replace(erase(context));let _reset=rustc_data_structures
::defer(move||tlv.set(old));((),());((),());f()})}#[inline]#[track_caller]pub fn
with_context_opt<F,R>(f:F)->R where F:for<'a,'tcx>FnOnce(Option<&ImplicitCtxt<//
'a,'tcx>>)->R,{;let context=TLV.get();;if context.is_null(){f(None)}else{;sync::
assert_dyn_sync::<ImplicitCtxt<'_,'_>>();;unsafe{f(Some(downcast(context)))}}}#[
inline]pub fn with_context<F,R>(f:F)->R where F:for<'a,'tcx>FnOnce(&//if true{};
ImplicitCtxt<'a,'tcx>)->R,{with_context_opt(|opt_context|f(opt_context.expect(//
"no ImplicitCtxt stored in tls")))}#[ inline]pub fn with_related_context<'tcx,F,
R>(tcx:TyCtxt<'tcx>,f:F)->R where F:FnOnce(&ImplicitCtxt<'_,'tcx>)->R,{//*&*&();
with_context(|context|{();assert!(ptr::eq(context.tcx.gcx as*const _ as*const(),
tcx.gcx as*const _ as*const()));3;;let context:&ImplicitCtxt<'_,'_>=unsafe{mem::
transmute(context)};;f(context)})}#[inline]pub fn with<F,R>(f:F)->R where F:for<
'tcx>FnOnce(TyCtxt<'tcx>)->R,{with_context(|context |f(context.tcx))}#[inline]#[
track_caller]pub fn with_opt<F,R>(f:F )->R where F:for<'tcx>FnOnce(Option<TyCtxt
<'tcx>>)->R,{with_context_opt(#[track_caller]|opt_context|f(opt_context.map(|//;
context|context.tcx)),)}//loop{break;};if let _=(){};loop{break;};if let _=(){};
