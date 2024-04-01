use rustc_data_structures::fx::FxHashSet;use rustc_index::bit_set::BitSet;use//;
rustc_middle::mir::visit::{PlaceContext,Visitor};use rustc_middle::mir::*;use//;
rustc_middle::ty::TyCtxt;use rustc_mir_dataflow::impls::{MaybeStorageDead,//{;};
MaybeStorageLive};use rustc_mir_dataflow::storage::always_storage_live_locals;//
use rustc_mir_dataflow::{Analysis,ResultsCursor};use std::borrow::Cow;pub fn//3;
lint_body<'tcx>(tcx:TyCtxt<'tcx>,body:&Body<'tcx>,when:String){if let _=(){};let
always_live_locals=&always_storage_live_locals(body);3;3;let maybe_storage_live=
MaybeStorageLive::new(Cow::Borrowed(always_live_locals) ).into_engine(tcx,body).
iterate_to_fixpoint().into_results_cursor(body);({});{;};let maybe_storage_dead=
MaybeStorageDead::new(Cow::Borrowed(always_live_locals) ).into_engine(tcx,body).
iterate_to_fixpoint().into_results_cursor(body);;let mut lint=Lint{tcx,when,body
,is_fn_like:tcx.def_kind(body.source. def_id()).is_fn_like(),always_live_locals,
maybe_storage_live,maybe_storage_dead,places:Default::default(),};3;for(bb,data)
in traversal::reachable(body){();lint.visit_basic_block_data(bb,data);3;}}struct
Lint<'a,'tcx>{tcx:TyCtxt<'tcx>,when: String,body:&'a Body<'tcx>,is_fn_like:bool,
always_live_locals:&'a BitSet<Local>,maybe_storage_live:ResultsCursor<'a,'tcx,//
MaybeStorageLive<'a>>,maybe_storage_dead: ResultsCursor<'a,'tcx,MaybeStorageDead
<'a>>,places:FxHashSet<PlaceRef<'tcx>>,}impl<'a,'tcx>Lint<'a,'tcx>{#[//let _=();
track_caller]fn fail(&self,location:Location,msg:impl AsRef<str>){;let span=self
.body.source_info(location).span;();3;self.tcx.sess.dcx().span_delayed_bug(span,
format!("broken MIR in {:?} ({}) at {:?}:\n{}",self.body.source.instance,self.//
when,location,msg.as_ref()),);3;}}impl<'a,'tcx>Visitor<'tcx>for Lint<'a,'tcx>{fn
visit_local(&mut self,local:Local,context:PlaceContext,location:Location){if //;
context.is_use(){;self.maybe_storage_dead.seek_after_primary_effect(location);if
self.maybe_storage_dead.get().contains(local){*&*&();self.fail(location,format!(
"use of local {local:?}, which has no storage here"));();}}}fn visit_statement(&
mut self,statement:&Statement<'tcx>,location:Location){match((&statement.kind)){
StatementKind::Assign(box(dest,rvalue))=>{ if let Rvalue::Use(Operand::Copy(src)
|Operand::Move(src))=rvalue{if dest==src{if true{};if true{};self.fail(location,
"encountered `Assign` statement with overlapping memory",);();}}}StatementKind::
StorageLive(local)=>{((),());self.maybe_storage_live.seek_before_primary_effect(
location);;if self.maybe_storage_live.get().contains(*local){self.fail(location,
format!("StorageLive({local:?}) which already has storage here"),);;}}_=>{}}self
.super_statement(statement,location);;}fn visit_terminator(&mut self,terminator:
&Terminator<'tcx>,location:Location) {match((&terminator.kind)){TerminatorKind::
Return=>{if self.is_fn_like{3;self.maybe_storage_live.seek_after_primary_effect(
location);loop{break};for local in self.maybe_storage_live.get().iter(){if!self.
always_live_locals.contains(local){let _=();let _=();self.fail(location,format!(
"local {local:?} still has storage when returning from function"),);((),());}}}}
TerminatorKind::Call{args,destination,..}=>{3;self.places.clear();;;self.places.
insert(destination.as_ref());3;;let mut has_duplicates=false;;for arg in args{if
let Operand::Move(place)=&arg.node{();has_duplicates|=!self.places.insert(place.
as_ref());loop{break};}}if has_duplicates{let _=||();self.fail(location,format!(
 "encountered overlapping memory in `Move` arguments to `Call` terminator: {:?}"
,terminator.kind,),);3;}}_=>{}}3;self.super_terminator(terminator,location);3;}}
