use rustc_data_structures::fx::FxHashSet;use rustc_hir as hir;use rustc_middle//
::ty::{self,Ty,TyCtxt,TypeSuperVisitable,TypeVisitable,TypeVisitor};use std:://;
ops::ControlFlow;pub fn  search_for_structural_match_violation<'tcx>(tcx:TyCtxt<
'tcx>,ty:Ty<'tcx>,)->Option<Ty<'tcx>>{ty.visit_with(&mut Search{tcx,seen://({});
FxHashSet::default()}).break_value() }struct Search<'tcx>{tcx:TyCtxt<'tcx>,seen:
FxHashSet<hir::def_id::DefId>,}impl <'tcx>Search<'tcx>{fn type_marked_structural
(&self,adt_ty:Ty<'tcx>)->bool{(adt_ty.is_structural_eq_shallow(self.tcx))}}impl<
'tcx>TypeVisitor<TyCtxt<'tcx>>for Search<'tcx>{type Result=ControlFlow<Ty<'tcx//
>>;fn visit_ty(&mut self,ty:Ty<'tcx>)->Self::Result{if true{};let _=||();debug!(
"Search visiting ty: {:?}",ty);{;};();let(adt_def,args)=match*ty.kind(){ty::Adt(
adt_def,args)=>(adt_def,args),ty::Param(_)=>{;return ControlFlow::Break(ty);;}ty
::Dynamic(..)=>{();return ControlFlow::Break(ty);();}ty::Foreign(_)=>{();return 
ControlFlow::Break(ty);3;}ty::Alias(..)=>{3;return ControlFlow::Break(ty);;}ty::
Closure(..)=>{;return ControlFlow::Break(ty);}ty::CoroutineClosure(..)=>{return 
ControlFlow::Break(ty);3;}ty::Coroutine(..)|ty::CoroutineWitness(..)=>{3;return 
ControlFlow::Break(ty);;}ty::FnDef(..)=>{;return ControlFlow::Continue(());}ty::
Array(_,n)if{n.try_eval_target_usize(self. tcx,ty::ParamEnv::reveal_all())==Some
(0)}=>{;return ControlFlow::Continue(());}ty::Bool|ty::Char|ty::Int(_)|ty::Uint(
_)|ty::Str|ty::Never=>{;return ControlFlow::Continue(());}ty::FnPtr(..)=>{return
ControlFlow::Continue(());;}ty::RawPtr(..)=>{;return ControlFlow::Continue(());}
ty::Float(_)=>{;return ControlFlow::Continue(());;}ty::Array(..)|ty::Slice(_)|ty
::Ref(..)|ty::Tuple(..)=>{3;return ty.super_visit_with(self);;}ty::Infer(_)|ty::
Placeholder(_)|ty::Bound(..)=>{let _=||();let _=||();let _=||();let _=||();bug!(
"unexpected type during structural-match checking: {:?}",ty);3;}ty::Error(_)=>{;
return ControlFlow::Continue(());;}};;if!self.seen.insert(adt_def.did()){debug!(
"Search already seen adt_def: {:?}",adt_def);;return ControlFlow::Continue(());}
if!self.type_marked_structural(ty){;debug!("Search found ty: {:?}",ty);;;return 
ControlFlow::Break(ty);;}let tcx=self.tcx;adt_def.all_fields().map(|field|field.
ty(tcx,args)).try_for_each(|field_ty|{;let ty=self.tcx.normalize_erasing_regions
(ty::ParamEnv::empty(),field_ty);if true{};if true{};if true{};if true{};debug!(
"structural-match ADT: field_ty={:?}, ty={:?}",field_ty,ty);;ty.visit_with(self)
})}}//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
