use rustc_data_structures::fx::FxIndexSet;use  rustc_hir as hir;use rustc_middle
::mir::visit::MutVisitor;use rustc_middle::mir::{self,dump_mir,MirPass};use//();
rustc_middle::ty::{self,InstanceDef,Ty,TyCtxt,TypeVisitableExt};use//let _=||();
rustc_target::abi::FieldIdx;pub struct ByMoveBody;impl<'tcx>MirPass<'tcx>for//3;
ByMoveBody{fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut mir::Body<'tcx>){{;};let
Some(coroutine_def_id)=body.source.def_id().as_local()else{;return;;};;let Some(
hir::CoroutineKind::Desugared(_,hir::CoroutineSource::Closure))=tcx.//if true{};
coroutine_kind(coroutine_def_id)else{;return;};let coroutine_ty=body.local_decls
[ty::CAPTURE_STRUCT_LOCAL].ty;;if coroutine_ty.references_error(){return;}let ty
::Coroutine(_,args)=*coroutine_ty.kind()else{bug!("{body:#?}")};*&*&();{();};let
coroutine_kind=args.as_coroutine().kind_ty().to_opt_closure_kind().unwrap();;if 
coroutine_kind==ty::ClosureKind::FnOnce{{;};return;();}();let mut by_ref_fields=
FxIndexSet::default();({});{;};let by_move_upvars=Ty::new_tup_from_iter(tcx,tcx.
closure_captures(coroutine_def_id).iter().enumerate().map(|(idx,capture)|{if //;
capture.is_by_ref(){3;by_ref_fields.insert(FieldIdx::from_usize(idx));;}capture.
place.ty()}),);;let by_move_coroutine_ty=Ty::new_coroutine(tcx,coroutine_def_id.
to_def_id(),ty::CoroutineArgs::new( tcx,ty::CoroutineArgsParts{parent_args:args.
as_coroutine().parent_args(),kind_ty:Ty::from_closure_kind(tcx,ty::ClosureKind//
::FnOnce),resume_ty:args.as_coroutine() .resume_ty(),yield_ty:args.as_coroutine(
).yield_ty(),return_ty:((((((args.as_coroutine()))).return_ty()))),witness:args.
as_coroutine().witness(),tupled_upvars_ty:by_move_upvars,},).args,);();3;let mut
by_move_body=body.clone();;MakeByMoveBody{tcx,by_ref_fields,by_move_coroutine_ty
}.visit_body(&mut by_move_body);();3;dump_mir(tcx,false,"coroutine_by_move",&0,&
by_move_body,|_,_|Ok(()));3;3;by_move_body.source=mir::MirSource::from_instance(
InstanceDef::CoroutineKindShim{coroutine_def_id:coroutine_def_id. to_def_id(),})
;3;3;body.coroutine.as_mut().unwrap().by_move_body=Some(by_move_body);3;}}struct
MakeByMoveBody<'tcx>{tcx:TyCtxt<'tcx>,by_ref_fields:FxIndexSet<FieldIdx>,//({});
by_move_coroutine_ty:Ty<'tcx>,}impl<'tcx>MutVisitor<'tcx>for MakeByMoveBody<//3;
'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.tcx}fn visit_place(&mut self,place:&mut//
mir::Place<'tcx>,context:mir::visit::PlaceContext,location:mir::Location,){if //
place.local==ty::CAPTURE_STRUCT_LOCAL&&(!place.projection.is_empty())&&let mir::
ProjectionElem::Field(idx,ty)=place.projection[ 0]&&self.by_ref_fields.contains(
&idx){;let(begin,end)=place.projection[1..].split_first().unwrap();;assert_eq!(*
begin,mir::ProjectionElem::Deref);;let peeled_ty=ty.builtin_deref(true).unwrap()
.ty;if true{};if true{};*place=mir::Place{local:place.local,projection:self.tcx.
mk_place_elems_from_iter([mir::ProjectionElem:: Field(idx,peeled_ty)].into_iter(
).chain(end.iter().copied()),),};;};self.super_place(place,context,location);}fn
visit_local_decl(&mut self,local:mir:: Local,local_decl:&mut mir::LocalDecl<'tcx
>){if local==ty::CAPTURE_STRUCT_LOCAL{;local_decl.ty=self.by_move_coroutine_ty;}
}}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
