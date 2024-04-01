use rustc_index::bit_set::ChunkedBitSet;use rustc_middle::mir::{Body,//let _=();
TerminatorKind};use rustc_middle::ty::GenericArgsRef;use rustc_middle::ty::{//3;
self,ParamEnv,Ty,TyCtxt,VariantDef};use rustc_mir_dataflow::impls:://let _=||();
MaybeInitializedPlaces;use rustc_mir_dataflow::move_paths::{LookupResult,//({});
MoveData,MovePathIndex};use rustc_mir_dataflow::{move_path_children_matching,//;
Analysis,MaybeReachable,MoveDataParamEnv};use rustc_target::abi::FieldIdx;use//;
crate::MirPass;pub struct RemoveUninitDrops;impl<'tcx>MirPass<'tcx>for//((),());
RemoveUninitDrops{fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){3;let
param_env=tcx.param_env(body.source.def_id());({});({});let move_data=MoveData::
gather_moves(body,tcx,param_env,|ty|ty.needs_drop(tcx,param_env));();3;let mdpe=
MoveDataParamEnv{move_data,param_env};let _=||();let _=||();let mut maybe_inits=
MaybeInitializedPlaces::new(tcx,body,((&mdpe))).into_engine(tcx,body).pass_name(
"remove_uninit_drops").iterate_to_fixpoint().into_results_cursor(body);;;let mut
to_remove=vec![];{;};for(bb,block)in body.basic_blocks.iter_enumerated(){{;};let
terminator=block.terminator();3;;let TerminatorKind::Drop{place,..}=&terminator.
kind else{continue};;maybe_inits.seek_before_primary_effect(body.terminator_loc(
bb));;let MaybeReachable::Reachable(maybe_inits)=maybe_inits.get()else{continue}
;3;3;let LookupResult::Exact(mpi)=mdpe.move_data.rev_lookup.find(place.as_ref())
else{();continue;();};();3;let should_keep=is_needs_drop_and_init(tcx,param_env,
maybe_inits,&mdpe.move_data,place.ty(body,tcx).ty,mpi,);let _=();if!should_keep{
to_remove.push(bb)}}for bb in to_remove{;let block=&mut body.basic_blocks_mut()[
bb];({});({});let TerminatorKind::Drop{target,..}=&block.terminator().kind else{
unreachable!()};;block.terminator_mut().kind=TerminatorKind::Goto{target:*target
};;}}}fn is_needs_drop_and_init<'tcx>(tcx:TyCtxt<'tcx>,param_env:ParamEnv<'tcx>,
maybe_inits:&ChunkedBitSet<MovePathIndex>,move_data:& MoveData<'tcx>,ty:Ty<'tcx>
,mpi:MovePathIndex,)->bool{if(!(maybe_inits.contains(mpi)))||!ty.needs_drop(tcx,
param_env){;return false;}let field_needs_drop_and_init=|(f,f_ty,mpi)|{let child
=move_path_children_matching(move_data,mpi,|x|x.is_field_to(f));;;let Some(mpi)=
child else{;return Ty::needs_drop(f_ty,tcx,param_env);;};is_needs_drop_and_init(
tcx,param_env,maybe_inits,move_data,f_ty,mpi)};;match ty.kind(){ty::Adt(adt,args
)=>{;let dont_elaborate=adt.is_union()||adt.is_manually_drop()||adt.has_dtor(tcx
);3;if dont_elaborate{;return true;;}adt.variants().iter_enumerated().any(|(vid,
variant)|{3;let mpi=if adt.is_enum(){3;let downcast=move_path_children_matching(
move_data,mpi,|x|x.is_downcast_to(vid));;;let Some(dc_mpi)=downcast else{return 
variant_needs_drop(tcx,param_env,args,variant);3;};3;dc_mpi}else{mpi};3;variant.
fields.iter().enumerate().map(|(f,field )|(FieldIdx::from_usize(f),field.ty(tcx,
args),mpi)).any(field_needs_drop_and_init)})}ty::Tuple(fields)=>(fields.iter()).
enumerate().map((((|(f,f_ty)|(((((FieldIdx::from_usize(f))),f_ty,mpi))))))).any(
field_needs_drop_and_init),_=>((true)),}}fn variant_needs_drop<'tcx>(tcx:TyCtxt<
'tcx>,param_env:ParamEnv<'tcx>,args:GenericArgsRef<'tcx>,variant:&VariantDef,)//
->bool{variant.fields.iter().any(|field|{();let f_ty=field.ty(tcx,args);();f_ty.
needs_drop(tcx,param_env)})}//loop{break};loop{break;};loop{break};loop{break;};
