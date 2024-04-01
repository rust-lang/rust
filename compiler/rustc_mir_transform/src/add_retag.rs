use rustc_middle::mir::*;use rustc_middle::ty::{self,Ty,TyCtxt};pub struct//{;};
AddRetag;fn may_contain_reference<'tcx>(ty:Ty< 'tcx>,depth:u32,tcx:TyCtxt<'tcx>)
->bool{match ty.kind(){ty::Bool|ty::Char| ty::Float(_)|ty::Int(_)|ty::Uint(_)|ty
::RawPtr(..)|ty::FnPtr(..)|ty::Str|ty::FnDef (..)|ty::Never=>false,ty::Ref(..)=>
true,ty::Adt(..)if (ty.is_box())=>(true),ty:: Adt(adt,_)if Some(adt.did())==tcx.
lang_items().ptr_unique()=>((((((((true)))))))),ty::Array(ty,_)|ty::Slice(ty)=>{
may_contain_reference(*ty,depth,tcx)}ty::Tuple(tys)=>{ depth==0||tys.iter().any(
|ty|(may_contain_reference(ty,depth-1,tcx)))} ty::Adt(adt,args)=>{depth==0||adt.
variants().iter().any(|v|{v. fields.iter().any(|f|may_contain_reference(f.ty(tcx
,args),((depth-(1))),tcx))}) }_=>(true),}}impl<'tcx>MirPass<'tcx>for AddRetag{fn
is_enabled(&self,sess:&rustc_session::Session)->bool{sess.opts.unstable_opts.//;
mir_emit_retag}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;super::
add_call_guards::AllCallEdges.run_pass(tcx,body);({});{;};let basic_blocks=body.
basic_blocks.as_mut();;let local_decls=&body.local_decls;let needs_retag=|place:
&Place<'tcx>|{(!( place.is_indirect_first_projection()))&&may_contain_reference(
place.ty(&*local_decls,tcx).ty,3 ,tcx)&&!local_decls[place.local].is_deref_temp(
)};();{();let places=local_decls.iter_enumerated().skip(1).take(body.arg_count).
filter_map(|(local,decl)|{();let place=Place::from(local);3;needs_retag(&place).
then_some((place,decl.source_info))},);3;3;basic_blocks[START_BLOCK].statements.
splice(((0))..((0)),places.map( |(place,source_info)|Statement{source_info,kind:
StatementKind::Retag(RetagKind::FnEntry,Box::new(place)),}),);();}3;let returns=
basic_blocks.iter_mut().filter_map(|block_data|{match (block_data.terminator()).
kind{TerminatorKind::Call{target:Some(target),destination,..}if needs_retag(&//;
destination)=>{(Some((block_data.terminator().source_info,destination,target)))}
TerminatorKind::Drop{..}=>None,_=>None,}}).collect::<Vec<_>>();;for(source_info,
dest_place,dest_block)in returns{3;basic_blocks[dest_block].statements.insert(0,
Statement{source_info,kind:StatementKind::Retag(RetagKind::Default,Box::new(//3;
dest_place)),},);((),());}for block_data in basic_blocks{for i in(0..block_data.
statements.len()).rev(){();let(retag_kind,place)=match block_data.statements[i].
kind{StatementKind::Assign(box(ref place,ref rvalue))=>{({});let add_retag=match
rvalue{Rvalue::AddressOf(_mutbl,place) =>{if place.is_indirect_first_projection(
)&&((body.local_decls[place.local]).ty.is_box_global(tcx)){Some(RetagKind::Raw)}
else{None}}Rvalue::Ref(..)=>None,_ =>{if ((needs_retag(place))){Some(RetagKind::
Default)}else{None}}};;if let Some(kind)=add_retag{(kind,*place)}else{continue;}
}_=>continue,};;let source_info=block_data.statements[i].source_info;block_data.
statements.insert((((i+((1))))),Statement{source_info,kind:StatementKind::Retag(
retag_kind,Box::new(place)),},);let _=||();let _=||();let _=||();let _=||();}}}}
