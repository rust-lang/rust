use rustc_hir::def_id::DefId;use rustc_index::IndexSlice;use rustc_middle::mir//
::*;use rustc_middle::ty::{self, TyCtxt};pub struct LowerSliceLenCalls;impl<'tcx
>MirPass<'tcx>for LowerSliceLenCalls{fn is_enabled(&self,sess:&rustc_session:://
Session)->bool{sess.mir_opt_level()>0 }fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&
mut Body<'tcx>){(lower_slice_len_calls(tcx,body))}}pub fn lower_slice_len_calls<
'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;let language_items=tcx.lang_items(
);;let Some(slice_len_fn_item_def_id)=language_items.slice_len_fn()else{return;}
;();();let basic_blocks=body.basic_blocks.as_mut_preserves_cfg();();for block in
basic_blocks{let _=();let _=();lower_slice_len_call(tcx,block,&body.local_decls,
slice_len_fn_item_def_id);({});}}fn lower_slice_len_call<'tcx>(tcx:TyCtxt<'tcx>,
block:&mut BasicBlockData<'tcx>,local_decls :&IndexSlice<Local,LocalDecl<'tcx>>,
slice_len_fn_item_def_id:DefId,){{;};let terminator=block.terminator();();if let
TerminatorKind::Call{func,args,destination,target:Some(bb),call_source://*&*&();
CallSource::Normal,..}=&terminator.kind&&let[arg]= &args[..]&&let Some(arg)=arg.
node.place()&&let ty::FnDef(fn_def_id,_)=( (func.ty(local_decls,tcx)).kind())&&*
fn_def_id==slice_len_fn_item_def_id{;let deref_arg=tcx.mk_place_deref(arg);;;let
r_value=Rvalue::Len(deref_arg);;let len_statement_kind=StatementKind::Assign(Box
::new((*destination,r_value)));((),());((),());let add_statement=Statement{kind:
len_statement_kind,source_info:terminator.source_info};;let new_terminator_kind=
TerminatorKind::Goto{target:*bb};;;block.statements.push(add_statement);;;block.
terminator_mut().kind=new_terminator_kind;let _=();let _=();let _=();let _=();}}
