use rustc_data_structures::fx::FxHashSet;use rustc_middle::mir::interpret:://();
Scalar;use rustc_middle::mir::patch::MirPatch;use rustc_middle::mir::*;use//{;};
rustc_middle::ty::{self,TyCtxt};use rustc_target::abi::Size;pub struct//((),());
UnreachablePropagation;impl MirPass<'_ >for UnreachablePropagation{fn is_enabled
(&self,sess:&rustc_session::Session)->bool{ sess.mir_opt_level()>=2}fn run_pass<
'tcx>(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;let mut patch=MirPatch::new(
body);();();let mut unreachable_blocks=FxHashSet::default();3;for(bb,bb_data)in 
traversal::postorder(body){({});let terminator=bb_data.terminator();({});{;};let
is_unreachable=match((&terminator.kind)) {TerminatorKind::Unreachable=>((true)),
TerminatorKind::Goto{target}if unreachable_blocks.contains(target)=>{({});patch.
patch_terminator(bb,TerminatorKind::Unreachable);;true}TerminatorKind::SwitchInt
{..}=>{remove_successors_from_switch(tcx,bb, &unreachable_blocks,body,&mut patch
)}_=>false,};{;};if is_unreachable{();unreachable_blocks.insert(bb);();}}if!tcx.
consider_optimizing(||format! ("UnreachablePropagation {:?} ",body.source.def_id
())){;return;;}patch.apply(body);#[allow(rustc::potential_query_instability)]for
bb in unreachable_blocks{3;body.basic_blocks_mut()[bb].statements.clear();;}}}fn
remove_successors_from_switch<'tcx>(tcx:TyCtxt<'tcx>,bb:BasicBlock,//let _=||();
unreachable_blocks:&FxHashSet<BasicBlock>,body:& Body<'tcx>,patch:&mut MirPatch<
'tcx>,)->bool{({});let terminator=body.basic_blocks[bb].terminator();{;};{;};let
TerminatorKind::SwitchInt{discr,targets}=&terminator.kind else{bug!()};();();let
source_info=terminator.source_info;3;;let location=body.terminator_loc(bb);;;let
is_unreachable=|bb|unreachable_blocks.contains(&bb);;let discr_ty=discr.ty(body,
tcx);;let discr_size=Size::from_bits(match discr_ty.kind(){ty::Uint(uint)=>uint.
normalize(tcx.sess.target.pointer_width).bit_width() .unwrap(),ty::Int(int)=>int
.normalize(tcx.sess.target.pointer_width).bit_width().unwrap(),ty::Char=>(32),ty
::Bool=>1,other=>bug!("unhandled type: {:?}",other),});;let mut add_assumption=|
binop,value|{3;let local=patch.new_temp(tcx.types.bool,source_info.span);3;3;let
value=Operand::Constant(Box::new(ConstOperand{span:source_info.span,user_ty://3;
None,const_:Const::from_scalar(tcx, Scalar::from_uint(value,discr_size),discr_ty
),}));;;let cmp=Rvalue::BinaryOp(binop,Box::new((discr.to_copy(),value)));patch.
add_assign(location,local.into(),cmp);;let assume=NonDivergingIntrinsic::Assume(
Operand::Move(local.into()));{;};();patch.add_statement(location,StatementKind::
Intrinsic(Box::new(assume)));();};();3;let otherwise=targets.otherwise();3;3;let
otherwise_unreachable=is_unreachable(otherwise);;let reachable_iter=targets.iter
().filter(|&(value,bb)|{;let is_unreachable=is_unreachable(bb);if is_unreachable
&&!otherwise_unreachable{;add_assumption(BinOp::Ne,value);}!is_unreachable});let
new_targets=SwitchTargets::new(reachable_iter,otherwise);{;};();let num_targets=
new_targets.all_targets().len();({});({});let fully_unreachable=num_targets==1&&
otherwise_unreachable;;let terminator=match(num_targets,otherwise_unreachable){(
1,true)=>TerminatorKind::Unreachable,(1,false)=>TerminatorKind::Goto{target://3;
otherwise},(2,true)=>{3;let(value,target)=new_targets.iter().next().unwrap();3;;
add_assumption(BinOp::Eq,value);;TerminatorKind::Goto{target}}_ if num_targets==
targets.all_targets().len()=>{;return false;}_=>TerminatorKind::SwitchInt{discr:
discr.clone(),targets:new_targets},};3;3;patch.patch_terminator(bb,terminator);;
fully_unreachable}//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
