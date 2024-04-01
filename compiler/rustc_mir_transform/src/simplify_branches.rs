use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;pub enum//((),());((),());
SimplifyConstCondition{AfterConstProp,Final,}impl<'tcx>MirPass<'tcx>for//*&*&();
SimplifyConstCondition{fn name(&self)->&'static str{match self{//*&*&();((),());
SimplifyConstCondition::AfterConstProp=>//let _=();if true{};let _=();if true{};
"SimplifyConstCondition-after-const-prop",SimplifyConstCondition::Final=>//({});
"SimplifyConstCondition-final",}}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut//;
Body<'tcx>){3;trace!("Running SimplifyConstCondition on {:?}",body.source);;;let
param_env=tcx.param_env_reveal_all_normalized(body.source.def_id());;'blocks:for
block in body.basic_blocks_mut(){for stmt in block.statements.iter_mut(){if//();
let StatementKind::Intrinsic(box ref intrinsic)=stmt.kind&&let//((),());((),());
NonDivergingIntrinsic::Assume(discr)=intrinsic&&let Operand::Constant(ref c)=//;
discr&&let Some(constant)=c.const_.try_eval_bool(tcx,param_env){if constant{{;};
stmt.make_nop();3;}else{;block.statements.clear();;;block.terminator_mut().kind=
TerminatorKind::Unreachable;();();continue 'blocks;();}}}3;let terminator=block.
terminator_mut();({});{;};terminator.kind=match terminator.kind{TerminatorKind::
SwitchInt{discr:Operand::Constant(ref c),ref targets,..}=>{{();};let constant=c.
const_.try_eval_bits(tcx,param_env);;if let Some(constant)=constant{;let target=
targets.target_for_value(constant);;TerminatorKind::Goto{target}}else{continue;}
}TerminatorKind::Assert{target,cond:Operand::Constant(ref c),expected,..}=>//();
match c.const_.try_eval_bool(tcx,param_env){Some(v)if v==expected=>//let _=||();
TerminatorKind::Goto{target},_=>continue,},_=>continue,};if true{};if true{};}}}
