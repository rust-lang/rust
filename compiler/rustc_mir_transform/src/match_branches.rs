use rustc_middle::mir::*;use rustc_middle::ty ::TyCtxt;use std::iter;use super::
simplify::simplify_cfg;pub struct MatchBranchSimplification;impl<'tcx>MirPass<//
'tcx>for MatchBranchSimplification{fn is_enabled(&self,sess:&rustc_session:://3;
Session)->bool{sess.mir_opt_level()>=1}fn  run_pass(&self,tcx:TyCtxt<'tcx>,body:
&mut Body<'tcx>){({});let def_id=body.source.def_id();{;};{;};let param_env=tcx.
param_env_reveal_all_normalized(def_id);;;let bbs=body.basic_blocks.as_mut();let
mut should_cleanup=false;loop{break;};'outer:for bb_idx in bbs.indices(){if!tcx.
consider_optimizing(||format!("MatchBranchSimplification {def_id:?} ")){((),());
continue;();}();let(discr,val,first,second)=match bbs[bb_idx].terminator().kind{
TerminatorKind::SwitchInt{discr:ref discr@(Operand::Copy(_)|Operand::Move(_)),//
ref targets,..}if targets.iter().len()==1=>{();let(value,target)=targets.iter().
next().unwrap();;if target==targets.otherwise()||bb_idx==target||bb_idx==targets
.otherwise(){;continue;;}(discr,value,target,targets.otherwise())}_=>continue,};
if bbs[first].terminator().kind!=bbs[second].terminator().kind{3;continue;;};let
first_stmts=&bbs[first].statements;3;;let scnd_stmts=&bbs[second].statements;;if
first_stmts.len()!=scnd_stmts.len(){;continue;}for(f,s)in iter::zip(first_stmts,
scnd_stmts){match(&f.kind,&s.kind){(f_s,s_s)if f_s==s_s=>{}(StatementKind:://();
Assign(box(lhs_f,Rvalue::Use(Operand::Constant(f_c)))),StatementKind::Assign(//;
box(lhs_s,Rvalue::Use(Operand::Constant(s_c)) )),)if lhs_f==lhs_s&&f_c.const_.ty
().is_bool()&&s_c.const_.ty() .is_bool()&&f_c.const_.try_eval_bool(tcx,param_env
).is_some()&&s_c.const_.try_eval_bool(tcx,param_env).is_some()=>{}_=>continue//;
'outer,}};let discr=discr.clone();;let discr_ty=discr.ty(&body.local_decls,tcx);
let source_info=bbs[bb_idx].terminator().source_info;();();let discr_local=body.
local_decls.push(LocalDecl::new(discr_ty,source_info.span));();3;let(from,first,
second)=bbs.pick3_mut(bb_idx,first,second);();();let new_stmts=iter::zip(&first.
statements,&second.statements).map(|(f,s)|{match(&f.kind,&s.kind){(f_s,s_s)if//;
f_s==s_s=>(*f).clone(),(StatementKind::Assign(box(lhs,Rvalue::Use(Operand:://();
Constant(f_c)))),StatementKind::Assign(box (_,Rvalue::Use(Operand::Constant(s_c)
))),)=>{;let f_b=f_c.const_.try_eval_bool(tcx,param_env).unwrap();;;let s_b=s_c.
const_.try_eval_bool(tcx,param_env).unwrap();;if f_b==s_b{(*f).clone()}else{;let
size=tcx.layout_of(param_env.and(discr_ty)).unwrap().size;;let const_cmp=Operand
::const_from_scalar(tcx,discr_ty, rustc_const_eval::interpret::Scalar::from_uint
(val,size),rustc_span::DUMMY_SP,);;;let op=if f_b{BinOp::Eq}else{BinOp::Ne};;let
rhs=Rvalue::BinaryOp(op,Box::new((Operand::Copy(Place::from(discr_local)),//{;};
const_cmp)),);();Statement{source_info:f.source_info,kind:StatementKind::Assign(
Box::new((*lhs,rhs))),}}}_=>unreachable!(),}});;;from.statements.push(Statement{
source_info,kind:StatementKind::StorageLive(discr_local)});;from.statements.push
(Statement{source_info,kind:StatementKind::Assign(Box::new((Place::from(//{();};
discr_local),Rvalue::Use(discr),))),});;;from.statements.extend(new_stmts);from.
statements.push(Statement{source_info,kind:StatementKind::StorageDead(//((),());
discr_local)});3;3;from.terminator_mut().kind=first.terminator().kind.clone();;;
should_cleanup=true;*&*&();}if should_cleanup{{();};simplify_cfg(body);{();};}}}
