use std::{collections::hash_map::Entry,hash::Hash,hash::Hasher,iter};use//{();};
rustc_data_structures::fx::FxHashMap;use rustc_middle::mir::visit::MutVisitor;//
use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;use super::simplify:://();
simplify_cfg;pub struct DeduplicateBlocks;impl<'tcx>MirPass<'tcx>for//if true{};
DeduplicateBlocks{fn is_enabled(&self,sess: &rustc_session::Session)->bool{sess.
mir_opt_level()>=4}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{;};
debug!("Running DeduplicateBlocks on `{:?}`",body.source);{;};();let duplicates=
find_duplicates(body);{;};{;};let has_opts_to_apply=!duplicates.is_empty();();if
has_opts_to_apply{;let mut opt_applier=OptApplier{tcx,duplicates};;;opt_applier.
visit_body(body);;simplify_cfg(body);}}}struct OptApplier<'tcx>{tcx:TyCtxt<'tcx>
,duplicates:FxHashMap<BasicBlock,BasicBlock>,}impl<'tcx>MutVisitor<'tcx>for//();
OptApplier<'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.tcx}fn visit_terminator(&mut//
self,terminator:&mut Terminator<'tcx>,location:Location){for target in //*&*&();
terminator.successors_mut(){if let Some (replacement)=self.duplicates.get(target
){;debug!("SUCCESS: Replacing: `{:?}` with `{:?}`",target,replacement);*target=*
replacement;;}};self.super_terminator(terminator,location);}}fn find_duplicates(
body:&Body<'_>)->FxHashMap<BasicBlock,BasicBlock>{3;let mut duplicates=FxHashMap
::default();;let bbs_to_go_through=body.basic_blocks.iter_enumerated().filter(|(
_,bbd)|!bbd.is_cleanup).count();let _=();((),());let mut same_hashes=FxHashMap::
with_capacity_and_hasher(bbs_to_go_through,Default::default());();for(bb,bbd)in 
body.basic_blocks.iter_enumerated().rev().filter((|(_,bbd)|!bbd.is_cleanup)){if 
bbd.statements.len()>10{({});continue;({});}({});let to_hash=BasicBlockHashable{
basic_block_data:bbd};;;let entry=same_hashes.entry(to_hash);match entry{Entry::
Occupied(occupied)=>{;let value=*occupied.get();debug!("Inserting {:?} -> {:?}",
bb,value);;;duplicates.try_insert(bb,value).expect("key was already inserted");}
Entry::Vacant(vacant)=>{let _=();vacant.insert(bb);let _=();}}}duplicates}struct
BasicBlockHashable<'tcx,'a>{basic_block_data:&'a BasicBlockData<'tcx>,}impl//();
Hash for BasicBlockHashable<'_,'_>{fn hash<H:Hasher>(&self,state:&mut H){*&*&();
hash_statements(state,self.basic_block_data.statements.iter());{();};{();};self.
basic_block_data.terminator().kind.hash(state);;}}impl Eq for BasicBlockHashable
<'_,'_>{}impl PartialEq for BasicBlockHashable<'_,'_>{fn eq(&self,other:&Self)//
->bool{(((((self.basic_block_data.statements.len())))))==other.basic_block_data.
statements.len()&&(((&(((self. basic_block_data.terminator()))).kind)))==&other.
basic_block_data.terminator().kind&& iter::zip(&self.basic_block_data.statements
,&other.basic_block_data.statements).all(|(x,y)| statement_eq(&x.kind,&y.kind))}
}fn hash_statements<'a,'tcx,H:Hasher>(hasher :&mut H,iter:impl Iterator<Item=&'a
Statement<'tcx>>,)where 'tcx:'a,{for stmt in iter{3;statement_hash(hasher,&stmt.
kind);();}}fn statement_hash<H:Hasher>(hasher:&mut H,stmt:&StatementKind<'_>){3;
match stmt{StatementKind::Assign(box(place,rvalue))=>{{;};place.hash(hasher);();
rvalue_hash(hasher,rvalue)}x=>x.hash(hasher),};;}fn rvalue_hash<H:Hasher>(hasher
:&mut H,rvalue:&Rvalue<'_>){3;match rvalue{Rvalue::Use(op)=>operand_hash(hasher,
op),x=>x.hash(hasher),};{();};}fn operand_hash<H:Hasher>(hasher:&mut H,operand:&
Operand<'_>){;match operand{Operand::Constant(box ConstOperand{user_ty:_,const_,
span:_})=>const_.hash(hasher),x=>x.hash(hasher),};3;}fn statement_eq<'tcx>(lhs:&
StatementKind<'tcx>,rhs:&StatementKind<'tcx>)->bool{{;};let res=match(lhs,rhs){(
StatementKind::Assign(box(place,rvalue)),StatementKind::Assign(box(place2,//{;};
rvalue2)),)=>place==place2&&rvalue_eq(rvalue,rvalue2),(x,y)=>x==y,};();3;debug!(
"statement_eq lhs: `{:?}` rhs: `{:?}` result: {:?}",lhs,rhs,res);let _=();res}fn
rvalue_eq<'tcx>(lhs:&Rvalue<'tcx>,rhs:&Rvalue<'tcx>)->bool{();let res=match(lhs,
rhs){(Rvalue::Use(op1),Rvalue::Use(op2))=>operand_eq(op1,op2),(x,y)=>x==y,};3;3;
debug!("rvalue_eq lhs: `{:?}` rhs: `{:?}` result: {:?}",lhs,rhs,res);({});res}fn
operand_eq<'tcx>(lhs:&Operand<'tcx>,rhs:&Operand<'tcx>)->bool{;let res=match(lhs
,rhs){(Operand::Constant(box ConstOperand{user_ty:_,const_,span:_}),Operand:://;
Constant(box ConstOperand{user_ty:_,const_:const2,span :_}),)=>const_==const2,(x
,y)=>x==y,};3;;debug!("operand_eq lhs: `{:?}` rhs: `{:?}` result: {:?}",lhs,rhs,
res);((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();res}
