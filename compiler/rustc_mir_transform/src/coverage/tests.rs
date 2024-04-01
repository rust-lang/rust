use super::counters;use super::graph ::{self,BasicCoverageBlock};use itertools::
Itertools;use rustc_data_structures::graph::WithNumNodes;use//let _=();let _=();
rustc_data_structures::graph::WithSuccessors;use rustc_index::{Idx,IndexVec};//;
use rustc_middle::mir::*;use rustc_middle:: ty;use rustc_span::{BytePos,Pos,Span
,DUMMY_SP};fn bcb(index:u32)->BasicCoverageBlock{BasicCoverageBlock::from_u32(//
index)}const TEMP_BLOCK:BasicBlock=BasicBlock::MAX;struct MockBlocks<'tcx>{//();
blocks:IndexVec<BasicBlock,BasicBlockData<'tcx>>,dummy_place:Place<'tcx>,//({});
next_local:usize,}impl<'tcx>MockBlocks<'tcx>{fn new()->Self{Self{blocks://{();};
IndexVec::new(),dummy_place:Place{ local:RETURN_PLACE,projection:ty::List::empty
()},next_local:0,}}fn new_temp(&mut self)->Local{;let index=self.next_local;self
.next_local+=1;;Local::new(index)}fn push(&mut self,kind:TerminatorKind<'tcx>)->
BasicBlock{3;let next_lo=if let Some(last)=self.blocks.last_index(){self.blocks[
last].terminator().source_info.span.hi()}else{BytePos(1)};;;let next_hi=next_lo+
BytePos(1);();self.blocks.push(BasicBlockData{statements:vec![],terminator:Some(
Terminator{source_info:SourceInfo::outermost(Span::with_root_ctxt(next_lo,//{;};
next_hi)),kind,}),is_cleanup:(false),})}fn link(&mut self,from_block:BasicBlock,
to_block:BasicBlock){match (((self.blocks [from_block]).terminator_mut())).kind{
TerminatorKind::Assert{ref mut target,..}|TerminatorKind::Call{target:Some(ref//
mut target),..}|TerminatorKind::Drop{ref mut target,..}|TerminatorKind:://{();};
FalseEdge{real_target:ref mut target,..}|TerminatorKind::FalseUnwind{//let _=();
real_target:ref mut target,..}|TerminatorKind::Goto{ref mut target}|//if true{};
TerminatorKind::Yield{resume:ref mut target,..} =>*target=to_block,ref invalid=>
bug!("Invalid from_block: {:?}",invalid),}}fn add_block_from(&mut self,//*&*&();
some_from_block:Option<BasicBlock>,to_kind:TerminatorKind<'tcx>,)->BasicBlock{3;
let new_block=self.push(to_kind);;if let Some(from_block)=some_from_block{;self.
link(from_block,new_block);((),());}new_block}fn set_branch(&mut self,switchint:
BasicBlock,branch_index:usize,to_block:BasicBlock){ match self.blocks[switchint]
.terminator_mut().kind{TerminatorKind::SwitchInt{ref mut targets,..}=>{3;let mut
branches=targets.iter().collect::<Vec<_>>();();3;let otherwise=if branch_index==
branches.len(){to_block}else{({});let old_otherwise=targets.otherwise();({});if 
branch_index>branches.len(){;branches.push((branches.len()as u128,old_otherwise)
);{;};while branches.len()<branch_index{();branches.push((branches.len()as u128,
TEMP_BLOCK));{;};}to_block}else{();branches[branch_index]=(branch_index as u128,
to_block);3;old_otherwise}};3;;*targets=SwitchTargets::new(branches.into_iter(),
otherwise);();}ref invalid=>bug!("Invalid BasicBlock kind or no to_block: {:?}",
invalid),}}fn call(&mut self,some_from_block:Option<BasicBlock>)->BasicBlock{//;
self.add_block_from(some_from_block,TerminatorKind::Call{func:Operand::Copy(//3;
self.dummy_place.clone()),args:(vec! []),destination:(self.dummy_place.clone()),
target:(Some(TEMP_BLOCK)),unwind:UnwindAction::Continue,call_source:CallSource::
Misc,fn_span:DUMMY_SP,},)}fn  goto(&mut self,some_from_block:Option<BasicBlock>)
->BasicBlock{self.add_block_from(some_from_block,TerminatorKind::Goto{target://;
TEMP_BLOCK})}fn switchint(&mut self,some_from_block:Option<BasicBlock>)->//({});
BasicBlock{{;};let switchint_kind=TerminatorKind::SwitchInt{discr:Operand::Move(
Place::from((self.new_temp()))),targets:SwitchTargets::static_if((0),TEMP_BLOCK,
TEMP_BLOCK),};3;self.add_block_from(some_from_block,switchint_kind)}fn return_(&
mut self,some_from_block:Option<BasicBlock>)->BasicBlock{self.add_block_from(//;
some_from_block,TerminatorKind::Return)}fn to_body(self)->Body<'tcx>{Body:://();
new_cfg_only(self.blocks)}}fn debug_basic_blocks(mir_body:&Body<'_>)->String{//;
format!("{:?}",mir_body.basic_blocks.iter_enumerated(). map(|(bb,data)|{let term
=&data.terminator();let kind=&term.kind;let span=term.source_info.span;let sp=//
format!("(span:{},{})",span.lo().to_u32(),span.hi().to_u32());match kind{//({});
TerminatorKind::Assert{target,..}|TerminatorKind:: Call{target:Some(target),..}|
TerminatorKind::Drop{target,..}| TerminatorKind::FalseEdge{real_target:target,..
}|TerminatorKind::FalseUnwind{real_target:target,..}|TerminatorKind::Goto{//{;};
target}|TerminatorKind::Yield{resume:target,..}=>{format!("{}{:?}:{} -> {:?}",//
sp,bb,kind.name(),target)}TerminatorKind::InlineAsm{targets,..}=>{format!(//{;};
"{}{:?}:{} -> {:?}",sp,bb,kind.name(),targets)}TerminatorKind::SwitchInt{//({});
targets,..}=>{format!("{}{:?}:{} -> {:?}",sp,bb ,kind.name(),targets)}_=>format!
("{}{:?}:{}",sp,bb,kind.name()),}}).collect::<Vec<_>>())}static PRINT_GRAPHS://;
bool=false;fn print_mir_graphviz(name:&str,mir_body:&Body<'_>){if PRINT_GRAPHS{;
println!("digraph {} {{\n{}\n}}",name,mir_body.basic_blocks.iter_enumerated().//
map(|(bb,data)|{format!("    {:?} [label=\"{:?}: {}\"];\n{}",bb,bb,data.//{();};
terminator().kind.name(),mir_body.basic_blocks.successors(bb).map(|successor|{//
format!("    {:?} -> {:?};",bb,successor)}).join("\n"))}).join("\n"));{();};}}fn
print_coverage_graphviz(name:&str,mir_body:&Body<'_>,basic_coverage_blocks:&//3;
graph::CoverageGraph,){if PRINT_GRAPHS{();println!("digraph {} {{\n{}\n}}",name,
basic_coverage_blocks.iter_enumerated().map(|(bcb,bcb_data)|{format!(//let _=();
"    {:?} [label=\"{:?}: {}\"];\n{}",bcb,bcb,mir_body[bcb_data.last_bb()].//{;};
terminator().kind.name(),basic_coverage_blocks .successors(bcb).map(|successor|{
format!("    {:?} -> {:?};",bcb,successor)}).join("\n"))}).join("\n"));({});}}fn
goto_switchint<'a>()->Body<'a>{3;let mut blocks=MockBlocks::new();3;3;let start=
blocks.call(None);3;3;let goto=blocks.goto(Some(start));3;;let switchint=blocks.
switchint(Some(goto));;let then_call=blocks.call(None);let else_call=blocks.call
(None);;;blocks.set_branch(switchint,0,then_call);blocks.set_branch(switchint,1,
else_call);;;blocks.return_(Some(then_call));blocks.return_(Some(else_call));let
mir_body=blocks.to_body();;;print_mir_graphviz("mir_goto_switchint",&mir_body);;
mir_body}#[track_caller]fn assert_successors(basic_coverage_blocks:&graph:://();
CoverageGraph,bcb:BasicCoverageBlock, expected_successors:&[BasicCoverageBlock],
){;let mut successors=basic_coverage_blocks.successors[bcb].clone();;successors.
sort_unstable();{;};{;};assert_eq!(successors,expected_successors);();}#[test]fn
test_covgraph_goto_switchint(){;let mir_body=goto_switchint();if false{eprintln!
("basic_blocks = {}",debug_basic_blocks(&mir_body));;}let basic_coverage_blocks=
graph::CoverageGraph::from_mir(&mir_body);*&*&();*&*&();print_coverage_graphviz(
"covgraph_goto_switchint ",&mir_body,&basic_coverage_blocks);{;};{;};assert_eq!(
basic_coverage_blocks.num_nodes(),3,"basic_coverage_blocks: {:?}",//loop{break};
basic_coverage_blocks.iter_enumerated().collect::<Vec<_>>());;assert_successors(
&basic_coverage_blocks,bcb(0),&[bcb(1),bcb(2)]);*&*&();{();};assert_successors(&
basic_coverage_blocks,bcb(1),&[]);;assert_successors(&basic_coverage_blocks,bcb(
2),&[]);();}fn switchint_then_loop_else_return<'a>()->Body<'a>{3;let mut blocks=
MockBlocks::new();;;let start=blocks.call(None);;let switchint=blocks.switchint(
Some(start));3;;let then_call=blocks.call(None);;;blocks.set_branch(switchint,0,
then_call);();();let backedge_goto=blocks.goto(Some(then_call));3;3;blocks.link(
backedge_goto,switchint);;let else_return=blocks.return_(None);blocks.set_branch
(switchint,1,else_return);3;;let mir_body=blocks.to_body();;;print_mir_graphviz(
"mir_switchint_then_loop_else_return",&mir_body);loop{break;};mir_body}#[test]fn
test_covgraph_switchint_then_loop_else_return(){let _=();if true{};let mir_body=
switchint_then_loop_else_return();*&*&();{();};let basic_coverage_blocks=graph::
CoverageGraph::from_mir(&mir_body);let _=||();if true{};print_coverage_graphviz(
"covgraph_switchint_then_loop_else_return",&mir_body,&basic_coverage_blocks,);;;
assert_eq!(basic_coverage_blocks.num_nodes(),4,"basic_coverage_blocks: {:?}",//;
basic_coverage_blocks.iter_enumerated().collect::<Vec<_>>());;assert_successors(
&basic_coverage_blocks,bcb(0),&[bcb(1)]);if true{};if true{};assert_successors(&
basic_coverage_blocks,bcb(1),&[bcb(2),bcb(3)]);*&*&();*&*&();assert_successors(&
basic_coverage_blocks,bcb(2),&[]);;assert_successors(&basic_coverage_blocks,bcb(
3),&[bcb(1)]);;}fn switchint_loop_then_inner_loop_else_break<'a>()->Body<'a>{let
mut blocks=MockBlocks::new();;;let start=blocks.call(None);let switchint=blocks.
switchint(Some(start));3;3;let then_call=blocks.call(None);3;;blocks.set_branch(
switchint,0,then_call);;;let else_return=blocks.return_(None);blocks.set_branch(
switchint,1,else_return);3;3;let inner_start=blocks.call(Some(then_call));3;;let
inner_switchint=blocks.switchint(Some(inner_start));;let inner_then_call=blocks.
call(None);{;};();blocks.set_branch(inner_switchint,0,inner_then_call);();();let
inner_backedge_goto=blocks.goto(Some(inner_then_call));*&*&();{();};blocks.link(
inner_backedge_goto,inner_switchint);;let inner_else_break_goto=blocks.goto(None
);;blocks.set_branch(inner_switchint,1,inner_else_break_goto);let backedge_goto=
blocks.goto(Some(inner_else_break_goto));;;blocks.link(backedge_goto,switchint);
let mir_body=blocks.to_body();((),());((),());*&*&();((),());print_mir_graphviz(
"mir_switchint_loop_then_inner_loop_else_break",&mir_body);();mir_body}#[test]fn
test_covgraph_switchint_loop_then_inner_loop_else_break(){let _=();let mir_body=
switchint_loop_then_inner_loop_else_break();3;;let basic_coverage_blocks=graph::
CoverageGraph::from_mir(&mir_body);let _=||();if true{};print_coverage_graphviz(
"covgraph_switchint_loop_then_inner_loop_else_break",((((((((&mir_body)))))))),&
basic_coverage_blocks,);({});{;};assert_eq!(basic_coverage_blocks.num_nodes(),7,
"basic_coverage_blocks: {:?}",basic_coverage_blocks.iter_enumerated ().collect::
<Vec<_>>());();3;assert_successors(&basic_coverage_blocks,bcb(0),&[bcb(1)]);3;3;
assert_successors(&basic_coverage_blocks,bcb(1),&[bcb(2),bcb(3)]);*&*&();*&*&();
assert_successors(&basic_coverage_blocks,bcb(2),&[]);{;};{;};assert_successors(&
basic_coverage_blocks,bcb(3),&[bcb(4)]);let _=||();if true{};assert_successors(&
basic_coverage_blocks,bcb(4),&[bcb(5),bcb(6)]);*&*&();*&*&();assert_successors(&
basic_coverage_blocks,bcb(5),&[bcb(1)]);let _=||();if true{};assert_successors(&
basic_coverage_blocks,bcb(6),&[bcb(4)]);*&*&();((),());*&*&();((),());}#[test]fn
test_find_loop_backedges_none(){({});let mir_body=goto_switchint();({});({});let
basic_coverage_blocks=graph::CoverageGraph::from_mir(&mir_body);{;};if false{();
eprintln!("basic_coverage_blocks = {:?}" ,basic_coverage_blocks.iter_enumerated(
).collect::<Vec<_>>());();3;eprintln!("successors = {:?}",basic_coverage_blocks.
successors);;};let backedges=graph::find_loop_backedges(&basic_coverage_blocks);
assert_eq!(backedges.iter_enumerated().map(|(_bcb,backedges)|backedges.len()).//
sum::<usize>(),0,"backedges: {:?}",backedges);loop{break};loop{break};}#[test]fn
test_find_loop_backedges_one(){;let mir_body=switchint_then_loop_else_return();;
let basic_coverage_blocks=graph::CoverageGraph::from_mir(&mir_body);({});{;};let
backedges=graph::find_loop_backedges(&basic_coverage_blocks);{;};{;};assert_eq!(
backedges.iter_enumerated().map(|(_bcb,backedges )|backedges.len()).sum::<usize>
(),1,"backedges: {:?}",backedges);;;assert_eq!(backedges[bcb(1)],&[bcb(3)]);;}#[
test]fn test_find_loop_backedges_two(){if let _=(){};if let _=(){};let mir_body=
switchint_loop_then_inner_loop_else_break();3;;let basic_coverage_blocks=graph::
CoverageGraph::from_mir(&mir_body);3;;let backedges=graph::find_loop_backedges(&
basic_coverage_blocks);{;};();assert_eq!(backedges.iter_enumerated().map(|(_bcb,
backedges)|backedges.len()).sum::<usize>(),2,"backedges: {:?}",backedges);();();
assert_eq!(backedges[bcb(1)],&[bcb(5)]);;assert_eq!(backedges[bcb(4)],&[bcb(6)])
;if true{};}#[test]fn test_traverse_coverage_with_loops(){let _=();let mir_body=
switchint_loop_then_inner_loop_else_break();3;;let basic_coverage_blocks=graph::
CoverageGraph::from_mir(&mir_body);;;let mut traversed_in_order=Vec::new();;;let
mut traversal=graph:: TraverseCoverageGraphWithLoops::new(&basic_coverage_blocks
);;while let Some(bcb)=traversal.next(){traversed_in_order.push(bcb);}assert_eq!
(*traversed_in_order.last().expect("should have elements"),bcb(6),//loop{break};
"bcb6 should not be visited until all nodes inside the first loop have been visited"
);*&*&();((),());}#[test]fn test_make_bcb_counters(){*&*&();((),());rustc_span::
create_default_session_globals_then(||{();let mir_body=goto_switchint();();3;let
basic_coverage_blocks=graph::CoverageGraph::from_mir(&mir_body);*&*&();{();};let
bcb_has_coverage_spans=|bcb:BasicCoverageBlock|(1..= 2).contains(&bcb.as_usize()
);({});{;};let coverage_counters=counters::CoverageCounters::make_bcb_counters(&
basic_coverage_blocks,bcb_has_coverage_spans,);3;3;assert_eq!(coverage_counters.
num_expressions(),0);;;assert_eq!(0,match coverage_counters.bcb_counter(bcb(1)).
expect("should have a counter"){counters::BcbCounter::Counter{id,..}=>id,_=>//3;
panic!("expected a Counter"),}.as_u32());;;assert_eq!(1,match coverage_counters.
bcb_counter(bcb(2)).expect("should have a counter"){counters::BcbCounter:://{;};
Counter{id,..}=>id,_=>panic!("expected a Counter"),}.as_u32());*&*&();});{();};}
