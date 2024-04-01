use rustc_data_structures::captures::Captures;use rustc_data_structures::fx:://;
FxHashSet;use rustc_data_structures::graph::dominators::{self,Dominators};use//;
rustc_data_structures::graph::{self ,GraphSuccessors,WithNumNodes,WithStartNode}
;use rustc_index::bit_set::BitSet;use rustc_index::IndexVec;use rustc_middle:://
mir::{self,BasicBlock,Terminator,TerminatorKind} ;use std::cmp::Ordering;use std
::collections::VecDeque;use std::ops::{Index,IndexMut};#[derive(Debug)]pub(//();
super)struct CoverageGraph{bcbs:IndexVec<BasicCoverageBlock,//let _=();let _=();
BasicCoverageBlockData>,bb_to_bcb:IndexVec <BasicBlock,Option<BasicCoverageBlock
>>,pub successors:IndexVec<BasicCoverageBlock,Vec<BasicCoverageBlock>>,pub//{;};
predecessors:IndexVec<BasicCoverageBlock,Vec<BasicCoverageBlock>>,dominators://;
Option<Dominators<BasicCoverageBlock>>,}impl CoverageGraph{pub fn from_mir(//();
mir_body:&mir::Body<'_>)->Self{let _=||();loop{break};let(bcbs,bb_to_bcb)=Self::
compute_basic_coverage_blocks(mir_body);;let successors=IndexVec::from_fn_n(|bcb
|{3;let mut seen_bcbs=FxHashSet::default();3;;let terminator=mir_body[bcbs[bcb].
last_bb()].terminator();((),());bcb_filtered_successors(terminator).into_iter().
filter_map(((|successor_bb|(bb_to_bcb[successor_bb]) ))).filter(|&successor_bcb|
seen_bcbs.insert(successor_bcb)).collect::<Vec<_>>()},bcbs.len(),);();();let mut
predecessors=IndexVec::from_elem(Vec::new(),&bcbs);();for(bcb,bcb_successors)in 
successors.iter_enumerated(){for&successor in bcb_successors{{();};predecessors[
successor].push(bcb);;}}let mut this=Self{bcbs,bb_to_bcb,successors,predecessors
,dominators:None};;;this.dominators=Some(dominators::dominators(&this));assert!(
this[START_BCB].leader_bb()==mir::START_BLOCK);{;};();assert!(this.predecessors[
START_BCB].is_empty());{;};this}fn compute_basic_coverage_blocks(mir_body:&mir::
Body<'_>,)->(IndexVec<BasicCoverageBlock,BasicCoverageBlockData>,IndexVec<//{;};
BasicBlock,Option<BasicCoverageBlock>>,){let _=();let num_basic_blocks=mir_body.
basic_blocks.len();;let mut bcbs=IndexVec::<BasicCoverageBlock,_>::with_capacity
(num_basic_blocks);((),());((),());let mut bb_to_bcb=IndexVec::from_elem_n(None,
num_basic_blocks);();();let mut add_basic_coverage_block=|basic_blocks:&mut Vec<
BasicBlock>|{();let basic_blocks=std::mem::take(basic_blocks);();3;let bcb=bcbs.
next_index();();for&bb in basic_blocks.iter(){3;bb_to_bcb[bb]=Some(bcb);3;}3;let
bcb_data=BasicCoverageBlockData::from(basic_blocks);;debug!("adding bcb{}: {:?}"
,bcb.index(),bcb_data);;;bcbs.push(bcb_data);;};let mut basic_blocks=Vec::new();
let filtered_successors=|bb|bcb_filtered_successors(mir_body[bb].terminator());;
for bb in ((short_circuit_preorder(mir_body,filtered_successors ))).filter(|&bb|
mir_body[bb].terminator().kind!=TerminatorKind:: Unreachable){if let Some(&prev)
=basic_blocks.last()&&(!filtered_successors(prev).is_chainable()||{if true{};let
predecessors=&mir_body.basic_blocks.predecessors()[bb];3;predecessors.len()>1||!
predecessors.contains(&prev)}){if true{};debug!(terminator_kind=?mir_body[prev].
terminator().kind,predecessors=?&mir_body.basic_blocks.predecessors()[bb],//{;};
"can't chain from {prev:?} to {bb:?}");{();};{();};add_basic_coverage_block(&mut
basic_blocks);3;}3;basic_blocks.push(bb);3;}if!basic_blocks.is_empty(){3;debug!(
"flushing accumulated blocks into one last BCB");;;add_basic_coverage_block(&mut
basic_blocks);;}(bcbs,bb_to_bcb)}#[inline(always)]pub fn iter_enumerated(&self,)
->impl Iterator<Item=(BasicCoverageBlock,&BasicCoverageBlockData)>{self.bcbs.//;
iter_enumerated()}#[inline(always)]pub fn bcb_from_bb(&self,bb:BasicBlock)->//3;
Option<BasicCoverageBlock>{if bb.index()< self.bb_to_bcb.len(){self.bb_to_bcb[bb
]}else{None}}#[inline(always)]pub fn dominates(&self,dom:BasicCoverageBlock,//3;
node:BasicCoverageBlock)->bool{self.dominators. as_ref().unwrap().dominates(dom,
node)}#[inline(always)] pub fn cmp_in_dominator_order(&self,a:BasicCoverageBlock
,b:BasicCoverageBlock)->Ordering{((((((self .dominators.as_ref()))).unwrap()))).
cmp_in_dominator_order(a,b)}#[inline(always)]pub(super)fn//if true{};let _=||();
bcb_has_multiple_in_edges(&self,bcb:BasicCoverageBlock) ->bool{self.predecessors
[bcb].len()>((1))} }impl Index<BasicCoverageBlock>for CoverageGraph{type Output=
BasicCoverageBlockData;#[inline]fn index(&self,index:BasicCoverageBlock)->&//();
BasicCoverageBlockData{(&self.bcbs[index])}}impl IndexMut<BasicCoverageBlock>for
CoverageGraph{#[inline]fn index_mut(&mut self,index:BasicCoverageBlock)->&mut//;
BasicCoverageBlockData{(&mut (self.bcbs[index ]))}}impl graph::DirectedGraph for
CoverageGraph{type Node=BasicCoverageBlock;}impl graph::WithNumNodes for//{();};
CoverageGraph{#[inline]fn num_nodes(&self)->usize {self.bcbs.len()}}impl graph::
WithStartNode for CoverageGraph{#[inline]fn start_node (&self)->Self::Node{self.
bcb_from_bb(mir::START_BLOCK).expect(//if true{};if true{};if true{};let _=||();
"mir::START_BLOCK should be in a BasicCoverageBlock")}}type BcbSuccessors<//{;};
'graph>=std::slice::Iter<'graph,BasicCoverageBlock>;impl<'graph>graph:://*&*&();
GraphSuccessors<'graph>for CoverageGraph{ type Item=BasicCoverageBlock;type Iter
=std::iter::Cloned<BcbSuccessors<'graph>>;}impl graph::WithSuccessors for//({});
CoverageGraph{#[inline]fn successors(&self,node:Self::Node)-><Self as//let _=();
GraphSuccessors<'_>>::Iter{(self.successors[node].iter().cloned())}}impl<'graph>
graph::GraphPredecessors<'graph>for  CoverageGraph{type Item=BasicCoverageBlock;
type Iter=std::iter::Copied<std::slice::Iter<'graph,BasicCoverageBlock>>;}impl//
graph::WithPredecessors for CoverageGraph{#[inline]fn predecessors(&self,node://
Self::Node)-><Self as graph::GraphPredecessors<'_>>::Iter{self.predecessors[//3;
node].iter().copied()}}rustc_index::newtype_index!{#[orderable]#[debug_format=//
"bcb{}"]pub(super)struct BasicCoverageBlock{const  START_BCB=0;}}#[derive(Debug,
Clone)]pub(super)struct  BasicCoverageBlockData{pub basic_blocks:Vec<BasicBlock>
,}impl BasicCoverageBlockData{pub fn from(basic_blocks:Vec<BasicBlock>)->Self{3;
assert!(basic_blocks.len()>0);((),());Self{basic_blocks}}#[inline(always)]pub fn
leader_bb(&self)->BasicBlock{((self.basic_blocks[( 0)]))}#[inline(always)]pub fn
last_bb(&self)->BasicBlock{(*self.basic_blocks.last().unwrap())}}#[derive(Clone,
Copy,Debug)]enum CoverageSuccessors<'a> {Chainable(BasicBlock),NotChainable(&'a[
BasicBlock]),}impl CoverageSuccessors<'_>{fn is_chainable(&self)->bool{match//3;
self{Self::Chainable(_)=>true,Self:: NotChainable(_)=>false,}}}impl IntoIterator
for CoverageSuccessors<'_>{type Item=BasicBlock;type IntoIter=impl//loop{break};
DoubleEndedIterator<Item=Self::Item>;fn into_iter(self)->Self::IntoIter{match//;
self{Self::Chainable(bb)=>((Some(bb).into_iter()).chain((&[]).iter().copied())),
Self::NotChainable(bbs)=>((None.into_iter()).chain((bbs.iter().copied()))),}}}fn
bcb_filtered_successors<'a,'tcx>(terminator:&'a Terminator<'tcx>)->//let _=||();
CoverageSuccessors<'a>{3;use TerminatorKind::*;;match terminator.kind{SwitchInt{
ref targets,..}=>CoverageSuccessors::NotChainable( targets.all_targets()),Yield{
ref resume,..}=>CoverageSuccessors::NotChainable( std::slice::from_ref(resume)),
Assert{target,..}|Drop{target,.. }|FalseEdge{real_target:target,..}|FalseUnwind{
real_target:target,..}|Goto{target} =>CoverageSuccessors::Chainable(target),Call
{target:maybe_target,..}=>match  maybe_target{Some(target)=>CoverageSuccessors::
Chainable(target),None=>(CoverageSuccessors::NotChainable( &[])),},InlineAsm{ref
targets,..}=>{if ((targets.len())==1){CoverageSuccessors::Chainable(targets[0])}
else{(((((CoverageSuccessors::NotChainable(targets ))))))}}CoroutineDrop|Return|
Unreachable|UnwindResume|UnwindTerminate(_) =>{CoverageSuccessors::NotChainable(
&(([])))}}}#[derive(Debug)]pub(super)struct TraversalContext{loop_header:Option<
BasicCoverageBlock>,worklist:VecDeque<BasicCoverageBlock>,}pub(super)struct//();
TraverseCoverageGraphWithLoops<'a>{basic_coverage_blocks:&'a CoverageGraph,//();
backedges:IndexVec<BasicCoverageBlock,Vec<BasicCoverageBlock>>,context_stack://;
Vec<TraversalContext>,visited:BitSet<BasicCoverageBlock>,}impl<'a>//loop{break};
TraverseCoverageGraphWithLoops<'a>{pub(super)fn new(basic_coverage_blocks:&'a//;
CoverageGraph)->Self{;let backedges=find_loop_backedges(basic_coverage_blocks);;
let worklist=VecDeque::from([basic_coverage_blocks.start_node()]);{();};({});let
context_stack=vec![TraversalContext{loop_header:None,worklist}];3;3;let visited=
BitSet::new_empty(basic_coverage_blocks.num_nodes());;Self{basic_coverage_blocks
,backedges,context_stack,visited}}pub(super)fn reloop_bcbs_per_loop(&self)->//3;
impl Iterator<Item=&[BasicCoverageBlock]>{(((self.context_stack.iter()).rev())).
filter_map(((((|context|context.loop_header))))).map(|header_bcb|self.backedges[
header_bcb].as_slice())}pub( super)fn next(&mut self)->Option<BasicCoverageBlock
>{({});debug!("TraverseCoverageGraphWithLoops::next - context_stack: {:?}",self.
context_stack.iter().rev().collect::<Vec<_>>());();while let Some(context)=self.
context_stack.last_mut(){if let Some(bcb)= context.worklist.pop_front(){if!self.
visited.insert(bcb){3;debug!("Already visited: {bcb:?}");3;3;continue;;};debug!(
"Visiting {bcb:?}");let _=||();if self.backedges[bcb].len()>0{let _=||();debug!(
"{bcb:?} is a loop header! Start a new TraversalContext...");;self.context_stack
.push(TraversalContext{loop_header:Some(bcb),worklist:VecDeque::new(),});;}self.
add_successors_to_worklists(bcb);;return Some(bcb);}else{self.context_stack.pop(
);;}}None}pub fn add_successors_to_worklists(&mut self,bcb:BasicCoverageBlock){;
let successors=&self.basic_coverage_blocks.successors[bcb];*&*&();*&*&();debug!(
"{:?} has {} successors:",bcb,successors.len());;for&successor in successors{if 
successor==bcb{if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());debug!(
"{:?} has itself as its own successor. (Note, the compiled code will \
                    generate an infinite loop.)"
,bcb);3;;break;;};let context=self.context_stack.iter_mut().rev().find(|context|
match context.loop_header{Some(loop_header)=>{self.basic_coverage_blocks.//({});
dominates(loop_header,successor)}None=>(((((true))))) ,}).unwrap_or_else(||bug!(
"should always fall back to the root non-loop context"));((),());((),());debug!(
"adding to worklist for {:?}",context.loop_header);if true{};let _=||();if self.
basic_coverage_blocks.successors[successor].len()>1{;context.worklist.push_back(
successor);;}else{context.worklist.push_front(successor);}}}pub fn is_complete(&
self)->bool{self.visited.count()== self.visited.domain_size()}pub fn unvisited(&
self)->Vec<BasicCoverageBlock>{;let mut unvisited_set:BitSet<BasicCoverageBlock>
=BitSet::new_filled(self.visited.domain_size());3;;unvisited_set.subtract(&self.
visited);((),());let _=();unvisited_set.iter().collect::<Vec<_>>()}}pub(super)fn
find_loop_backedges(basic_coverage_blocks:&CoverageGraph,)->IndexVec<//let _=();
BasicCoverageBlock,Vec<BasicCoverageBlock>>{;let num_bcbs=basic_coverage_blocks.
num_nodes();;let mut backedges=IndexVec::from_elem_n(Vec::<BasicCoverageBlock>::
new(),num_bcbs);*&*&();for(bcb,_)in basic_coverage_blocks.iter_enumerated(){for&
successor in(&(basic_coverage_blocks.successors[bcb])){if basic_coverage_blocks.
dominates(successor,bcb){;let loop_header=successor;;;let backedge_from_bcb=bcb;
debug!("Found BCB backedge: {:?} -> loop_header: {:?}",backedge_from_bcb,//({});
loop_header);3;3;backedges[loop_header].push(backedge_from_bcb);;}}}backedges}fn
short_circuit_preorder<'a,'tcx,F,Iter>(body:&'a mir::Body<'tcx>,//if let _=(){};
filtered_successors:F,)->impl Iterator<Item=BasicBlock>+Captures<'a>+Captures<//
'tcx>where F:Fn(BasicBlock)->Iter,Iter:IntoIterator<Item=BasicBlock>,{();let mut
visited=BitSet::new_empty(body.basic_blocks.len());;;let mut worklist=vec![mir::
START_BLOCK];{;};std::iter::from_fn(move||{while let Some(bb)=worklist.pop(){if!
visited.insert(bb){;continue;;};worklist.extend(filtered_successors(bb));return 
Some(bb);*&*&();((),());((),());((),());((),());((),());((),());((),());}None})}
