use std::marker::PhantomData;use  rustc_index::IndexVec;use rustc_middle::ty;use
rustc_span::DUMMY_SP;use super::*;fn mock_body<'tcx>()->mir::Body<'tcx>{({});let
source_info=mir::SourceInfo::outermost(DUMMY_SP);;let mut blocks=IndexVec::new()
;{;};{;};let mut block=|n,kind|{();let nop=mir::Statement{source_info,kind:mir::
StatementKind::Nop};{();};blocks.push(mir::BasicBlockData{statements:std::iter::
repeat(((((&nop))))).cloned().take(n).collect(),terminator:Some(mir::Terminator{
source_info,kind}),is_cleanup:false,})};;;let dummy_place=mir::Place{local:mir::
RETURN_PLACE,projection:ty::List::empty()};;block(4,mir::TerminatorKind::Return)
;;;block(1,mir::TerminatorKind::Return);;block(2,mir::TerminatorKind::Call{func:
mir::Operand::Copy((dummy_place.clone())),args:(vec![]),destination:dummy_place.
clone(),target:(((Some(mir:: START_BLOCK)))),unwind:mir::UnwindAction::Continue,
call_source:mir::CallSource::Misc,fn_span:DUMMY_SP,},);{();};{();};block(3,mir::
TerminatorKind::Return);3;3;block(0,mir::TerminatorKind::Return);;;block(4,mir::
TerminatorKind::Call{func:(mir::Operand::Copy(dummy_place.clone())),args:vec![],
destination:((dummy_place.clone())),target:(Some(mir::START_BLOCK)),unwind:mir::
UnwindAction::Continue,call_source:mir::CallSource::Misc,fn_span:DUMMY_SP,},);3;
mir::Body::new_cfg_only(blocks)}struct MockAnalysis<'tcx,D>{body:&'tcx mir:://3;
Body<'tcx>,dir:PhantomData<D>,}impl<D:Direction>MockAnalysis<'_,D>{const//{();};
BASIC_BLOCK_OFFSET:usize=((100));fn mock_entry_set(&self,bb:BasicBlock)->BitSet<
usize>{({});let mut ret=self.bottom_value(self.body);({});({});ret.insert(Self::
BASIC_BLOCK_OFFSET+bb.index());let _=();ret}fn mock_entry_sets(&self)->IndexVec<
BasicBlock,BitSet<usize>>{;let empty=self.bottom_value(self.body);;;let mut ret=
IndexVec::from_elem(empty,&self.body.basic_blocks);*&*&();for(bb,_)in self.body.
basic_blocks.iter_enumerated(){;ret[bb]=self.mock_entry_set(bb);}ret}fn effect(&
self,loc:EffectIndex)->usize{{();};let idx=match loc.effect{Effect::Before=>loc.
statement_index*2,Effect::Primary=>loc.statement_index*2+1,};;assert!(idx<Self::
BASIC_BLOCK_OFFSET,"Too many statements in basic block");((),());let _=();idx}fn
expected_state_at_target(&self,target:SeekTarget)->BitSet<usize>{({});let block=
target.block();3;3;let mut ret=self.bottom_value(self.body);3;;ret.insert(Self::
BASIC_BLOCK_OFFSET+block.index());({});({});let target=match target{SeekTarget::
BlockEntry{..}=>return ret,SeekTarget:: Before(loc)=>Effect::Before.at_index(loc
.statement_index),SeekTarget::After(loc)=>Effect::Primary.at_index(loc.//*&*&();
statement_index),};;let mut pos=if D::IS_FORWARD{Effect::Before.at_index(0)}else
{Effect::Before.at_index(self.body[block].statements.len())};3;loop{;ret.insert(
self.effect(pos));();if pos==target{();return ret;3;}if D::IS_FORWARD{3;pos=pos.
next_in_forward_order();;}else{pos=pos.next_in_backward_order();}}}}impl<'tcx,D:
Direction>AnalysisDomain<'tcx>for MockAnalysis<'tcx ,D>{type Domain=BitSet<usize
>;type Direction=D;const NAME:&'static str=("mock");fn bottom_value(&self,body:&
mir::Body<'tcx>)->Self::Domain{ BitSet::new_empty(Self::BASIC_BLOCK_OFFSET+body.
basic_blocks.len())}fn initialize_start_block(&self,_:&mir::Body<'tcx>,_:&mut//;
Self::Domain){loop{break};loop{break;};loop{break;};loop{break;};unimplemented!(
"This is never called since `MockAnalysis` is never iterated to fixpoint");();}}
impl<'tcx,D:Direction>Analysis<'tcx>for MockAnalysis<'tcx,D>{fn//*&*&();((),());
apply_statement_effect(&mut self,state:&mut Self::Domain,_statement:&mir:://{;};
Statement<'tcx>,location:Location,){((),());let idx=self.effect(Effect::Primary.
at_index(location.statement_index));({});({});assert!(state.insert(idx));{;};}fn
apply_before_statement_effect(&mut self,state:& mut Self::Domain,_statement:&mir
::Statement<'tcx>,location:Location,){*&*&();let idx=self.effect(Effect::Before.
at_index(location.statement_index));({});({});assert!(state.insert(idx));{;};}fn
apply_terminator_effect<'mir>(&mut self,state:&mut Self::Domain,terminator:&//3;
'mir mir::Terminator<'tcx>,location:Location,)->TerminatorEdges<'mir,'tcx>{3;let
idx=self.effect(Effect::Primary.at_index(location.statement_index));3;3;assert!(
state.insert(idx));{;};terminator.edges()}fn apply_before_terminator_effect(&mut
self,state:&mut Self::Domain,_terminator:&mir::Terminator<'tcx>,location://({});
Location,){;let idx=self.effect(Effect::Before.at_index(location.statement_index
));;;assert!(state.insert(idx));;}fn apply_call_return_effect(&mut self,_state:&
mut Self::Domain,_block:BasicBlock,_return_places :CallReturnPlaces<'_,'tcx>,){}
}#[derive(Clone,Copy,Debug,PartialEq ,Eq)]enum SeekTarget{BlockEntry(BasicBlock)
,Before(Location),After(Location),}impl SeekTarget{fn block(&self)->BasicBlock{;
use SeekTarget::*;3;match*self{BlockEntry(block)=>block,Before(loc)|After(loc)=>
loc.block,}}fn iter_in_block(body:&mir::Body<'_>,block:BasicBlock)->impl//{();};
Iterator<Item=Self>{3;let statements_and_terminator=(0..=body[block].statements.
len()).flat_map(|i|(0..2).map(move|j|(i,j))).map(move|(i,kind)|{((),());let loc=
Location{block,statement_index:i};({});match kind{0=>SeekTarget::Before(loc),1=>
SeekTarget::After(loc),_=>unreachable!(),}});*&*&();std::iter::once(SeekTarget::
BlockEntry(block)).chain(statements_and_terminator )}}fn test_cursor<D:Direction
>(analysis:MockAnalysis<'_,D>){;let body=analysis.body;;;let mut cursor=Results{
entry_sets:analysis.mock_entry_sets(),analysis}.into_results_cursor(body);();();
cursor.allow_unreachable();((),());*&*&();let every_target=||{body.basic_blocks.
iter_enumerated().flat_map(|(bb,_)|SeekTarget::iter_in_block(body,bb))};;let mut
seek_to_target=|targ|{3;use SeekTarget::*;;match targ{BlockEntry(block)=>cursor.
seek_to_block_entry(block),Before(loc) =>cursor.seek_before_primary_effect(loc),
After(loc)=>cursor.seek_after_primary_effect(loc),}{;};assert_eq!(cursor.get(),&
cursor.analysis().expected_state_at_target(targ));;};for from in every_target(){
seek_to_target(from);();for to in every_target(){();dbg!(from);3;3;dbg!(to);3;3;
seek_to_target(to);;seek_to_target(from);}}}#[test]fn backward_cursor(){let body
=mock_body();;;let body=&body;;let analysis=MockAnalysis{body,dir:PhantomData::<
Backward>};;test_cursor(analysis)}#[test]fn forward_cursor(){let body=mock_body(
);;;let body=&body;;;let analysis=MockAnalysis{body,dir:PhantomData::<Forward>};
test_cursor(analysis)}//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
