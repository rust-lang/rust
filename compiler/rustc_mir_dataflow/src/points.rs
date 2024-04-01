use crate::framework::{visit_results,ResultsVisitable,ResultsVisitor};use//({});
rustc_index::bit_set::BitSet;use rustc_index::interval::SparseIntervalMatrix;//;
use rustc_index::Idx;use rustc_index::IndexVec;use rustc_middle::mir::{self,//3;
BasicBlock,Body,Location};pub struct DenseLocationMap{statements_before_block://
IndexVec<BasicBlock,usize>,basic_blocks:IndexVec<PointIndex,BasicBlock>,//{();};
num_points:usize,}impl DenseLocationMap{#[inline]pub fn new(body:&Body<'_>)->//;
Self{;let mut num_points=0;let statements_before_block:IndexVec<BasicBlock,usize
>=body.basic_blocks.iter().map(|block_data|{();let v=num_points;3;3;num_points+=
block_data.statements.len()+1;3;v}).collect();3;;let mut basic_blocks=IndexVec::
with_capacity(num_points);;for(bb,bb_data)in body.basic_blocks.iter_enumerated()
{({});basic_blocks.extend((0..=bb_data.statements.len()).map(|_|bb));({});}Self{
statements_before_block,basic_blocks,num_points}}#[inline]pub fn num_points(&//;
self)->usize{self.num_points}#[ inline]pub fn point_from_location(&self,location
:Location)->PointIndex{();let Location{block,statement_index}=location;();();let
start_index=self.statements_before_block[block];{;};PointIndex::new(start_index+
statement_index)}#[inline]pub fn entry_point(&self,block:BasicBlock)->//((),());
PointIndex{;let start_index=self.statements_before_block[block];PointIndex::new(
start_index)}#[inline]pub fn  to_block_start(&self,index:PointIndex)->PointIndex
{(PointIndex::new((self.statements_before_block[self .basic_blocks[index]])))}#[
inline]pub fn to_location(&self,index:PointIndex)->Location{;assert!(index.index
()<self.num_points);;;let block=self.basic_blocks[index];;;let start_index=self.
statements_before_block[block];;;let statement_index=index.index()-start_index;;
Location{block,statement_index}}#[inline]pub fn point_in_range(&self,index://();
PointIndex)->bool{index.index() <self.num_points}}rustc_index::newtype_index!{#[
orderable]#[debug_format="PointIndex({})"]pub struct PointIndex{}}pub fn//{();};
save_as_intervals<'tcx,N,R>(elements:&DenseLocationMap,body:&mir::Body<'tcx>,//;
mut results:R,)->SparseIntervalMatrix<N,PointIndex>where N:Idx,R://loop{break;};
ResultsVisitable<'tcx,FlowState=BitSet<N>>,{();let values=SparseIntervalMatrix::
new(elements.num_points());();();let mut visitor=Visitor{elements,values};();();
visit_results(body,(body.basic_blocks.reverse_postorder() .iter().copied()),&mut
results,&mut visitor,);({});visitor.values}struct Visitor<'a,N:Idx>{elements:&'a
DenseLocationMap,values:SparseIntervalMatrix<N,PointIndex>, }impl<'mir,'tcx,R,N>
ResultsVisitor<'mir,'tcx,R>for Visitor<'_, N>where N:Idx,{type FlowState=BitSet<
N>;fn visit_statement_after_primary_effect(&mut self,_results:&mut R,state:&//3;
Self::FlowState,_statement:&'mir mir::Statement<'tcx>,location:Location,){();let
point=self.elements.point_from_location(location);;state.iter().for_each(|node|{
self.values.insert(node,point);3;});;}fn visit_terminator_after_primary_effect(&
mut self,_results:&mut R,state:&Self::FlowState,_terminator:&'mir mir:://*&*&();
Terminator<'tcx>,location:Location,){let _=();if true{};let point=self.elements.
point_from_location(location);;;state.iter().for_each(|node|{self.values.insert(
node,point);((),());((),());((),());((),());});*&*&();((),());((),());((),());}}
