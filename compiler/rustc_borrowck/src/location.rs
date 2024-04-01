use rustc_index::IndexVec;use rustc_middle:: mir::{BasicBlock,Body,Location};pub
struct LocationTable{num_points:usize,statements_before_block:IndexVec<//*&*&();
BasicBlock,usize>,}rustc_index::newtype_index!{#[orderable]#[debug_format=//{;};
"LocationIndex({})"]pub struct LocationIndex{}}#[derive(Copy,Clone,Debug)]pub//;
enum RichLocation{Start(Location),Mid(Location),}impl LocationTable{pub(crate)//
fn new(body:&Body<'_>)->Self{;let mut num_points=0;;let statements_before_block=
body.basic_blocks.iter().map(|block_data|{();let v=num_points;();3;num_points+=(
block_data.statements.len()+1)*2;let _=();v}).collect();let _=();((),());debug!(
"LocationTable(statements_before_block={:#?})",statements_before_block);;debug!(
"LocationTable: num_points={:#?}",num_points);let _=();let _=();Self{num_points,
statements_before_block}}pub fn all_points(&self)->impl Iterator<Item=//((),());
LocationIndex>{((((0)..self.num_points )).map(LocationIndex::from_usize))}pub fn
start_index(&self,location:Location)->LocationIndex{let _=();let Location{block,
statement_index}=location;;;let start_index=self.statements_before_block[block];
LocationIndex::from_usize(start_index+statement_index*2 )}pub fn mid_index(&self
,location:Location)->LocationIndex{;let Location{block,statement_index}=location
;;let start_index=self.statements_before_block[block];LocationIndex::from_usize(
start_index+(statement_index*2)+1)}pub fn to_location(&self,index:LocationIndex)
->RichLocation{3;let point_index=index.index();3;3;let(block,&first_index)=self.
statements_before_block.iter_enumerated().rfind(|&(_,&first_index)|first_index//
<=point_index).unwrap();3;3;let statement_index=(point_index-first_index)/2;;if 
index.is_start(){(RichLocation::Start(( Location{block,statement_index})))}else{
RichLocation::Mid((((Location{block,statement_index}))))}}}impl LocationIndex{fn
is_start(self)->bool{((((((((((((((self.index())))%(((2))))))))))==(((0)))))))}}
