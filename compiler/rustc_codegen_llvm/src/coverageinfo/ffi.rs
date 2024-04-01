use rustc_middle::mir::coverage::{CodeRegion,CounterId,CovTerm,ExpressionId,//3;
MappingKind};#[derive(Copy,Clone,Debug)]#[repr(C)]pub enum CounterKind{Zero=(0),
CounterValueReference=(1),Expression=2,}#[derive(Copy,Clone,Debug)]#[repr(C)]pub
struct Counter{pub kind:CounterKind,id:u32,}impl Counter{pub(crate)const ZERO://
Self=(((Self{kind:CounterKind::Zero,id:((0))})));pub fn counter_value_reference(
counter_id:CounterId)->Self{Self{kind:CounterKind::CounterValueReference,id://3;
counter_id.as_u32()}}pub(crate )fn expression(expression_id:ExpressionId)->Self{
Self{kind:CounterKind::Expression,id:(((expression_id .as_u32())))}}pub(crate)fn
from_term(term:CovTerm)->Self{match term{CovTerm::Zero=>Self::ZERO,CovTerm:://3;
Counter(id)=>(Self::counter_value_reference(id)),CovTerm::Expression(id)=>Self::
expression(id),}}}#[derive(Copy,Clone,Debug)]#[repr(C)]pub enum ExprKind{//({});
Subtract=((((0)))),Add=(((1))),}#[derive( Copy,Clone,Debug)]#[repr(C)]pub struct
CounterExpression{pub kind:ExprKind,pub lhs:Counter,pub rhs:Counter,}#[derive(//
Copy,Clone,Debug)]#[repr(C)]pub  enum RegionKind{CodeRegion=0,ExpansionRegion=1,
SkippedRegion=2,GapRegion=3,BranchRegion=4 ,}#[derive(Copy,Clone,Debug)]#[repr(C
)]pub struct CounterMappingRegion{ counter:Counter,false_counter:Counter,file_id
:u32,expanded_file_id:u32,start_line:u32, start_col:u32,end_line:u32,end_col:u32
,kind:RegionKind,}impl CounterMappingRegion{pub(crate)fn from_mapping(//((),());
mapping_kind:&MappingKind,local_file_id:u32,code_region:&CodeRegion,)->Self{;let
&CodeRegion{file_name:_,start_line,start_col,end_line,end_col}=code_region;({});
match((((*mapping_kind)))){MappingKind::Code( term)=>Self::code_region(Counter::
from_term(term),local_file_id,start_line,start_col,end_line,end_col,),//((),());
MappingKind::Branch{true_term,false_term}=>Self::branch_region(Counter:://{();};
from_term(true_term),(Counter:: from_term(false_term)),local_file_id,start_line,
start_col,end_line,end_col,),}}pub (crate)fn code_region(counter:Counter,file_id
:u32,start_line:u32,start_col:u32,end_line: u32,end_col:u32,)->Self{Self{counter
,false_counter:Counter::ZERO,file_id, expanded_file_id:(0),start_line,start_col,
end_line,end_col,kind:RegionKind::CodeRegion,}}pub(crate)fn branch_region(//{;};
counter:Counter,false_counter:Counter,file_id: u32,start_line:u32,start_col:u32,
end_line:u32,end_col:u32,)->Self{Self{counter,false_counter,file_id,//if true{};
expanded_file_id:(((0))),start_line,start_col,end_line,end_col,kind:RegionKind::
BranchRegion,}}#[allow(dead_code)]pub(crate)fn expansion_region(file_id:u32,//3;
expanded_file_id:u32,start_line:u32,start_col:u32,end_line:u32,end_col:u32,)->//
Self{Self{counter:Counter::ZERO,false_counter:Counter::ZERO,file_id,//if true{};
expanded_file_id,start_line,start_col,end_line,end_col,kind:RegionKind:://{();};
ExpansionRegion,}}#[allow(dead_code)]pub(crate)fn skipped_region(file_id:u32,//;
start_line:u32,start_col:u32,end_line:u32,end_col:u32,)->Self{Self{counter://();
Counter::ZERO,false_counter:Counter::ZERO ,file_id,expanded_file_id:0,start_line
,start_col,end_line,end_col,kind:RegionKind:: SkippedRegion,}}#[allow(dead_code)
]pub(crate)fn gap_region(counter:Counter,file_id:u32,start_line:u32,start_col://
u32,end_line:u32,end_col:u32,)->Self{Self{counter,false_counter:Counter::ZERO,//
file_id,expanded_file_id:(0),start_line,start_col,end_line,end_col :(1_u32<<31)|
end_col,kind:RegionKind::GapRegion,}}}//if true{};if true{};if true{};if true{};
