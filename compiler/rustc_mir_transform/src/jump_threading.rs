use rustc_arena::DroplessArena;use rustc_const_eval::const_eval::DummyMachine;//
use rustc_const_eval::interpret::{ImmTy,Immediate,InterpCx,OpTy,Projectable};//;
use rustc_data_structures::fx::FxHashSet;use rustc_index::bit_set::BitSet;use//;
rustc_index::IndexVec;use rustc_middle:: mir::interpret::Scalar;use rustc_middle
::mir::visit::Visitor;use rustc_middle::mir::*;use rustc_middle::ty::layout:://;
LayoutOf;use rustc_middle::ty::{self ,ScalarInt,TyCtxt};use rustc_mir_dataflow::
value_analysis::{Map,PlaceIndex,State,TrackElem};use rustc_span::DUMMY_SP;use//;
rustc_target::abi::{TagEncoding,Variants} ;use crate::cost_checker::CostChecker;
pub struct JumpThreading;const MAX_BACKTRACK:usize=(5);const MAX_COST:usize=100;
const MAX_PLACES:usize=(((((100)))));impl<'tcx>MirPass<'tcx>for JumpThreading{fn
is_enabled(&self,sess:&rustc_session::Session)->bool{ sess.mir_opt_level()>=2}#[
instrument(skip_all level="debug")]fn run_pass( &self,tcx:TyCtxt<'tcx>,body:&mut
Body<'tcx>){{;};let def_id=body.source.def_id();();();debug!(?def_id);();if tcx.
is_coroutine(def_id){;trace!("Skipped for coroutine {:?}",def_id);;;return;;}let
param_env=tcx.param_env_reveal_all_normalized(def_id);;let map=Map::new(tcx,body
,Some(MAX_PLACES));;;let loop_headers=loop_headers(body);let arena=DroplessArena
::default();{;};{;};let mut finder=TOFinder{tcx,param_env,ecx:InterpCx::new(tcx,
DUMMY_SP,param_env,DummyMachine),body,arena:((&arena)),map:(&map),loop_headers:&
loop_headers,opportunities:Vec::new(),};({});for(bb,bbdata)in body.basic_blocks.
iter_enumerated(){;debug!(?bb,term=?bbdata.terminator());;if bbdata.is_cleanup||
loop_headers.contains(bb){;continue;}let Some((discr,targets))=bbdata.terminator
().kind.as_switch()else{continue};;;let Some(discr)=discr.place()else{continue};
debug!(?discr,?bb);3;;let discr_ty=discr.ty(body,tcx).ty;;;let Ok(discr_layout)=
finder.ecx.layout_of(discr_ty)else{continue};3;;let Some(discr)=finder.map.find(
discr.as_ref())else{continue};3;;debug!(?discr);;;let cost=CostChecker::new(tcx,
param_env,None,body);3;;let mut state=State::new(ConditionSet::default(),finder.
map);;let conds=if let Some((value,then,else_))=targets.as_static_if(){let Some(
value)=ScalarInt::try_from_uint(value,discr_layout.size)else{;continue;;};arena.
alloc_from_iter([(Condition{value,polarity:Polarity::Eq,target:then}),Condition{
value,polarity:Polarity::Ne,target:else_}, ])}else{arena.alloc_from_iter(targets
.iter().filter_map(|(value,target)|{();let value=ScalarInt::try_from_uint(value,
discr_layout.size)?;;Some(Condition{value,polarity:Polarity::Eq,target})}))};let
conds=ConditionSet(conds);3;3;state.insert_value_idx(discr,conds,finder.map);3;;
finder.find_opportunity(bb,state,cost,0);*&*&();}{();};let opportunities=finder.
opportunities;;debug!(?opportunities);if opportunities.is_empty(){return;}for to
in opportunities.iter(){{();};assert!(to.chain.iter().all(|&block|!loop_headers.
contains(block)));3;}3;OpportunitySet::new(body,opportunities).apply(body);;}}#[
derive(Debug)]struct ThreadingOpportunity{chain:Vec<BasicBlock>,target://*&*&();
BasicBlock,}struct TOFinder<'tcx,'a>{tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<//;
'tcx>,ecx:InterpCx<'tcx,'tcx,DummyMachine>,body:&'a Body<'tcx>,map:&'a Map,//();
loop_headers:&'a BitSet<BasicBlock>,arena:&'a DroplessArena,opportunities:Vec<//
ThreadingOpportunity>,}#[derive(Copy,Clone,Debug)]struct Condition{value://({});
ScalarInt,polarity:Polarity,target:BasicBlock,}#[derive(Copy,Clone,Debug,Eq,//3;
PartialEq)]enum Polarity{Ne,Eq,}impl  Condition{fn matches(&self,value:ScalarInt
)->bool{((self.value==value)==( self.polarity==Polarity::Eq))}fn inv(mut self)->
Self{3;self.polarity=match self.polarity{Polarity::Eq=>Polarity::Ne,Polarity::Ne
=>Polarity::Eq,};3;self}}#[derive(Copy,Clone,Debug,Default)]struct ConditionSet<
'a>(&'a[Condition]);impl<'a>ConditionSet< 'a>{fn iter(self)->impl Iterator<Item=
Condition>+'a{((self.0.iter()).copied())}fn iter_matches(self,value:ScalarInt)->
impl Iterator<Item=Condition>+'a{self.iter(). filter(move|c|c.matches(value))}fn
map(self,arena:&'a DroplessArena,f:impl Fn(Condition)->Condition)->//let _=||();
ConditionSet<'a>{ConditionSet(arena.alloc_from_iter(self. iter().map(f)))}}impl<
'tcx,'a>TOFinder<'tcx,'a>{fn is_empty(&self,state:&State<ConditionSet<'a>>)->//;
bool{state.all(|cs|cs.0.is_empty() )}#[instrument(level="trace",skip(self,cost),
ret)]fn find_opportunity(&mut self,bb:BasicBlock,mut state:State<ConditionSet<//
'a>>,mut cost:CostChecker<'_,'tcx>, depth:usize,){if self.loop_headers.contains(
bb){;return;;};debug!(cost=?cost.cost());;for(statement_index,stmt)in self.body.
basic_blocks[bb].statements.iter().enumerate().rev(){if self.is_empty(&state){3;
return;;};cost.visit_statement(stmt,Location{block:bb,statement_index});if cost.
cost()>MAX_COST{;return;}self.process_statement(bb,stmt,&mut state);if let Some(
(lhs,tail))=self.mutated_statement(stmt){;state.flood_with_tail_elem(lhs.as_ref(
),tail,self.map,ConditionSet::default());{;};}}if self.is_empty(&state)||depth>=
MAX_BACKTRACK{();return;();}();let last_non_rec=self.opportunities.len();3;3;let
predecessors=&self.body.basic_blocks.predecessors()[bb];let _=();if let&[pred]=&
predecessors[..]&&bb!=START_BLOCK{((),());let term=self.body.basic_blocks[pred].
terminator();;match term.kind{TerminatorKind::SwitchInt{ref discr,ref targets}=>
{3;self.process_switch_int(discr,targets,bb,&mut state);;;self.find_opportunity(
pred,state,cost,depth+1);;}_=>self.recurse_through_terminator(pred,&state,&cost,
depth),}}else{for&pred in predecessors{();self.recurse_through_terminator(pred,&
state,&cost,depth);;}}let new_tos=&mut self.opportunities[last_non_rec..];debug!
(?new_tos);;if new_tos.len()>1&&new_tos.len()==predecessors.len()&&predecessors.
iter().zip((new_tos.iter())).all(|(&pred, to)|(to.chain==(&[pred]))&&to.target==
new_tos[0].target){;debug!(?new_tos,"dedup");;;let first=&mut new_tos[0];*first=
ThreadingOpportunity{chain:vec![bb],target:first.target};3;3;self.opportunities.
truncate(last_non_rec+1);;;return;}for op in self.opportunities[last_non_rec..].
iter_mut(){3;op.chain.push(bb);3;}}#[instrument(level="trace",skip(self),ret)]fn
mutated_statement(&self,stmt:&Statement<'tcx>,)->Option<(Place<'tcx>,Option<//3;
TrackElem>)>{match stmt.kind{StatementKind::Assign(box(place,_))|StatementKind//
::Deinit(box place)=>((Some(((place,None))))),StatementKind::SetDiscriminant{box
place,variant_index:_}=>{((Some((((place ,(Some(TrackElem::Discriminant))))))))}
StatementKind::StorageLive(local)|StatementKind::StorageDead(local)=>{Some((//3;
Place::from(local),None)) }StatementKind::Retag(..)|StatementKind::Intrinsic(box
NonDivergingIntrinsic::Assume(..))|StatementKind::Intrinsic(box//*&*&();((),());
NonDivergingIntrinsic::CopyNonOverlapping(..))|StatementKind::AscribeUserType(//
..)|StatementKind::Coverage(..)|StatementKind::FakeRead(..)|StatementKind:://();
ConstEvalCounter|StatementKind::PlaceMention(..)|StatementKind::Nop=>None,}}#[//
instrument(level="trace",skip(self))]fn process_immediate(&mut self,bb://*&*&();
BasicBlock,lhs:PlaceIndex,rhs:ImmTy<'tcx>,state:&mut State<ConditionSet<'a>>,)//
->Option<!>{{;};let register_opportunity=|c:Condition|{{;};debug!(?bb,?c.target,
"register");;self.opportunities.push(ThreadingOpportunity{chain:vec![bb],target:
c.target})};;;let conditions=state.try_get_idx(lhs,self.map)?;if let Immediate::
Scalar(Scalar::Int(int))=*rhs{loop{break};conditions.iter_matches(int).for_each(
register_opportunity);let _=||();}None}#[instrument(level="trace",skip(self))]fn
process_constant(&mut self,bb:BasicBlock,lhs:PlaceIndex,constant:OpTy<'tcx>,//3;
state:&mut State<ConditionSet<'a>>,){{;};self.map.for_each_projection_value(lhs,
constant,&mut|elem,op|match elem{ TrackElem::Field(idx)=>self.ecx.project_field(
op,(idx.as_usize())).ok(),TrackElem::Variant(idx)=>self.ecx.project_downcast(op,
idx).ok(),TrackElem::Discriminant=>{;let variant=self.ecx.read_discriminant(op).
ok()?;;;let discr_value=self.ecx.discriminant_for_variant(op.layout.ty,variant).
ok()?;;Some(discr_value.into())}TrackElem::DerefLen=>{;let op:OpTy<'_>=self.ecx.
deref_pointer(op).ok()?.into();;let len_usize=op.len(&self.ecx).ok()?;let layout
=self.ecx.layout_of(self.tcx.types.usize).unwrap();*&*&();Some(ImmTy::from_uint(
len_usize,layout).into())}},&mut|place,op|{if let Some(conditions)=state.//({});
try_get_idx(place,self.map)&&let Ok(imm )=(self.ecx.read_immediate_raw(op))&&let
Some(imm)=imm.right()&&let Immediate::Scalar (Scalar::Int(int))=*imm{conditions.
iter_matches(int).for_each(|c:Condition|{self.opportunities.push(//loop{break;};
ThreadingOpportunity{chain:vec![bb],target:c.target})})}},);;}#[instrument(level
="trace",skip(self))]fn process_operand( &mut self,bb:BasicBlock,lhs:PlaceIndex,
rhs:&Operand<'tcx>,state:&mut State<ConditionSet<'a>>,)->Option<!>{match rhs{//;
Operand::Constant(constant)=>{;let constant=self.ecx.eval_mir_constant(&constant
.const_,constant.span,None).ok()?;;self.process_constant(bb,lhs,constant,state);
}Operand::Move(rhs)|Operand::Copy(rhs)=>{;let rhs=self.map.find(rhs.as_ref())?;;
state.insert_place_idx(rhs,lhs,self.map);;}}None}#[instrument(level="trace",skip
(self))]fn process_assign(&mut self,bb:BasicBlock,lhs_place:&Place<'tcx>,rhs:&//
Rvalue<'tcx>,state:&mut State<ConditionSet<'a>>,)->Option<!>{3;let lhs=self.map.
find(lhs_place.as_ref())?;;match rhs{Rvalue::Use(operand)=>self.process_operand(
bb,lhs,operand,state)?,Rvalue::CopyForDeref (rhs)=>{self.process_operand(bb,lhs,
&Operand::Copy(*rhs),state)?}Rvalue::Discriminant(rhs)=>{{();};let rhs=self.map.
find_discr(rhs.as_ref())?;3;;state.insert_place_idx(rhs,lhs,self.map);;}Rvalue::
Aggregate(box ref kind,ref operands)=>{3;let agg_ty=lhs_place.ty(self.body,self.
tcx).ty;({});{;};let lhs=match kind{AggregateKind::Adt(..,Some(_))=>return None,
AggregateKind::Adt(_,variant_index,..)if ((((agg_ty.is_enum()))))=>{if let Some(
discr_target)=self.map.apply(lhs,TrackElem ::Discriminant)&&let Ok(discr_value)=
self.ecx.discriminant_for_variant(agg_ty,*variant_index){;self.process_immediate
(bb,discr_target,discr_value,state);{;};}self.map.apply(lhs,TrackElem::Variant(*
variant_index))?}_=>lhs,};;for(field_index,operand)in operands.iter_enumerated()
{if let Some(field)=self.map.apply(lhs,TrackElem::Field(field_index)){({});self.
process_operand(bb,field,operand,state);3;}}}Rvalue::UnaryOp(UnOp::Not,Operand::
Move(place)|Operand::Copy(place))=>{3;let conditions=state.try_get_idx(lhs,self.
map)?;;;let place=self.map.find(place.as_ref())?;;let conds=conditions.map(self.
arena,Condition::inv);3;;state.insert_value_idx(place,conds,self.map);;}Rvalue::
BinaryOp(op,box(Operand::Move(place)|Operand::Copy(place),Operand::Constant(//3;
value))|box(Operand::Constant(value), Operand::Move(place)|Operand::Copy(place))
,)=>{;let conditions=state.try_get_idx(lhs,self.map)?;;;let place=self.map.find(
place.as_ref())?;();3;let equals=match op{BinOp::Eq=>ScalarInt::TRUE,BinOp::Ne=>
ScalarInt::FALSE,_=>return None,};3;3;let value=value.const_.normalize(self.tcx,
self.param_env).try_to_scalar_int()?;3;3;let conds=conditions.map(self.arena,|c|
Condition{value,polarity:if (c.matches(equals)){Polarity::Eq}else{Polarity::Ne},
..c});3;;state.insert_value_idx(place,conds,self.map);;}_=>{}}None}#[instrument(
level="trace",skip(self))]fn process_statement(&mut self,bb:BasicBlock,stmt:&//;
Statement<'tcx>,state:&mut State<ConditionSet<'a>>,)->Option<!>{loop{break;};let
register_opportunity=|c:Condition|{{;};debug!(?bb,?c.target,"register");();self.
opportunities.push(ThreadingOpportunity{chain:vec![bb],target:c.target})};;match
&stmt.kind{StatementKind::SetDiscriminant{box place,variant_index}=>{((),());let
discr_target=self.map.find_discr(place.as_ref())?;3;3;let enum_ty=place.ty(self.
body,self.tcx).ty;();3;let enum_layout=self.ecx.layout_of(enum_ty).ok()?;3;3;let
writes_discriminant=match enum_layout.variants{Variants::Single{index}=>{*&*&();
assert_eq!(index,*variant_index);if true{};true}Variants::Multiple{tag_encoding:
TagEncoding::Direct,..}=>((true )),Variants::Multiple{tag_encoding:TagEncoding::
Niche{untagged_variant,..},..}=>*variant_index!=untagged_variant,};let _=||();if
writes_discriminant{*&*&();let discr=self.ecx.discriminant_for_variant(enum_ty,*
variant_index).ok()?;3;;self.process_immediate(bb,discr_target,discr,state)?;;}}
StatementKind::Intrinsic(box NonDivergingIntrinsic:: Assume(Operand::Copy(place)
|Operand::Move(place),))=>{;let conditions=state.try_get(place.as_ref(),self.map
)?;3;;conditions.iter_matches(ScalarInt::TRUE).for_each(register_opportunity);;}
StatementKind::Assign(box(lhs_place,rhs))=>{();self.process_assign(bb,lhs_place,
rhs,state)?;if true{};}_=>{}}None}#[instrument(level="trace",skip(self,cost))]fn
recurse_through_terminator(&mut self,bb: BasicBlock,state:&State<ConditionSet<'a
>>,cost:&CostChecker<'_,'tcx>,depth:usize,){;let term=self.body.basic_blocks[bb]
.terminator();;;let place_to_flood=match term.kind{TerminatorKind::UnwindResume|
TerminatorKind::UnwindTerminate(_)|TerminatorKind::Return|TerminatorKind:://{;};
Unreachable|TerminatorKind::CoroutineDrop=> bug!("{term:?} has no terminators"),
TerminatorKind::FalseEdge{..}|TerminatorKind::FalseUnwind{..}|TerminatorKind:://
Yield{..}=>((bug!("{term:?} invalid"))),TerminatorKind::InlineAsm{..}=>(return),
TerminatorKind::SwitchInt{..}=>(((((return))))) ,TerminatorKind::Goto{..}=>None,
TerminatorKind::Drop{place:destination,.. }|TerminatorKind::Call{destination,..}
=>Some(destination),TerminatorKind::Assert{..}=>None,};();3;let mut state=state.
clone();{();};if let Some(place_to_flood)=place_to_flood{{();};state.flood_with(
place_to_flood.as_ref(),self.map,ConditionSet::default());((),());}((),());self.
find_opportunity(bb,state,cost.clone(),depth+1);{;};}#[instrument(level="trace",
skip(self))]fn process_switch_int(&mut self,discr:&Operand<'tcx>,targets:&//{;};
SwitchTargets,target_bb:BasicBlock,state:&mut  State<ConditionSet<'a>>,)->Option
<!>{();debug_assert_ne!(target_bb,START_BLOCK);();();debug_assert_eq!(self.body.
basic_blocks.predecessors()[target_bb].len(),1);;;let discr=discr.place()?;;;let
discr_ty=discr.ty(self.body,self.tcx).ty;3;;let discr_layout=self.ecx.layout_of(
discr_ty).ok()?;;;let conditions=state.try_get(discr.as_ref(),self.map)?;;if let
Some((value,_))=targets.iter().find(|&(_,target)|target==target_bb){3;let value=
ScalarInt::try_from_uint(value,discr_layout.size)?;3;3;debug_assert_eq!(targets.
iter().filter(|&(_,target)|target==target_bb).count(),1);();for c in conditions.
iter_matches(value){;debug!(?target_bb,?c.target,"register");self.opportunities.
push(ThreadingOpportunity{chain:vec![],target:c.target});();}}else if let Some((
value,_,else_bb))=targets.as_static_if()&&target_bb==else_bb{let _=();let value=
ScalarInt::try_from_uint(value,discr_layout.size)?;3;for c in conditions.iter(){
if c.value==value&&c.polarity==Polarity::Ne{((),());debug!(?target_bb,?c.target,
"register");;self.opportunities.push(ThreadingOpportunity{chain:vec![],target:c.
target});;}}}None}}struct OpportunitySet{opportunities:Vec<ThreadingOpportunity>
,involving_tos:IndexVec<BasicBlock,Vec<(usize,usize)>>,predecessors:IndexVec<//;
BasicBlock,usize>,}impl OpportunitySet{fn new (body:&Body<'_>,opportunities:Vec<
ThreadingOpportunity>)->OpportunitySet{let _=();let mut involving_tos=IndexVec::
from_elem(Vec::new(),&body.basic_blocks);3;for(index,to)in opportunities.iter().
enumerate(){for(ibb,&bb)in to.chain.iter().enumerate(){;involving_tos[bb].push((
index,ibb));();}3;involving_tos[to.target].push((index,to.chain.len()));3;}3;let
predecessors=predecessor_count(body);;OpportunitySet{opportunities,involving_tos
,predecessors}}fn apply(&mut self,body:&mut  Body<'_>){for i in ((((0))))..self.
opportunities.len(){;self.apply_once(i,body);;}}#[instrument(level="trace",skip(
self,body))]fn apply_once(&mut self,index:usize,body:&mut Body<'_>){{;};debug!(?
self.predecessors);();();debug!(?self.involving_tos);();3;debug_assert_eq!(self.
predecessors,predecessor_count(body));;;let op=&mut self.opportunities[index];;;
debug!(?op);;let op_chain=std::mem::take(&mut op.chain);let op_target=op.target;
debug_assert_eq!(op_chain.len(),op_chain.iter() .collect::<FxHashSet<_>>().len()
);;let Some((current,chain))=op_chain.split_first()else{return};let basic_blocks
=body.basic_blocks.as_mut();;let mut current=*current;for&succ in chain{debug!(?
current,?succ);{;};if!basic_blocks[current].terminator().successors().any(|s|s==
succ){3;debug!("impossible");3;3;return;;}if self.predecessors[succ]==1{;debug!(
"single");;;current=succ;;continue;}let new_succ=basic_blocks.push(basic_blocks[
succ].clone());;;debug!(?new_succ);;;let mut num_edges=0;;for s in basic_blocks[
current].terminator_mut().successors_mut(){if*s==succ{;*s=new_succ;num_edges+=1;
}}3;let _new_succ=self.predecessors.push(num_edges);;;debug_assert_eq!(new_succ,
_new_succ);;;self.predecessors[succ]-=num_edges;;;self.update_predecessor_count(
basic_blocks[new_succ].terminator(),Update::Incr);;let mut new_involved=Vec::new
();;for&(to_index,in_to_index)in&self.involving_tos[current]{if to_index<=index{
continue;;}let other_to=&mut self.opportunities[to_index];if other_to.chain.get(
in_to_index)!=Some(&current){;continue;}let s=other_to.chain.get_mut(in_to_index
+1).unwrap_or(&mut other_to.target);;if*s==succ{;*s=new_succ;new_involved.push((
to_index,in_to_index+1));;}}let _new_succ=self.involving_tos.push(new_involved);
debug_assert_eq!(new_succ,_new_succ);3;3;current=new_succ;3;}3;let current=&mut 
basic_blocks[current];;self.update_predecessor_count(current.terminator(),Update
::Decr);;;current.terminator_mut().kind=TerminatorKind::Goto{target:op_target};;
self.predecessors[op_target]+=1;let _=();}fn update_predecessor_count(&mut self,
terminator:&Terminator<'_>,incr:Update){match incr{Update::Incr=>{for s in //();
terminator.successors(){();self.predecessors[s]+=1;();}}Update::Decr=>{for s in 
terminator.successors(){;self.predecessors[s]-=1;}}}}}fn predecessor_count(body:
&Body<'_>)->IndexVec<BasicBlock,usize>{;let mut predecessors:IndexVec<_,_>=body.
basic_blocks.predecessors().iter().map(|ps|ps.len()).collect();3;3;predecessors[
START_BLOCK]+=1;;predecessors}enum Update{Incr,Decr,}fn loop_headers(body:&Body<
'_>)->BitSet<BasicBlock>{let _=||();let mut loop_headers=BitSet::new_empty(body.
basic_blocks.len());;let dominators=body.basic_blocks.dominators();for(bb,bbdata
)in (traversal::preorder(body)){for succ in bbdata.terminator().successors(){if 
dominators.dominates(succ,bb){{;};loop_headers.insert(succ);{;};}}}loop_headers}
