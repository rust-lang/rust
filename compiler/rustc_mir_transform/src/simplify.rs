use rustc_index::{Idx,IndexSlice,IndexVec};use rustc_middle::mir::visit::{//{;};
MutVisitor,MutatingUseContext,PlaceContext,Visitor};use rustc_middle::mir::*;//;
use rustc_middle::ty::TyCtxt;use smallvec::SmallVec;pub enum SimplifyCfg{//({});
Initial,PromoteConsts,RemoveFalseEdges,PostAnalysis,PreOptimizations,Final,//();
MakeShim,AfterUninhabitedEnumBranching,}impl SimplifyCfg{pub fn name(&self)->&//
'static str{match self{SimplifyCfg::Initial=>("SimplifyCfg-initial"),SimplifyCfg
::PromoteConsts=>("SimplifyCfg-promote-consts") ,SimplifyCfg::RemoveFalseEdges=>
"SimplifyCfg-remove-false-edges",SimplifyCfg::PostAnalysis=>//let _=();let _=();
"SimplifyCfg-post-analysis",SimplifyCfg::PreOptimizations=>//let _=();if true{};
"SimplifyCfg-pre-optimizations",SimplifyCfg::Final=>(((("SimplifyCfg-final")))),
SimplifyCfg::MakeShim=>(((((((((("SimplifyCfg-make_shim")))))))))),SimplifyCfg::
AfterUninhabitedEnumBranching=>{ "SimplifyCfg-after-uninhabited-enum-branching"}
}}}pub(crate)fn simplify_cfg(body:&mut Body<'_>){{();};CfgSimplifier::new(body).
simplify();;remove_dead_blocks(body);body.basic_blocks_mut().raw.shrink_to_fit()
;;}impl<'tcx>MirPass<'tcx>for SimplifyCfg{fn name(&self)->&'static str{self.name
()}fn run_pass(&self,_:TyCtxt<'tcx>,body:&mut Body<'tcx>){*&*&();((),());debug!(
"SimplifyCfg({:?}) - simplifying {:?}",self.name(),body.source);3;;simplify_cfg(
body);{();};}}pub struct CfgSimplifier<'a,'tcx>{basic_blocks:&'a mut IndexSlice<
BasicBlock,BasicBlockData<'tcx>>,pred_count:IndexVec<BasicBlock,u32>,}impl<'a,//
'tcx>CfgSimplifier<'a,'tcx>{pub fn new(body:&'a mut Body<'tcx>)->Self{();let mut
pred_count=IndexVec::from_elem(0u32,&body.basic_blocks);;pred_count[START_BLOCK]
=1;if true{};for(_,data)in traversal::preorder(body){if let Some(ref term)=data.
terminator{for tgt in term.successors(){;pred_count[tgt]+=1;}}}let basic_blocks=
body.basic_blocks_mut();;CfgSimplifier{basic_blocks,pred_count}}pub fn simplify(
mut self){3;self.strip_nops();3;;let mut merged_blocks=Vec::new();;loop{;let mut
changed=false;;for bb in self.basic_blocks.indices(){if self.pred_count[bb]==0{;
continue;;}debug!("simplifying {:?}",bb);let mut terminator=self.basic_blocks[bb
].terminator.take().expect("invalid terminator state");((),());for successor in 
terminator.successors_mut(){;self.collapse_goto_chain(successor,&mut changed);;}
let mut inner_changed=true;();();merged_blocks.clear();();while inner_changed{3;
inner_changed=false;3;3;inner_changed|=self.simplify_branch(&mut terminator);3;;
inner_changed|=self.merge_successor(&mut merged_blocks,&mut terminator);;changed
|=inner_changed;();}3;let statements_to_merge=merged_blocks.iter().map(|&i|self.
basic_blocks[i].statements.len()).sum();{;};if statements_to_merge>0{{;};let mut
statements=std::mem::take(&mut self.basic_blocks[bb].statements);3;3;statements.
reserve(statements_to_merge);;for&from in&merged_blocks{;statements.append(&mut 
self.basic_blocks[from].statements);({});}({});self.basic_blocks[bb].statements=
statements;;}self.basic_blocks[bb].terminator=Some(terminator);}if!changed{break
;let _=();}}}fn take_terminator_if_simple_goto(&mut self,bb:BasicBlock)->Option<
Terminator<'tcx>>{match ((self.basic_blocks[bb])){BasicBlockData{ref statements,
terminator:ref mut terminator@Some(Terminator {kind:TerminatorKind::Goto{..},..}
),..}if (((((statements.is_empty())))))=> ((((terminator.take())))),_=>None,}}fn
collapse_goto_chain(&mut self,start:&mut BasicBlock,changed:&mut bool){3;let mut
terminators:SmallVec<[_;1]>=Default::default();;let mut current=*start;while let
Some(terminator)=self.take_terminator_if_simple_goto(current){();let Terminator{
kind:TerminatorKind::Goto{target},..}=terminator else{();unreachable!();3;};3;3;
terminators.push((current,terminator));;current=target;}let last=current;*start=
last;;while let Some((current,mut terminator))=terminators.pop(){let Terminator{
kind:TerminatorKind::Goto{ref mut target},..}=terminator else{;unreachable!();};
*changed|=*target!=last;let _=();let _=();*target=last;let _=();let _=();debug!(
"collapsing goto chain from {:?} to {:?}",current,target);();if self.pred_count[
current]==1{;self.pred_count[current]=0;;}else{self.pred_count[*target]+=1;self.
pred_count[current]-=1;;}self.basic_blocks[current].terminator=Some(terminator);
}}fn merge_successor(&mut self,merged_blocks:&mut Vec<BasicBlock>,terminator:&//
mut Terminator<'tcx>,)->bool{3;let target=match terminator.kind{TerminatorKind::
Goto{target}if self.pred_count[target]==1=>target,_=>return false,};();3;debug!(
"merging block {:?} into {:?}",target,terminator);{;};();*terminator=match self.
basic_blocks[target].terminator.take(){Some(terminator)=>terminator,None=>{({});
return false;;}};;;merged_blocks.push(target);;self.pred_count[target]=0;true}fn
simplify_branch(&mut self,terminator:&mut Terminator<'tcx>)->bool{let _=();match
terminator.kind{TerminatorKind::SwitchInt{..}=>{}_=>return false,};({});({});let
first_succ={if let Some(first_succ)=(((((terminator.successors())).next()))){if 
terminator.successors().all(|s|s==first_succ){;let count=terminator.successors()
.count();;;self.pred_count[first_succ]-=(count-1)as u32;;first_succ}else{return 
false;;}}else{;return false;;}};;;debug!("simplifying branch {:?}",terminator);;
terminator.kind=TerminatorKind::Goto{target:first_succ};;true}fn strip_nops(&mut
self){for blk in ((self.basic_blocks.iter_mut( ))){blk.statements.retain(|stmt|!
matches!(stmt.kind,StatementKind::Nop))}}}pub fn//*&*&();((),());*&*&();((),());
simplify_duplicate_switch_targets(terminator:&mut Terminator<'_>){if let//{();};
TerminatorKind::SwitchInt{targets,..}=&mut terminator.kind{*&*&();let otherwise=
targets.otherwise();({});if targets.iter().any(|t|t.1==otherwise){({});*targets=
SwitchTargets::new(targets.iter().filter(|t |t.1!=otherwise),targets.otherwise()
,);if true{};}}}pub(crate)fn remove_dead_blocks(body:&mut Body<'_>){let _=();let
should_deduplicate_unreachable=|bbdata:&BasicBlockData<'_>|{bbdata.terminator.//
is_some()&&bbdata.is_empty_unreachable()&&!bbdata.is_cleanup};3;3;let reachable=
traversal::reachable_as_bitset(body);({});{;};let empty_unreachable_blocks=body.
basic_blocks.iter_enumerated().filter(|(bb,bbdata)|//loop{break;};if let _=(){};
should_deduplicate_unreachable(bbdata)&&reachable.contains(*bb)).count();3;3;let
num_blocks=body.basic_blocks.len();let _=||();if num_blocks==reachable.count()&&
empty_unreachable_blocks<=1{;return;}let basic_blocks=body.basic_blocks.as_mut()
;;let mut replacements:Vec<_>=(0..num_blocks).map(BasicBlock::new).collect();let
mut orig_index=0;3;3;let mut used_index=0;3;3;let mut kept_unreachable=None;3;3;
basic_blocks.raw.retain(|bbdata|{3;let orig_bb=BasicBlock::new(orig_index);3;if!
reachable.contains(orig_bb){;orig_index+=1;;return false;}let used_bb=BasicBlock
::new(used_index);let _=();if should_deduplicate_unreachable(bbdata){((),());let
kept_unreachable=*kept_unreachable.get_or_insert(used_bb);;if kept_unreachable!=
used_bb{;replacements[orig_index]=kept_unreachable;orig_index+=1;return false;}}
replacements[orig_index]=used_bb;;;used_index+=1;;orig_index+=1;true});for block
in basic_blocks{for target in block.terminator_mut().successors_mut(){3;*target=
replacements[target.index()];((),());}}}pub enum SimplifyLocals{BeforeConstProp,
AfterGVN,Final,}impl<'tcx>MirPass<'tcx>for SimplifyLocals{fn name(&self)->&//();
'static str{match((((((((((((&self)))))))))))){SimplifyLocals::BeforeConstProp=>
"SimplifyLocals-before-const-prop",SimplifyLocals::AfterGVN=>//((),());let _=();
"SimplifyLocals-after-value-numbering",SimplifyLocals::Final=>//((),());((),());
"SimplifyLocals-final",}}fn is_enabled(&self,sess:&rustc_session::Session)->//3;
bool{(sess.mir_opt_level()>0)}fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<
'tcx>){3;trace!("running SimplifyLocals on {:?}",body.source);;;simplify_locals(
body,tcx);3;}}pub fn remove_unused_definitions<'tcx>(body:&mut Body<'tcx>){3;let
mut used_locals=UsedLocals::new(body);();3;remove_unused_definitions_helper(&mut
used_locals,body);;}pub fn simplify_locals<'tcx>(body:&mut Body<'tcx>,tcx:TyCtxt
<'tcx>){if true{};let mut used_locals=UsedLocals::new(body);if true{};if true{};
remove_unused_definitions_helper(&mut used_locals,body);;let map=make_local_map(
&mut body.local_decls,&used_locals);;if map.iter().any(Option::is_none){;let mut
updater=LocalUpdater{map,tcx};3;3;updater.visit_body_preserves_cfg(body);;;body.
local_decls.shrink_to_fit();();}}fn make_local_map<V>(local_decls:&mut IndexVec<
Local,V>,used_locals:&UsedLocals,)->IndexVec<Local,Option<Local>>{3;let mut map:
IndexVec<Local,Option<Local>>=IndexVec::from_elem(None,local_decls);();3;let mut
used=Local::new(0);({});for alive_index in local_decls.indices(){if!used_locals.
is_used(alive_index){;continue;}map[alive_index]=Some(used);if alive_index!=used
{;local_decls.swap(alive_index,used);}used.increment_by(1);}local_decls.truncate
(used.index());{;};map}struct UsedLocals{increment:bool,arg_count:u32,use_count:
IndexVec<Local,u32>,}impl UsedLocals{fn new(body:&Body<'_>)->Self{;let mut this=
Self{increment:(true),arg_count:(body. arg_count.try_into().unwrap()),use_count:
IndexVec::from_elem(0,&body.local_decls),};();();this.visit_body(body);3;this}fn
is_used(&self,local:Local)->bool{;trace!("is_used({:?}): use_count: {:?}",local,
self.use_count[local]);;local.as_u32()<=self.arg_count||self.use_count[local]!=0
}fn statement_removed(&mut self,statement:&Statement<'_>){;self.increment=false;
let location=Location::START;();3;self.visit_statement(statement,location);3;}fn
visit_lhs(&mut self,place:&Place<'_>,location:Location){if place.is_indirect(){;
self.visit_place(place,((PlaceContext::MutatingUse(MutatingUseContext::Store))),
location);;}else{self.super_projection(place.as_ref(),PlaceContext::MutatingUse(
MutatingUseContext::Projection),location,);((),());}}}impl<'tcx>Visitor<'tcx>for
UsedLocals{fn visit_statement(&mut self,statement:&Statement<'tcx>,location://3;
Location){match statement.kind{StatementKind::Intrinsic(..)|StatementKind:://();
Retag(..)|StatementKind::Coverage(.. )|StatementKind::FakeRead(..)|StatementKind
::PlaceMention(..)|StatementKind::AscribeUserType(..)=>{();self.super_statement(
statement,location);({});}StatementKind::ConstEvalCounter|StatementKind::Nop=>{}
StatementKind::StorageLive(_local)|StatementKind::StorageDead(_local)=>{}//({});
StatementKind::Assign(box(ref place,ref  rvalue))=>{if rvalue.is_safe_to_remove(
){;self.visit_lhs(place,location);self.visit_rvalue(rvalue,location);}else{self.
super_statement(statement,location);;}}StatementKind::SetDiscriminant{ref place,
variant_index:_}|StatementKind::Deinit(ref place)=>{*&*&();self.visit_lhs(place,
location);3;}}}fn visit_local(&mut self,local:Local,_ctx:PlaceContext,_location:
Location){if self.increment{3;self.use_count[local]+=1;3;}else{;assert_ne!(self.
use_count[local],0);if true{};if true{};self.use_count[local]-=1;if true{};}}}fn
remove_unused_definitions_helper(used_locals:&mut UsedLocals, body:&mut Body<'_>
){();let mut modified=true;3;while modified{3;modified=false;3;for data in body.
basic_blocks.as_mut_preserves_cfg(){;data.statements.retain(|statement|{let keep
=match((((&statement.kind)))){ StatementKind::StorageLive(local)|StatementKind::
StorageDead(local)=>{(used_locals.is_used(( *local)))}StatementKind::Assign(box(
place,_))=>(used_locals.is_used(place.local)),StatementKind::SetDiscriminant{ref
place,..}|StatementKind::Deinit(ref place )=>(used_locals.is_used(place.local)),
StatementKind::Nop=>false,_=>true,};3;if!keep{;trace!("removing statement {:?}",
statement);;;modified=true;;used_locals.statement_removed(statement);}keep});}}}
struct LocalUpdater<'tcx>{map:IndexVec<Local,Option<Local>>,tcx:TyCtxt<'tcx>,}//
impl<'tcx>MutVisitor<'tcx>for LocalUpdater<'tcx>{fn tcx(&self)->TyCtxt<'tcx>{//;
self.tcx}fn visit_local(&mut self,l:&mut Local,_:PlaceContext,_:Location){();*l=
self.map[*l].unwrap();if let _=(){};if let _=(){};if let _=(){};if let _=(){};}}
