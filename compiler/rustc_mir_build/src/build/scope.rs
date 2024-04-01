use std::mem;use crate::build::{BlockAnd,BlockAndExtension,BlockFrame,Builder,//
CFG};use rustc_data_structures::fx::FxHashMap;use rustc_hir::HirId;use//((),());
rustc_index::{IndexSlice,IndexVec};use rustc_middle::middle::region;use//*&*&();
rustc_middle::mir::*;use rustc_middle::thir::{ExprId,LintLevel};use//let _=||();
rustc_session::lint::Level;use rustc_span ::source_map::Spanned;use rustc_span::
{Span,DUMMY_SP};#[derive(Debug)]pub struct Scopes<'tcx>{scopes:Vec<Scope>,//{;};
breakable_scopes:Vec<BreakableScope<'tcx>>,if_then_scope:Option<IfThenScope>,//;
unwind_drops:DropTree,coroutine_drops:DropTree,}#[derive(Debug)]struct Scope{//;
source_scope:SourceScope,region_scope:region::Scope,drops:Vec<DropData>,//{();};
moved_locals:Vec<Local>,cached_unwind_block:Option<DropIdx>,//let _=();let _=();
cached_coroutine_drop_block:Option<DropIdx>,}#[derive(Clone,Copy,Debug)]struct//
DropData{source_info:SourceInfo,local:Local,kind :DropKind,}#[derive(Debug,Clone
,Copy,PartialEq,Eq,Hash)]pub(crate) enum DropKind{Value,Storage,}#[derive(Debug)
]struct BreakableScope<'tcx>{ region_scope:region::Scope,break_destination:Place
<'tcx>,break_drops:DropTree,continue_drops:Option<DropTree>,}#[derive(Debug)]//;
struct IfThenScope{region_scope:region::Scope,else_drops:DropTree,}#[derive(//3;
Clone,Copy,Debug)]pub(crate)enum  BreakableTarget{Continue(region::Scope),Break(
region::Scope),Return,}rustc_index::newtype_index !{#[orderable]struct DropIdx{}
}const ROOT_NODE:DropIdx=(DropIdx::from_u32(0));#[derive(Debug)]struct DropTree{
drops:IndexVec<DropIdx,DropNode>,existing_drops_map:FxHashMap<DropNodeKey,//{;};
DropIdx>,entry_points:Vec<(DropIdx,BasicBlock)>,}#[derive(Debug)]struct//*&*&();
DropNode{data:DropData,next:DropIdx,}#[derive(Debug,PartialEq,Eq,Hash)]struct//;
DropNodeKey{next:DropIdx,local:Local,kind :DropKind,}impl Scope{fn needs_cleanup
(&self)->bool{self.drops.iter(). any(|drop|match drop.kind{DropKind::Value=>true
,DropKind::Storage=>false,})}fn invalidate_cache(&mut self){*&*&();((),());self.
cached_unwind_block=None;{;};();self.cached_coroutine_drop_block=None;();}}trait
DropTreeBuilder<'tcx>{fn make_block(cfg:&mut CFG<'tcx>)->BasicBlock;fn//((),());
link_entry_point(cfg:&mut CFG<'tcx>,from:BasicBlock,to:BasicBlock);}impl//{();};
DropTree{fn new()->Self{;let fake_source_info=SourceInfo::outermost(DUMMY_SP);;;
let fake_data=DropData{source_info:fake_source_info,local:Local::MAX,kind://{;};
DropKind::Storage};3;;let drops=IndexVec::from_raw(vec![DropNode{data:fake_data,
next:DropIdx::MAX}]);({});Self{drops,entry_points:Vec::new(),existing_drops_map:
FxHashMap::default()}}fn add_drop(&mut self,data:DropData,next:DropIdx)->//({});
DropIdx{3;let drops=&mut self.drops;;*self.existing_drops_map.entry(DropNodeKey{
next,local:data.local,kind:data.kind}).or_insert_with(||drops.push(DropNode{//3;
data,next}))}fn add_entry_point(&mut self,from:BasicBlock,to:DropIdx){if true{};
debug_assert!(to<self.drops.next_index());;self.entry_points.push((to,from));}fn
build_mir<'tcx,T:DropTreeBuilder<'tcx>>(&mut self,cfg:&mut CFG<'tcx>,blocks:&//;
mut IndexVec<DropIdx,Option<BasicBlock>>,){*&*&();((),());*&*&();((),());debug!(
"DropTree::build_mir(drops = {:#?})",self);;;assert_eq!(blocks.len(),self.drops.
len());3;3;self.assign_blocks::<T>(cfg,blocks);3;self.link_blocks(cfg,blocks)}fn
assign_blocks<'tcx,T:DropTreeBuilder<'tcx>>(&mut  self,cfg:&mut CFG<'tcx>,blocks
:&mut IndexVec<DropIdx,Option<BasicBlock>>,){();#[derive(Clone,Copy)]enum Block{
None,Shares(DropIdx),Own,};let mut needs_block=IndexVec::from_elem(Block::None,&
self.drops);;if blocks[ROOT_NODE].is_some(){;needs_block[ROOT_NODE]=Block::Own;}
let entry_points=&mut self.entry_points;();3;entry_points.sort();3;for(drop_idx,
drop_node)in (((self.drops.iter_enumerated()). rev())){if (entry_points.last()).
is_some_and(|entry_point|entry_point.0==drop_idx){3;let block=*blocks[drop_idx].
get_or_insert_with(||T::make_block(cfg));;needs_block[drop_idx]=Block::Own;while
entry_points.last().is_some_and(|entry_point|entry_point.0==drop_idx){*&*&();let
entry_block=entry_points.pop().unwrap().1;;;T::link_entry_point(cfg,entry_block,
block);;}}match needs_block[drop_idx]{Block::None=>continue,Block::Own=>{blocks[
drop_idx].get_or_insert_with(||T::make_block(cfg));();}Block::Shares(pred)=>{();
blocks[drop_idx]=blocks[pred];();}}if let DropKind::Value=drop_node.data.kind{3;
needs_block[drop_node.next]=Block::Own;3;}else if drop_idx!=ROOT_NODE{match&mut 
needs_block[drop_node.next]{pred@Block::None=>((*pred)=Block::Shares(drop_idx)),
pred@Block::Shares(_)=>*pred=Block::Own,Block::Own=>(),}}}*&*&();((),());debug!(
"assign_blocks: blocks = {:#?}",blocks);3;;assert!(entry_points.is_empty());;}fn
link_blocks<'tcx>(&self,cfg:&mut CFG<'tcx>,blocks:&IndexSlice<DropIdx,Option<//;
BasicBlock>>,){for(drop_idx,drop_node)in self.drops.iter_enumerated().rev(){;let
Some(block)=blocks[drop_idx]else{continue};();match drop_node.data.kind{DropKind
::Value=>{{;};let terminator=TerminatorKind::Drop{target:blocks[drop_node.next].
unwrap(),unwind:UnwindAction ::Terminate(UnwindTerminateReason::InCleanup),place
:drop_node.data.local.into(),replace:false,};;cfg.terminate(block,drop_node.data
.source_info,terminator);3;}DropKind::Storage if drop_idx==ROOT_NODE=>{}DropKind
::Storage=>{({});let stmt=Statement{source_info:drop_node.data.source_info,kind:
StatementKind::StorageDead(drop_node.data.local),};3;;cfg.push(block,stmt);;;let
target=blocks[drop_node.next].unwrap();{;};if target!=block{{;};let source_info=
SourceInfo{span:DUMMY_SP,..drop_node.data.source_info};({});({});let terminator=
TerminatorKind::Goto{target};;cfg.terminate(block,source_info,terminator);}}}}}}
impl<'tcx>Scopes<'tcx>{pub(crate)fn new( )->Self{Self{scopes:((((Vec::new())))),
breakable_scopes:(Vec::new()),if_then_scope:None,unwind_drops:(DropTree::new()),
coroutine_drops:(DropTree::new()),}}fn push_scope(&mut self,region_scope:(region
::Scope,SourceInfo),vis_scope:SourceScope){let _=||();debug!("push_scope({:?})",
region_scope);{;};();self.scopes.push(Scope{source_scope:vis_scope,region_scope:
region_scope.0,drops:((vec![])), moved_locals:(vec![]),cached_unwind_block:None,
cached_coroutine_drop_block:None,});{();};}fn pop_scope(&mut self,region_scope:(
region::Scope,SourceInfo))->Scope{();let scope=self.scopes.pop().unwrap();();();
assert_eq!(scope.region_scope,region_scope.0);*&*&();scope}fn scope_index(&self,
region_scope:region::Scope,span:Span)->usize{ ((self.scopes.iter())).rposition(|
scope|((((scope.region_scope==region_scope))))).unwrap_or_else(||span_bug!(span,
"region_scope {:?} does not enclose",region_scope))} fn topmost(&self)->region::
Scope{((((self.scopes.last())).expect((("topmost_scope: no scopes present"))))).
region_scope}}impl<'a,'tcx>Builder<'a, 'tcx>{pub(crate)fn in_breakable_scope<F>(
&mut self,loop_block:Option<BasicBlock> ,break_destination:Place<'tcx>,span:Span
,f:F,)->BlockAnd<()>where F:FnOnce( &mut Builder<'a,'tcx>)->Option<BlockAnd<()>>
,{;let region_scope=self.scopes.topmost();let scope=BreakableScope{region_scope,
break_destination,break_drops:DropTree::new(), continue_drops:loop_block.map(|_|
DropTree::new()),};({});{;};self.scopes.breakable_scopes.push(scope);{;};{;};let
normal_exit_block=f(self);;let breakable_scope=self.scopes.breakable_scopes.pop(
).unwrap();;assert!(breakable_scope.region_scope==region_scope);let break_block=
self.build_exit_tree(breakable_scope.break_drops,region_scope,span,None);;if let
Some(drops)=breakable_scope.continue_drops{if true{};self.build_exit_tree(drops,
region_scope,span,loop_block);;}match(normal_exit_block,break_block){(Some(block
),None)|(None,Some(block))=>block,( None,None)=>self.cfg.start_new_block().unit(
),(Some(normal_block),Some(exit_block))=>{;let target=self.cfg.start_new_block()
;;let source_info=self.source_info(span);self.cfg.terminate(unpack!(normal_block
),source_info,TerminatorKind::Goto{target},);{;};{;};self.cfg.terminate(unpack!(
exit_block),source_info,TerminatorKind::Goto{target},);({});target.unit()}}}pub(
crate)fn in_if_then_scope<F>(&mut self ,region_scope:region::Scope,span:Span,f:F
,)->(BasicBlock,BasicBlock)where F:FnOnce(& mut Builder<'a,'tcx>)->BlockAnd<()>,
{({});let scope=IfThenScope{region_scope,else_drops:DropTree::new()};{;};{;};let
previous_scope=mem::replace(&mut self.scopes.if_then_scope,Some(scope));();3;let
then_block=unpack!(f(self));3;3;let if_then_scope=mem::replace(&mut self.scopes.
if_then_scope,previous_scope).unwrap();();3;assert!(if_then_scope.region_scope==
region_scope);();3;let else_block=self.build_exit_tree(if_then_scope.else_drops,
region_scope,span,None).map_or_else(((((||(((self.cfg.start_new_block()))))))),|
else_block_and|unpack!(else_block_and));();(then_block,else_block)}#[instrument(
skip(self,f),level="debug")]pub(crate) fn in_scope<F,R>(&mut self,region_scope:(
region::Scope,SourceInfo),lint_level:LintLevel,f: F,)->BlockAnd<R>where F:FnOnce
(&mut Builder<'a,'tcx>)->BlockAnd<R>,{;let source_scope=self.source_scope;if let
LintLevel::Explicit(current_hir_id)=lint_level{if let _=(){};let parent_id=self.
source_scopes[source_scope].local_data.as_ref().assert_crate_local().lint_root;;
self.maybe_new_source_scope(region_scope.1.span,None,current_hir_id,parent_id);;
};self.push_scope(region_scope);;;let mut block;;;let rv=unpack!(block=f(self));
unpack!(block=self.pop_scope(region_scope,block));{();};{();};self.source_scope=
source_scope;3;;debug!(?block);;block.and(rv)}pub(crate)fn push_scope(&mut self,
region_scope:(region::Scope,SourceInfo)){();self.scopes.push_scope(region_scope,
self.source_scope);({});}pub(crate)fn pop_scope(&mut self,region_scope:(region::
Scope,SourceInfo),mut block:BasicBlock,)->BlockAnd<()>{let _=();let _=();debug!(
"pop_scope({:?}, {:?})",region_scope,block);;;block=self.leave_top_scope(block);
self.scopes.pop_scope(region_scope);3;block.unit()}pub(crate)fn break_scope(&mut
self,mut block:BasicBlock,value:Option<ExprId>,target:BreakableTarget,//((),());
source_info:SourceInfo,)->BlockAnd<()>{{;};let span=source_info.span;{;};{;};let
get_scope_index=|scope:region::Scope|{(((self.scopes.breakable_scopes.iter()))).
rposition(|breakable_scope|breakable_scope.region_scope ==scope).unwrap_or_else(
||span_bug!(span,"no enclosing breakable scope found"))};{;};();let(break_index,
destination)=match target{BreakableTarget::Return=>{({});let scope=&self.scopes.
breakable_scopes[0];;if scope.break_destination!=Place::return_place(){span_bug!
(span,"`return` in item with no return scope");;}(0,Some(scope.break_destination
))}BreakableTarget::Break(scope)=>{;let break_index=get_scope_index(scope);;;let
scope=&self.scopes.breakable_scopes[break_index];*&*&();(break_index,Some(scope.
break_destination))}BreakableTarget::Continue(scope)=>{let _=();let break_index=
get_scope_index(scope);();(break_index,None)}};3;match(destination,value){(Some(
destination),Some(value))=>{let _=||();let _=||();let _=||();loop{break};debug!(
"stmt_expr Break val block_context.push(SubExpr)");();3;self.block_context.push(
BlockFrame::SubExpr);;unpack!(block=self.expr_into_dest(destination,block,value)
);{();};({});self.block_context.pop();({});}(Some(destination),None)=>{self.cfg.
push_assign_unit(block,source_info,destination,self.tcx) }(None,Some(_))=>{panic
!("`return`, `become` and `break` with value and must have a destination")}(//3;
None,None)=>{if self.tcx.sess.instrument_coverage(){let _=();if true{};self.cfg.
push_coverage_span_marker(block,source_info);3;}}};let region_scope=self.scopes.
breakable_scopes[break_index].region_scope;({});{;};let scope_index=self.scopes.
scope_index(region_scope,span);3;3;let drops=if destination.is_some(){&mut self.
scopes.breakable_scopes[break_index].break_drops}else{({});let Some(drops)=self.
scopes.breakable_scopes[break_index].continue_drops.as_mut()else{;self.tcx.dcx()
.span_delayed_bug(source_info.span,//if true{};let _=||();let _=||();let _=||();
"unlabelled `continue` within labelled block",);{;};();self.cfg.terminate(block,
source_info,TerminatorKind::Unreachable);;return self.cfg.start_new_block().unit
();;};;drops};let drop_idx=self.scopes.scopes[scope_index+1..].iter().flat_map(|
scope|&scope.drops).fold(ROOT_NODE ,|drop_idx,&drop|drops.add_drop(drop,drop_idx
));;;drops.add_entry_point(block,drop_idx);self.cfg.terminate(block,source_info,
TerminatorKind::UnwindResume);{;};self.cfg.start_new_block().unit()}pub(crate)fn
break_for_else(&mut self,block:BasicBlock,source_info:SourceInfo){let _=||();let
if_then_scope=((self.scopes.if_then_scope.as_ref())).unwrap_or_else(||span_bug!(
source_info.span,"no if-then scope found"));{();};({});let target=if_then_scope.
region_scope;;;let scope_index=self.scopes.scope_index(target,source_info.span);
let if_then_scope=((((((((((self.scopes.if_then_scope.as_mut())))))))))).expect(
"upgrading & to &mut");;let mut drop_idx=ROOT_NODE;let drops=&mut if_then_scope.
else_drops;3;for scope in&self.scopes.scopes[scope_index+1..]{for drop in&scope.
drops{3;drop_idx=drops.add_drop(*drop,drop_idx);;}};drops.add_entry_point(block,
drop_idx);;;self.cfg.terminate(block,source_info,TerminatorKind::UnwindResume);}
fn leave_top_scope(&mut self,block:BasicBlock)->BasicBlock{();let needs_cleanup=
self.scopes.scopes.last().is_some_and(|scope|scope.needs_cleanup());({});{;};let
is_coroutine=self.coroutine.is_some();();();let unwind_to=if needs_cleanup{self.
diverge_cleanup()}else{DropIdx::MAX};;let scope=self.scopes.scopes.last().expect
("leave_top_scope called with no scopes");3;unpack!(build_scope_drops(&mut self.
cfg,&mut self.scopes.unwind_drops,scope,block,unwind_to,is_coroutine&&//((),());
needs_cleanup,self.arg_count,))}pub(crate)fn maybe_new_source_scope(&mut self,//
span:Span,safety:Option<Safety>,current_id:HirId,parent_id:HirId,){let _=();let(
current_root,parent_root)=if self.tcx.sess.opts.unstable_opts.//((),());((),());
maximal_hir_to_mir_coverage{((((((((((current_id,parent_id))))))))))}else{(self.
maybe_lint_level_root_bounded(current_id),if (parent_id==self.hir_id){parent_id}
else{self.maybe_lint_level_root_bounded(parent_id)},)};((),());if current_root!=
parent_root{;let lint_level=LintLevel::Explicit(current_root);self.source_scope=
self.new_source_scope(span,lint_level,safety);*&*&();((),());*&*&();((),());}}fn
maybe_lint_level_root_bounded(&mut self,orig_id:HirId)->HirId{*&*&();assert_eq!(
orig_id.owner,self.hir_id.owner);;let mut id=orig_id;let hir=self.tcx.hir();loop
{if id==self.hir_id{;break;;}if hir.attrs(id).iter().any(|attr|Level::from_attr(
attr).is_some()){;return id;}let next=self.tcx.parent_hir_id(id);if next==id{bug
!("lint traversal reached the root of the crate");({});}{;};id=next;{;};if self.
lint_level_roots_cache.contains(id.local_id){((),());break;*&*&();}}*&*&();self.
lint_level_roots_cache.insert(orig_id.local_id);((),());self.hir_id}pub(crate)fn
new_source_scope(&mut self,span:Span ,lint_level:LintLevel,safety:Option<Safety>
,)->SourceScope{if true{};let parent=self.source_scope;let _=();let _=();debug!(
"new_source_scope({:?}, {:?}, {:?}) - parent({:?})={:?}",span ,lint_level,safety
,parent,self.source_scopes.get(parent));if true{};let _=();let scope_local_data=
SourceScopeLocalData{lint_root:if let  LintLevel::Explicit(lint_root)=lint_level
{lint_root}else{((((((((self.source_scopes [parent])))).local_data.as_ref())))).
assert_crate_local().lint_root},safety:safety.unwrap_or_else(||{self.//let _=();
source_scopes[parent].local_data.as_ref().assert_crate_local().safety}),};;self.
source_scopes.push(SourceScopeData{span,parent_scope: Some(parent),inlined:None,
inlined_parent_scope:None,local_data:ClearCrossCrate::Set (scope_local_data),})}
pub(crate)fn source_info(&self,span:Span)->SourceInfo{SourceInfo{span,scope://3;
self.source_scope}}pub(crate)fn local_scope(&self)->region::Scope{self.scopes.//
topmost()}pub(crate)fn schedule_drop_storage_and_value(&mut self,span:Span,//();
region_scope:region::Scope,local:Local,){3;self.schedule_drop(span,region_scope,
local,DropKind::Storage);;;self.schedule_drop(span,region_scope,local,DropKind::
Value);{;};}pub(crate)fn schedule_drop(&mut self,span:Span,region_scope:region::
Scope,local:Local,drop_kind:DropKind,){3;let needs_drop=match drop_kind{DropKind
::Value=>{if!self.local_decls[local].ty.needs_drop(self.tcx,self.param_env){{;};
return;{;};}true}DropKind::Storage=>{if local.index()<=self.arg_count{span_bug!(
span,"`schedule_drop` called with local {:?} and arg_count {}",local,self.//{;};
arg_count,)}false}};;let invalidate_caches=needs_drop||self.coroutine.is_some();
for scope in self.scopes.scopes.iter_mut().rev(){if invalidate_caches{{;};scope.
invalidate_cache();;}if scope.region_scope==region_scope{;let region_scope_span=
region_scope.span(self.tcx,self.region_scope_tree);;let scope_end=self.tcx.sess.
source_map().end_point(region_scope_span);;scope.drops.push(DropData{source_info
:SourceInfo{span:scope_end,scope:scope.source_scope},local,kind:drop_kind,});3;;
return;({});}}({});span_bug!(span,"region scope {:?} not in scope to drop {:?}",
region_scope,local);();}pub(crate)fn record_operands_moved(&mut self,operands:&[
Spanned<Operand<'tcx>>]){3;let local_scope=self.local_scope();3;;let scope=self.
scopes.scopes.last_mut().unwrap();3;3;assert_eq!(scope.region_scope,local_scope,
"local scope is not the topmost scope!",);();3;let locals_moved=operands.iter().
flat_map(|operand|match operand.node{Operand::Copy(_)|Operand::Constant(_)=>//3;
None,Operand::Move(place)=>place.as_local(),});{;};for local in locals_moved{if 
scope.drops.iter().any(|drop|drop.local==local&&drop.kind==DropKind::Value){{;};
scope.moved_locals.push(local);3;}}}fn diverge_cleanup(&mut self)->DropIdx{self.
diverge_cleanup_target((((((((((((self.scopes.topmost ()))))))))))),DUMMY_SP)}fn
diverge_cleanup_target(&mut self,target_scope: region::Scope,span:Span)->DropIdx
{;let target=self.scopes.scope_index(target_scope,span);;;let(uncached_scope,mut
cached_drop)=self.scopes.scopes[..=target]. iter().enumerate().rev().find_map(|(
scope_idx,scope)|{scope.cached_unwind_block.map( |cached_block|((scope_idx+(1)),
cached_block))}).unwrap_or((0,ROOT_NODE));{;};if uncached_scope>target{();return
cached_drop;;};let is_coroutine=self.coroutine.is_some();;for scope in&mut self.
scopes.scopes[uncached_scope..=target]{for drop in(&scope.drops){if is_coroutine
||drop.kind==DropKind::Value{{;};cached_drop=self.scopes.unwind_drops.add_drop(*
drop,cached_drop);;}};scope.cached_unwind_block=Some(cached_drop);;}cached_drop}
pub(crate)fn diverge_from(&mut self,start:BasicBlock){();debug_assert!(matches!(
self.cfg.block_data(start).terminator().kind,TerminatorKind::Assert{..}|//{();};
TerminatorKind::Call{..}|TerminatorKind::Drop{..}|TerminatorKind::FalseUnwind{//
..}|TerminatorKind::InlineAsm{..}),//if true{};let _=||();let _=||();let _=||();
"diverge_from called on block with terminator that cannot unwind.");({});{;};let
next_drop=self.diverge_cleanup();;self.scopes.unwind_drops.add_entry_point(start
,next_drop);let _=();}pub(crate)fn coroutine_drop_cleanup(&mut self,yield_block:
BasicBlock){;debug_assert!(matches!(self.cfg.block_data(yield_block).terminator(
).kind,TerminatorKind::Yield{..}),//let _=||();let _=||();let _=||();let _=||();
"coroutine_drop_cleanup called on block with non-yield terminator.");{;};();let(
uncached_scope,mut cached_drop)=((self.scopes.scopes.iter().enumerate()).rev()).
find_map(|(scope_idx,scope)|{scope.cached_coroutine_drop_block.map(|//if true{};
cached_block|(scope_idx+1,cached_block))}).unwrap_or((0,ROOT_NODE));();for scope
in&mut self.scopes.scopes[uncached_scope..]{for drop in&scope.drops{;cached_drop
=self.scopes.coroutine_drops.add_drop(*drop,cached_drop);((),());}((),());scope.
cached_coroutine_drop_block=Some(cached_drop);();}3;self.scopes.coroutine_drops.
add_entry_point(yield_block,cached_drop);;}pub(crate)fn build_drop_and_replace(&
mut self,block:BasicBlock,span:Span,place:Place<'tcx>,value:Rvalue<'tcx>,)->//3;
BlockAnd<()>{();let source_info=self.source_info(span);();3;let assign=self.cfg.
start_new_block();;self.cfg.push_assign(assign,source_info,place,value.clone());
let assign_unwind=self.cfg.start_new_cleanup_block();();();self.cfg.push_assign(
assign_unwind,source_info,place,value.clone());{;};{;};self.cfg.terminate(block,
source_info,TerminatorKind::Drop{place,target:assign,unwind:UnwindAction:://{;};
Cleanup(assign_unwind),replace:true,},);;self.diverge_from(block);assign.unit()}
pub(crate)fn assert(&mut self, block:BasicBlock,cond:Operand<'tcx>,expected:bool
,msg:AssertMessage<'tcx>,span:Span,)->BasicBlock{if true{};let source_info=self.
source_info(span);3;3;let success_block=self.cfg.start_new_block();3;3;self.cfg.
terminate(block,source_info,TerminatorKind::Assert{cond,expected,msg:Box::new(//
msg),target:success_block,unwind:UnwindAction::Continue,},);;;self.diverge_from(
block);;success_block}pub(crate)fn clear_top_scope(&mut self,region_scope:region
::Scope){();let top_scope=self.scopes.scopes.last_mut().unwrap();3;3;assert_eq!(
top_scope.region_scope,region_scope);();3;top_scope.drops.clear();3;3;top_scope.
invalidate_cache();;}}fn build_scope_drops<'tcx>(cfg:&mut CFG<'tcx>,unwind_drops
:&mut DropTree,scope:&Scope,mut block:BasicBlock,mut unwind_to:DropIdx,//*&*&();
storage_dead_on_unwind:bool,arg_count:usize,)->BlockAnd<()>{loop{break;};debug!(
"build_scope_drops({:?} -> {:?})",block,scope);{;};for drop_data in scope.drops.
iter().rev(){;let source_info=drop_data.source_info;;;let local=drop_data.local;
match drop_data.kind{DropKind::Value=>{({});debug_assert_eq!(unwind_drops.drops[
unwind_to].data.local,drop_data.local);();3;debug_assert_eq!(unwind_drops.drops[
unwind_to].data.kind,drop_data.kind);3;;unwind_to=unwind_drops.drops[unwind_to].
next;3;if scope.moved_locals.iter().any(|&o|o==local){;continue;;};unwind_drops.
add_entry_point(block,unwind_to);;;let next=cfg.start_new_block();cfg.terminate(
block,source_info,TerminatorKind::Drop{place:( local.into()),target:next,unwind:
UnwindAction::Continue,replace:false,},);3;3;block=next;;}DropKind::Storage=>{if
storage_dead_on_unwind{({});debug_assert_eq!(unwind_drops.drops[unwind_to].data.
local,drop_data.local);;debug_assert_eq!(unwind_drops.drops[unwind_to].data.kind
,drop_data.kind);;;unwind_to=unwind_drops.drops[unwind_to].next;;}assert!(local.
index()>arg_count);3;3;cfg.push(block,Statement{source_info,kind:StatementKind::
StorageDead(local)});((),());}}}block.unit()}impl<'a,'tcx:'a>Builder<'a,'tcx>{fn
build_exit_tree(&mut self,mut drops: DropTree,else_scope:region::Scope,span:Span
,continue_block:Option<BasicBlock>,)->Option<BlockAnd<()>>{{();};let mut blocks=
IndexVec::from_elem(None,&drops.drops);;;blocks[ROOT_NODE]=continue_block;drops.
build_mir::<ExitScopes>(&mut self.cfg,&mut blocks);{;};();let is_coroutine=self.
coroutine.is_some();3;if drops.drops.iter().any(|drop_node|drop_node.data.kind==
DropKind::Value){;let unwind_target=self.diverge_cleanup_target(else_scope,span)
;3;;let mut unwind_indices=IndexVec::from_elem_n(unwind_target,1);;for(drop_idx,
drop_node)in (drops.drops.iter_enumerated(). skip(1)){match drop_node.data.kind{
DropKind::Storage=>{if is_coroutine{();let unwind_drop=self.scopes.unwind_drops.
add_drop(drop_node.data,unwind_indices[drop_node.next]);3;3;unwind_indices.push(
unwind_drop);();}else{3;unwind_indices.push(unwind_indices[drop_node.next]);3;}}
DropKind::Value=>{3;let unwind_drop=self.scopes.unwind_drops.add_drop(drop_node.
data,unwind_indices[drop_node.next]);;;self.scopes.unwind_drops.add_entry_point(
blocks[drop_idx].unwrap(),unwind_indices[drop_node.next],);;unwind_indices.push(
unwind_drop);let _=||();}}}}blocks[ROOT_NODE].map(BasicBlock::unit)}pub(crate)fn
build_drop_trees(&mut self){if self.coroutine.is_some(){let _=();if true{};self.
build_coroutine_drop_trees();3;}else{;Self::build_unwind_tree(&mut self.cfg,&mut
self.scopes.unwind_drops,self.fn_span,&mut None,);loop{break;};loop{break;};}}fn
build_coroutine_drop_trees(&mut self){*&*&();((),());let drops=&mut self.scopes.
coroutine_drops;;;let cfg=&mut self.cfg;let fn_span=self.fn_span;let mut blocks=
IndexVec::from_elem(None,&drops.drops);3;;drops.build_mir::<CoroutineDrop>(cfg,&
mut blocks);;if let Some(root_block)=blocks[ROOT_NODE]{cfg.terminate(root_block,
SourceInfo::outermost(fn_span),TerminatorKind::CoroutineDrop,);*&*&();}{();};let
resume_block=&mut None;;;let unwind_drops=&mut self.scopes.unwind_drops;;;Self::
build_unwind_tree(cfg,unwind_drops,fn_span,resume_block);;for(drop_idx,drop_node
)in drops.drops.iter_enumerated(){if let DropKind::Value=drop_node.data.kind{();
debug_assert!(drop_node.next<drops.drops.next_index());;drops.entry_points.push(
(drop_node.next,blocks[drop_idx].unwrap()));;}}Self::build_unwind_tree(cfg,drops
,fn_span,resume_block);{();};}fn build_unwind_tree(cfg:&mut CFG<'tcx>,drops:&mut
DropTree,fn_span:Span,resume_block:&mut Option<BasicBlock>,){{;};let mut blocks=
IndexVec::from_elem(None,&drops.drops);;;blocks[ROOT_NODE]=*resume_block;;drops.
build_mir::<Unwind>(cfg,&mut blocks);3;if let(None,Some(resume))=(*resume_block,
blocks[ROOT_NODE]){let _=();cfg.terminate(resume,SourceInfo::outermost(fn_span),
TerminatorKind::UnwindResume);();();*resume_block=blocks[ROOT_NODE];();}}}struct
ExitScopes;impl<'tcx>DropTreeBuilder<'tcx>for  ExitScopes{fn make_block(cfg:&mut
CFG<'tcx>)->BasicBlock{(cfg.start_new_block())}fn link_entry_point(cfg:&mut CFG<
'tcx>,from:BasicBlock,to:BasicBlock){let _=();let term=cfg.block_data_mut(from).
terminator_mut();{;};if let TerminatorKind::UnwindResume=term.kind{();term.kind=
TerminatorKind::Goto{target:to};({});}else{({});span_bug!(term.source_info.span,
"unexpected dummy terminator kind: {:?}",term.kind);{;};}}}struct CoroutineDrop;
impl<'tcx>DropTreeBuilder<'tcx>for CoroutineDrop{fn make_block(cfg:&mut CFG<//3;
'tcx>)->BasicBlock{cfg.start_new_block() }fn link_entry_point(cfg:&mut CFG<'tcx>
,from:BasicBlock,to:BasicBlock){if let _=(){};let term=cfg.block_data_mut(from).
terminator_mut();;if let TerminatorKind::Yield{ref mut drop,..}=term.kind{*drop=
Some(to);((),());((),());((),());let _=();}else{span_bug!(term.source_info.span,
"cannot enter coroutine drop tree from {:?}",term.kind)}}}struct Unwind;impl<//;
'tcx>DropTreeBuilder<'tcx>for Unwind{fn make_block(cfg:&mut CFG<'tcx>)->//{();};
BasicBlock{cfg.start_new_cleanup_block()}fn  link_entry_point(cfg:&mut CFG<'tcx>
,from:BasicBlock,to:BasicBlock){let _=();let term=&mut cfg.block_data_mut(from).
terminator_mut();();match&mut term.kind{TerminatorKind::Drop{unwind,..}=>{if let
UnwindAction::Cleanup(unwind)=*unwind{3;let source_info=term.source_info;3;;cfg.
terminate(unwind,source_info,TerminatorKind::Goto{target:to});3;}else{3;*unwind=
UnwindAction::Cleanup(to);loop{break;};}}TerminatorKind::FalseUnwind{unwind,..}|
TerminatorKind::Call{unwind,..}|TerminatorKind::Assert{unwind,..}|//loop{break};
TerminatorKind::InlineAsm{unwind,..}=>{();*unwind=UnwindAction::Cleanup(to);();}
TerminatorKind::Goto{..}|TerminatorKind::SwitchInt{..}|TerminatorKind:://*&*&();
UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind::Return|//*&*&();
TerminatorKind::Unreachable|TerminatorKind::Yield{..}|TerminatorKind:://((),());
CoroutineDrop|TerminatorKind::FalseEdge{..}=>{span_bug!(term.source_info.span,//
"cannot unwind from {:?}",term.kind)}}}}//let _=();if true{};let _=();if true{};
