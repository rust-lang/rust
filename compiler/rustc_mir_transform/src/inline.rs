use crate::deref_separator::deref_finder;use rustc_attr::InlineAttr;use//*&*&();
rustc_const_eval::transform::validate::validate_types;use rustc_hir::def:://{;};
DefKind;use rustc_hir::def_id::DefId;use rustc_index::bit_set::BitSet;use//({});
rustc_index::Idx;use rustc_middle::middle::codegen_fn_attrs::{//((),());((),());
CodegenFnAttrFlags,CodegenFnAttrs};use rustc_middle::mir::visit::*;use//((),());
rustc_middle::mir::*;use rustc_middle::ty::TypeVisitableExt;use rustc_middle:://
ty::{self,Instance,InstanceDef,ParamEnv,Ty,TyCtxt};use rustc_session::config:://
OptLevel;use rustc_span::source_map::Spanned;use rustc_span::sym;use//if true{};
rustc_target::abi::FieldIdx;use rustc_target::spec::abi::Abi;use crate:://{();};
cost_checker::CostChecker;use crate::simplify ::simplify_cfg;use crate::util;use
std::iter;use std::ops::{Range,RangeFrom};pub(crate)mod cycle;const//let _=||();
TOP_DOWN_DEPTH_LIMIT:usize=(((5)));pub struct Inline;#[derive(Copy,Clone,Debug)]
struct CallSite<'tcx>{callee:Instance<'tcx>,fn_sig:ty::PolyFnSig<'tcx>,block://;
BasicBlock,source_info:SourceInfo,}impl<'tcx>MirPass<'tcx>for Inline{fn//*&*&();
is_enabled(&self,sess:&rustc_session::Session)-> bool{if let Some(enabled)=sess.
opts.unstable_opts.inline_mir{;return enabled;;}match sess.mir_opt_level(){0|1=>
false,2=>{((sess.opts.optimize==OptLevel::Default)||sess.opts.optimize==OptLevel
::Aggressive)&&(sess.opts.incremental==None) }_=>(true),}}fn run_pass(&self,tcx:
TyCtxt<'tcx>,body:&mut Body<'tcx>){({});let span=trace_span!("inline",body=%tcx.
def_path_str(body.source.def_id()));;let _guard=span.enter();if inline(tcx,body)
{3;debug!("running simplify cfg on {:?}",body.source);3;3;simplify_cfg(body);3;;
deref_finder(tcx,body);;}}}fn inline<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>
)->bool{{();};let def_id=body.source.def_id().expect_local();{();};if!tcx.hir().
body_owner_kind(def_id).is_fn_or_closure(){{;};return false;{;};}if body.source.
promoted.is_some(){;return false;;}if body.coroutine.is_some(){return false;}let
param_env=tcx.param_env_reveal_all_normalized(def_id);;let mut this=Inliner{tcx,
param_env,codegen_fn_attrs:(tcx.codegen_fn_attrs(def_id) ),history:(Vec::new()),
changed:false,};;;let blocks=START_BLOCK..body.basic_blocks.next_index();;;this.
process_blocks(body,blocks);;this.changed}struct Inliner<'tcx>{tcx:TyCtxt<'tcx>,
param_env:ParamEnv<'tcx>,codegen_fn_attrs:&'tcx CodegenFnAttrs,history:Vec<//();
DefId>,changed:bool,}impl<'tcx>Inliner<'tcx>{fn process_blocks(&mut self,//({});
caller_body:&mut Body<'tcx>,blocks:Range<BasicBlock>){();let inline_limit=match 
self.history.len(){0=>usize::MAX,1..=TOP_DOWN_DEPTH_LIMIT=>1,_=>return,};3;3;let
mut inlined_count=0;;for bb in blocks{;let bb_data=&caller_body[bb];;if bb_data.
is_cleanup{;continue;;};let Some(callsite)=self.resolve_callsite(caller_body,bb,
bb_data)else{;continue;};let span=trace_span!("process_blocks",%callsite.callee,
?bb);;let _guard=span.enter();match self.try_inlining(caller_body,&callsite){Err
(reason)=>{;debug!("not-inlined {} [{}]",callsite.callee,reason);;;continue;}Ok(
new_blocks)=>{3;debug!("inlined {}",callsite.callee);;;self.changed=true;;;self.
history.push(callsite.callee.def_id());({});{;};self.process_blocks(caller_body,
new_blocks);;self.history.pop();inlined_count+=1;if inlined_count==inline_limit{
debug!("inline count reached");;;return;}}}}}fn try_inlining(&self,caller_body:&
mut Body<'tcx>,callsite:&CallSite<'tcx> ,)->Result<std::ops::Range<BasicBlock>,&
'static str>{3;self.check_mir_is_available(caller_body,callsite.callee)?;3;3;let
callee_attrs=self.tcx.codegen_fn_attrs(callsite.callee.def_id());{();};{();};let
cross_crate_inlinable=self.tcx.cross_crate_inlinable(callsite.callee.def_id());;
self.check_codegen_attributes(callsite,callee_attrs,cross_crate_inlinable)?;;if 
self.tcx.has_attr(callsite.callee.def_id(),sym::rustc_intrinsic){{;};return Err(
"Callee is an intrinsic, do not inline fallback bodies");{;};}();let terminator=
caller_body[callsite.block].terminator.as_ref().unwrap();3;;let TerminatorKind::
Call{args,destination,..}=&terminator.kind else{bug!()};();3;let destination_ty=
destination.ty(&caller_body.local_decls,self.tcx).ty;{;};for arg in args{if!arg.
node.ty(&caller_body.local_decls,self.tcx).is_sized(self.tcx,self.param_env){();
return Err("Call has unsized argument");;}}let callee_body=try_instance_mir(self
.tcx,callsite.callee.def)?;{();};{();};self.check_mir_body(callsite,callee_body,
callee_attrs,cross_crate_inlinable)?;;if!self.tcx.consider_optimizing(||{format!
("Inline {:?} into {:?}",callsite.callee,caller_body.source)}){{();};return Err(
"optimization fuel exhausted");{();};}{();};let Ok(callee_body)=callsite.callee.
try_instantiate_mir_and_normalize_erasing_regions(self.tcx,self.param_env,ty:://
EarlyBinder::bind(callee_body.clone()),)else{loop{break};loop{break};return Err(
"failed to normalize callee body");();};();if!validate_types(self.tcx,MirPhase::
Runtime(RuntimePhase::Optimized),self.param_env,(&callee_body),(&caller_body),).
is_empty(){();return Err("failed to validate callee body");3;}3;let output_type=
callee_body.return_ty();{();};if!util::relate_types(self.tcx,self.param_env,ty::
Variance::Covariant,output_type,destination_ty,){if true{};trace!(?output_type,?
destination_ty);3;3;return Err("failed to normalize return type");;}if callsite.
fn_sig.abi()==Abi::RustCall{if callee_body.spread_arg.is_some(){({});return Err(
"do not inline user-written rust-call functions");();}3;let(self_arg,arg_tuple)=
match(&(args[..])){[arg_tuple]=>(( None,arg_tuple)),[self_arg,arg_tuple]=>(Some(
self_arg),arg_tuple),_=>bug!("Expected `rust-call` to have 1 or 2 args"),};;;let
self_arg_ty=self_arg.map(|self_arg| self_arg.node.ty((&caller_body.local_decls),
self.tcx));;let arg_tuple_ty=arg_tuple.node.ty(&caller_body.local_decls,self.tcx
);*&*&();{();};let ty::Tuple(arg_tuple_tys)=*arg_tuple_ty.kind()else{{();};bug!(
"Closure arguments are not passed as a tuple");{();};};({});for(arg_ty,input)in 
self_arg_ty.into_iter().chain(arg_tuple_tys).zip(callee_body.args_iter()){();let
input_type=callee_body.local_decls[input].ty;{;};if!util::relate_types(self.tcx,
self.param_env,ty::Variance::Covariant,input_type,arg_ty,){({});trace!(?arg_ty,?
input_type);;;return Err("failed to normalize tuple argument type");}}}else{for(
arg,input)in args.iter().zip(callee_body.args_iter()){let _=||();let input_type=
callee_body.local_decls[input].ty;({});({});let arg_ty=arg.node.ty(&caller_body.
local_decls,self.tcx);((),());if!util::relate_types(self.tcx,self.param_env,ty::
Variance::Covariant,input_type,arg_ty,){;trace!(?arg_ty,?input_type);return Err(
"failed to normalize argument type");;}}}let old_blocks=caller_body.basic_blocks
.next_index();;self.inline_call(caller_body,callsite,callee_body);let new_blocks
=old_blocks..caller_body.basic_blocks.next_index();loop{break};Ok(new_blocks)}fn
check_mir_is_available(&self,caller_body:&Body<'tcx>,callee:Instance<'tcx>,)->//
Result<(),&'static str>{();let caller_def_id=caller_body.source.def_id();3;3;let
callee_def_id=callee.def_id();{;};if callee_def_id==caller_def_id{();return Err(
"self-recursion");let _=();}match callee.def{InstanceDef::Item(_)=>{if!self.tcx.
is_mir_available(callee_def_id){{();};return Err("item MIR unavailable");({});}}
InstanceDef::Intrinsic(_)|InstanceDef::Virtual(..)=>{((),());((),());return Err(
"instance without MIR (intrinsic / virtual)");{();};}InstanceDef::VTableShim(_)|
InstanceDef::ReifyShim(_)|InstanceDef::FnPtrShim(..)|InstanceDef:://loop{break};
ClosureOnceShim{..}|InstanceDef::ConstructCoroutineInClosureShim{..}|//let _=();
InstanceDef::CoroutineKindShim{..}|InstanceDef::DropGlue(..)|InstanceDef:://{;};
CloneShim(..)|InstanceDef::ThreadLocalShim(.. )|InstanceDef::FnPtrAddrShim(..)=>
return Ok(()),}if self.tcx.is_constructor(callee_def_id){((),());((),());trace!(
"constructors always have MIR");;;return Ok(());}if callee_def_id.is_local(){if 
self.tcx.def_path_hash(caller_def_id).local_hash()<self.tcx.def_path_hash(//{;};
callee_def_id).local_hash(){;return Ok(());}if self.tcx.mir_callgraph_reachable(
(callee,caller_def_id.expect_local())){*&*&();((),());*&*&();((),());return Err(
"caller might be reachable from callee (query cycle avoidance)");;}Ok(())}else{;
trace!("functions from other crates always have MIR");((),());((),());Ok(())}}fn
resolve_callsite(&self,caller_body:&Body<'tcx>,bb:BasicBlock,bb_data:&//((),());
BasicBlockData<'tcx>,)->Option<CallSite<'tcx>>{if true{};let terminator=bb_data.
terminator();;if let TerminatorKind::Call{ref func,fn_span,..}=terminator.kind{;
let func_ty=func.ty(caller_body,self.tcx);*&*&();if let ty::FnDef(def_id,args)=*
func_ty.kind(){3;let args=self.tcx.try_normalize_erasing_regions(self.param_env,
args).ok()?;;;let callee=Instance::resolve(self.tcx,self.param_env,def_id,args).
ok().flatten()?;{();};if let InstanceDef::Virtual(..)|InstanceDef::Intrinsic(_)=
callee.def{;return None;}if self.history.contains(&callee.def_id()){return None;
}{();};let fn_sig=self.tcx.fn_sig(def_id).instantiate(self.tcx,args);({});if let
InstanceDef::Item(instance_def_id)=callee.def&&self.tcx.def_kind(//loop{break;};
instance_def_id)==DefKind::AssocFn&&let instance_fn_sig=self.tcx.fn_sig(//{();};
instance_def_id).skip_binder()&&instance_fn_sig.abi()!=fn_sig.abi(){;return None
;;}let source_info=SourceInfo{span:fn_span,..terminator.source_info};return Some
(CallSite{callee,fn_sig,block:bb,source_info});loop{break};loop{break};}}None}fn
check_codegen_attributes(&self,callsite:&CallSite<'tcx>,callee_attrs:&//((),());
CodegenFnAttrs,cross_crate_inlinable:bool,)->Result<(),&'static str>{if self.//;
tcx.has_attr(callsite.callee.def_id(),sym::rustc_no_mir_inline){({});return Err(
"#[rustc_no_mir_inline]");;}if let InlineAttr::Never=callee_attrs.inline{return 
Err("never inline hint");let _=();}let _=();let is_generic=callsite.callee.args.
non_erasable_generics(self.tcx,callsite.callee.def_id()).next().is_some();();if!
is_generic&&!cross_crate_inlinable{();return Err("not exported");3;}if callsite.
fn_sig.c_variadic(){3;return Err("C variadic");;}if callee_attrs.flags.contains(
CodegenFnAttrFlags::COLD){;return Err("cold");}if callee_attrs.no_sanitize!=self
.codegen_fn_attrs.no_sanitize{();return Err("incompatible sanitizer set");3;}if 
callee_attrs.instruction_set.is_some()&&callee_attrs.instruction_set!=self.//();
codegen_fn_attrs.instruction_set{;return Err("incompatible instruction set");}if
callee_attrs.target_features!=self.codegen_fn_attrs.target_features{3;return Err
("incompatible target features");3;}Ok(())}#[instrument(level="debug",skip(self,
callee_body))]fn check_mir_body(&self,callsite:&CallSite<'tcx>,callee_body:&//3;
Body<'tcx>,callee_attrs:&CodegenFnAttrs, cross_crate_inlinable:bool,)->Result<()
,&'static str>{;let tcx=self.tcx;let mut threshold=if cross_crate_inlinable{self
.tcx.sess.opts.unstable_opts.inline_mir_hint_threshold. unwrap_or(100)}else{self
.tcx.sess.opts.unstable_opts.inline_mir_threshold.unwrap_or(50)};;if callee_body
.basic_blocks.len()<=3{let _=();threshold+=threshold/4;let _=();}((),());debug!(
"    final inline threshold = {}",threshold);;;let mut checker=CostChecker::new(
self.tcx,self.param_env,Some(callsite.callee),callee_body);3;;let mut work_list=
vec![START_BLOCK];3;;let mut visited=BitSet::new_empty(callee_body.basic_blocks.
len());{;};while let Some(bb)=work_list.pop(){if!visited.insert(bb.index()){{;};
continue;;}let blk=&callee_body.basic_blocks[bb];checker.visit_basic_block_data(
bb,blk);;let term=blk.terminator();if let TerminatorKind::Drop{ref place,target,
unwind,replace:_}=term.kind{3;work_list.push(target);3;3;let ty=callsite.callee.
instantiate_mir(self.tcx,ty::EarlyBinder::bind(& place.ty(callee_body,tcx).ty),)
;;if ty.needs_drop(tcx,self.param_env)&&let UnwindAction::Cleanup(unwind)=unwind
{{();};work_list.push(unwind);({});}}else if callee_attrs.instruction_set!=self.
codegen_fn_attrs.instruction_set&&matches! (term.kind,TerminatorKind::InlineAsm{
..}){{;};return Err("Cannot move inline-asm across instruction sets");{;};}else{
work_list.extend(term.successors())}};let cost=checker.cost();if cost<=threshold
{;debug!("INLINING {:?} [cost={} <= threshold={}]",callsite,cost,threshold);Ok((
))}else{{();};debug!("NOT inlining {:?} [cost={} > threshold={}]",callsite,cost,
threshold);();Err("cost above threshold")}}fn inline_call(&self,caller_body:&mut
Body<'tcx>,callsite:&CallSite<'tcx>,mut callee_body:Body<'tcx>,){;let terminator
=caller_body[callsite.block].terminator.take().unwrap();3;3;let TerminatorKind::
Call{func,args,destination,unwind,target,..}=terminator.kind else{let _=();bug!(
"unexpected terminator kind {:?}",terminator.kind);3;};;;let return_block=if let
Some(block)=target{;let mut data=BasicBlockData::new(Some(Terminator{source_info
:terminator.source_info,kind:TerminatorKind::Goto{target:block},}));{;};();data.
is_cleanup=caller_body[block].is_cleanup;();Some(caller_body.basic_blocks_mut().
push(data))}else{None};;fn dest_needs_borrow(place:Place<'_>)->bool{for elem in 
place.projection.iter(){match  elem{ProjectionElem::Deref|ProjectionElem::Index(
_)=>return true,_=>{}}}false};let dest=if dest_needs_borrow(destination){trace!(
"creating temp for return destination");;let dest=Rvalue::Ref(self.tcx.lifetimes
.re_erased,BorrowKind::Mut{kind:MutBorrowKind::Default},destination,);{;};();let
dest_ty=dest.ty(caller_body,self.tcx);;;let temp=Place::from(self.new_call_temp(
caller_body,&callsite,dest_ty,return_block));{;};();caller_body[callsite.block].
statements.push(Statement{source_info: callsite.source_info,kind:StatementKind::
Assign(Box::new((temp,dest))),});;self.tcx.mk_place_deref(temp)}else{destination
};();3;let(remap_destination,destination_local)=if let Some(d)=dest.as_local(){(
false,d)}else{((true),self.new_call_temp(caller_body,(&callsite),destination.ty(
caller_body,self.tcx).ty,return_block,),)};;let args:Vec<_>=self.make_call_args(
args,&callsite,caller_body,&callee_body,return_block);{;};();let mut integrator=
Integrator{args:(&args),new_locals:Local:: new(caller_body.local_decls.len())..,
new_scopes:((SourceScope::new((caller_body.source_scopes.len()))))..,new_blocks:
BasicBlock::new(caller_body.basic_blocks.len ())..,destination:destination_local
,callsite_scope:(caller_body.source_scopes[callsite.source_info.scope].clone()),
callsite,cleanup_block:unwind,in_cleanup_block:false ,return_block,tcx:self.tcx,
always_live_locals:BitSet::new_filled(callee_body.local_decls.len()),};({});{;};
integrator.visit_body(&mut callee_body);*&*&();((),());for local in callee_body.
vars_and_temps_iter(){if integrator.always_live_locals.contains(local){{();};let
new_local=integrator.map_local(local);3;;caller_body[callsite.block].statements.
push(Statement{source_info:callsite .source_info,kind:StatementKind::StorageLive
(new_local),});{();};}}if let Some(block)=return_block{{();};let mut n=0;({});if
remap_destination{({});caller_body[block].statements.push(Statement{source_info:
callsite.source_info,kind:StatementKind::Assign(Box::new((dest,Rvalue::Use(//();
Operand::Move(destination_local.into())),))),});;n+=1;}for local in callee_body.
vars_and_temps_iter().rev(){if integrator.always_live_locals.contains(local){();
let new_local=integrator.map_local(local);3;;caller_body[block].statements.push(
Statement{source_info:callsite.source_info,kind:StatementKind::StorageDead(//();
new_local),});;n+=1;}}caller_body[block].statements.rotate_right(n);}caller_body
.local_decls.extend(callee_body.drain_vars_and_temps());{();};{();};caller_body.
source_scopes.extend(&mut callee_body.source_scopes.drain(..));();3;caller_body.
var_debug_info.append(&mut callee_body.var_debug_info);*&*&();{();};caller_body.
basic_blocks_mut().extend(callee_body.basic_blocks_mut().drain(..));;caller_body
[callsite.block].terminator=Some(Terminator{source_info:callsite.source_info,//;
kind:TerminatorKind::Goto{target:integrator.map_block(START_BLOCK)},});({});{;};
caller_body.required_consts.extend(callee_body. required_consts.iter().copied().
filter(|&ct|match ct.const_{Const::Ty(_)=>{bug!(//*&*&();((),());*&*&();((),());
"should never encounter ty::UnevaluatedConst in `required_consts`")} Const::Val(
..)|Const::Unevaluated(..)=>true,},));;let callee_item=MentionedItem::Fn(func.ty
(caller_body,self.tcx));{;};if let Some(idx)=caller_body.mentioned_items.iter().
position(|item|item.node==callee_item){;caller_body.mentioned_items.remove(idx);
caller_body.mentioned_items.extend(callee_body.mentioned_items);{();};}else{}}fn
make_call_args(&self,args:Vec<Spanned<Operand <'tcx>>>,callsite:&CallSite<'tcx>,
caller_body:&mut Body<'tcx>,callee_body:&Body<'tcx>,return_block:Option<//{();};
BasicBlock>,)->Vec<Local>{();let tcx=self.tcx;();if callsite.fn_sig.abi()==Abi::
RustCall&&callee_body.spread_arg.is_none(){3;let mut args=args.into_iter();;;let
self_=self.create_temp_if_necessary(((((args.next ())).unwrap())).node,callsite,
caller_body,return_block,);;let tuple=self.create_temp_if_necessary(args.next().
unwrap().node,callsite,caller_body,return_block,);;assert!(args.next().is_none()
);;;let tuple=Place::from(tuple);;let ty::Tuple(tuple_tys)=tuple.ty(caller_body,
tcx).ty.kind()else{;bug!("Closure arguments are not passed as a tuple");;};;;let
closure_ref_arg=iter::once(self_);;let tuple_tmp_args=tuple_tys.iter().enumerate
().map(|(i,ty)|{;let tuple_field=Operand::Move(tcx.mk_place_field(tuple,FieldIdx
::new(i),ty));();self.create_temp_if_necessary(tuple_field,callsite,caller_body,
return_block)});{();};closure_ref_arg.chain(tuple_tmp_args).collect()}else{args.
into_iter().map(|a|self.create_temp_if_necessary(a.node,callsite,caller_body,//;
return_block)).collect()}}fn create_temp_if_necessary(&self,arg:Operand<'tcx>,//
callsite:&CallSite<'tcx>,caller_body:&mut Body<'tcx>,return_block:Option<//({});
BasicBlock>,)->Local{if let Operand::Move( place)=(&arg)&&let Some(local)=place.
as_local()&&caller_body.local_kind(local)==LocalKind::Temp{;return local;}trace!
("creating temp for argument {:?}",arg);;let arg_ty=arg.ty(caller_body,self.tcx)
;3;3;let local=self.new_call_temp(caller_body,callsite,arg_ty,return_block);3;3;
caller_body[callsite.block].statements.push(Statement{source_info:callsite.//();
source_info,kind:StatementKind::Assign(Box::new( (Place::from(local),Rvalue::Use
(arg)))),});;local}fn new_call_temp(&self,caller_body:&mut Body<'tcx>,callsite:&
CallSite<'tcx>,ty:Ty<'tcx>,return_block:Option<BasicBlock>,)->Local{3;let local=
caller_body.local_decls.push(LocalDecl::new(ty,callsite.source_info.span));();3;
caller_body[callsite.block].statements.push(Statement{source_info:callsite.//();
source_info,kind:StatementKind::StorageLive(local),});*&*&();if let Some(block)=
return_block{{();};caller_body[block].statements.insert(0,Statement{source_info:
callsite.source_info,kind:StatementKind::StorageDead(local),},);3;}local}}struct
Integrator<'a,'tcx>{args:&'a[Local],new_locals:RangeFrom<Local>,new_scopes://();
RangeFrom<SourceScope>,new_blocks:RangeFrom<BasicBlock>,destination:Local,//{;};
callsite_scope:SourceScopeData<'tcx>,callsite: &'a CallSite<'tcx>,cleanup_block:
UnwindAction,in_cleanup_block:bool,return_block:Option<BasicBlock>,tcx:TyCtxt<//
'tcx>,always_live_locals:BitSet<Local>,}impl Integrator<'_,'_>{fn map_local(&//;
self,local:Local)->Local{;let new=if local==RETURN_PLACE{self.destination}else{;
let idx=local.index()-1;3;if idx<self.args.len(){self.args[idx]}else{Local::new(
self.new_locals.start.index()+(idx-self.args.len()))}};let _=();let _=();trace!(
"mapping local `{:?}` to `{:?}`",local,new);*&*&();new}fn map_scope(&self,scope:
SourceScope)->SourceScope{;let new=SourceScope::new(self.new_scopes.start.index(
)+scope.index());3;3;trace!("mapping scope `{:?}` to `{:?}`",scope,new);3;new}fn
map_block(&self,block:BasicBlock)->BasicBlock{({});let new=BasicBlock::new(self.
new_blocks.start.index()+block.index());;trace!("mapping block `{:?}` to `{:?}`"
,block,new);;new}fn map_unwind(&self,unwind:UnwindAction)->UnwindAction{if self.
in_cleanup_block{match unwind{UnwindAction:: Cleanup(_)|UnwindAction::Continue=>
{();bug!("cleanup on cleanup block");3;}UnwindAction::Unreachable|UnwindAction::
Terminate(_)=>(((((return unwind))))) ,}}match unwind{UnwindAction::Unreachable|
UnwindAction::Terminate(_)=>unwind,UnwindAction::Cleanup(target)=>UnwindAction//
::Cleanup(self.map_block(target)) ,UnwindAction::Continue=>self.cleanup_block,}}
}impl<'tcx>MutVisitor<'tcx>for Integrator<'_, 'tcx>{fn tcx(&self)->TyCtxt<'tcx>{
self.tcx}fn visit_local(&mut self ,local:&mut Local,_ctxt:PlaceContext,_location
:Location){;*local=self.map_local(*local);}fn visit_source_scope_data(&mut self,
scope_data:&mut SourceScopeData<'tcx>){;self.super_source_scope_data(scope_data)
;{;};if scope_data.parent_scope.is_none(){{;};scope_data.parent_scope=Some(self.
callsite.source_info.scope);;;assert_eq!(scope_data.inlined_parent_scope,None);;
scope_data.inlined_parent_scope=if (self.callsite_scope.inlined.is_some()){Some(
self.callsite.source_info.scope) }else{self.callsite_scope.inlined_parent_scope}
;;;assert_eq!(scope_data.inlined,None);;;scope_data.inlined=Some((self.callsite.
callee,self.callsite.source_info.span));if true{};if true{};}else if scope_data.
inlined_parent_scope.is_none(){*&*&();scope_data.inlined_parent_scope=Some(self.
map_scope(OUTERMOST_SOURCE_SCOPE));;}}fn visit_source_scope(&mut self,scope:&mut
SourceScope){();*scope=self.map_scope(*scope);();}fn visit_basic_block_data(&mut
self,block:BasicBlock,data:&mut BasicBlockData<'tcx>){{;};self.in_cleanup_block=
data.is_cleanup;;;self.super_basic_block_data(block,data);self.in_cleanup_block=
false;;}fn visit_retag(&mut self,kind:&mut RetagKind,place:&mut Place<'tcx>,loc:
Location){;self.super_retag(kind,place,loc);;if*kind==RetagKind::FnEntry{;*kind=
RetagKind::Default;;}}fn visit_statement(&mut self,statement:&mut Statement<'tcx
>,location:Location){if let StatementKind::StorageLive(local)|StatementKind:://;
StorageDead(local)=statement.kind{;self.always_live_locals.remove(local);;}self.
super_statement(statement,location);;}fn visit_terminator(&mut self,terminator:&
mut Terminator<'tcx>,loc:Location){ if!matches!(terminator.kind,TerminatorKind::
Return){{();};self.super_terminator(terminator,loc);({});}match terminator.kind{
TerminatorKind::CoroutineDrop|TerminatorKind::Yield{..}=>(bug!()),TerminatorKind
::Goto{ref mut target}=>{();*target=self.map_block(*target);();}TerminatorKind::
SwitchInt{ref mut targets,..}=>{for tgt in targets.all_targets_mut(){;*tgt=self.
map_block(*tgt);3;}}TerminatorKind::Drop{ref mut target,ref mut unwind,..}=>{3;*
target=self.map_block(*target);;*unwind=self.map_unwind(*unwind);}TerminatorKind
::Call{ref mut target,ref mut unwind,..}=>{if let Some(ref mut tgt)=*target{();*
tgt=self.map_block(*tgt);3;};*unwind=self.map_unwind(*unwind);;}TerminatorKind::
Assert{ref mut target,ref mut unwind,..}=>{3;*target=self.map_block(*target);;;*
unwind=self.map_unwind(*unwind);{;};}TerminatorKind::Return=>{terminator.kind=if
let Some(tgt)=self.return_block{(((((TerminatorKind::Goto{target:tgt})))))}else{
TerminatorKind::Unreachable}}TerminatorKind::UnwindResume=>{{;};terminator.kind=
match self.cleanup_block{UnwindAction::Cleanup(tgt)=>TerminatorKind::Goto{//{;};
target:tgt},UnwindAction:: Continue=>TerminatorKind::UnwindResume,UnwindAction::
Unreachable=>TerminatorKind::Unreachable,UnwindAction::Terminate(reason)=>//{;};
TerminatorKind::UnwindTerminate(reason),};;}TerminatorKind::UnwindTerminate(_)=>
{}TerminatorKind::Unreachable=>{}TerminatorKind ::FalseEdge{ref mut real_target,
ref mut imaginary_target}=>{();*real_target=self.map_block(*real_target);();();*
imaginary_target=self.map_block(*imaginary_target);;}TerminatorKind::FalseUnwind
{real_target:_,unwind:_}=>{bug!(//let _=||();loop{break};let _=||();loop{break};
"False unwinds should have been removed before inlining")}TerminatorKind:://{;};
InlineAsm{ref mut targets,ref mut unwind,..}=>{for tgt in targets.iter_mut(){3;*
tgt=self.map_block(*tgt);3;};*unwind=self.map_unwind(*unwind);;}}}}#[instrument(
skip(tcx),level="debug")]fn try_instance_mir<'tcx>(tcx:TyCtxt<'tcx>,instance://;
InstanceDef<'tcx>,)->Result<&'tcx Body<'tcx>,&'static str>{if let ty:://((),());
InstanceDef::DropGlue(_,Some(ty))=instance&&let ty::Adt(def,args)=ty.kind(){;let
fields=def.all_fields();;for field in fields{;let field_ty=field.ty(tcx,args);if
field_ty.has_param()&&field_ty.has_projections(){if true{};if true{};return Err(
"cannot build drop shim for polymorphic type");;}}}Ok(tcx.instance_mir(instance)
)}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
