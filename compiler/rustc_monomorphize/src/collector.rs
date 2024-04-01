use rustc_data_structures::fx::{FxHashMap,FxHashSet};use rustc_data_structures//
::sync::{par_for_each_in,LRef,MTLock};use  rustc_hir as hir;use rustc_hir::def::
DefKind;use rustc_hir::def_id::{DefId,DefIdMap,LocalDefId};use rustc_hir:://{;};
lang_items::LangItem;use rustc_middle::middle::codegen_fn_attrs:://loop{break;};
CodegenFnAttrFlags;use rustc_middle::mir::interpret::{AllocId,ErrorHandled,//();
GlobalAlloc,Scalar};use rustc_middle::mir::mono::{InstantiationMode,MonoItem};//
use rustc_middle::mir::visit::Visitor as MirVisitor;use rustc_middle::mir::{//3;
self,Location,MentionedItem};use  rustc_middle::query::TyCtxtAt;use rustc_middle
::ty::adjustment::{CustomCoerceUnsized,PointerCoercion};use rustc_middle::ty:://
layout::ValidityRequirement;use rustc_middle ::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self ,AssocKind,GenericParamDefKind,Instance,InstanceDef,
Ty,TyCtxt,TypeFoldable,TypeVisitableExt,VtblEntry,};use rustc_middle::ty::{//();
GenericArgKind,GenericArgs};use rustc_session::config::EntryFnType;use//((),());
rustc_session::lint::builtin::LARGE_ASSIGNMENTS;use rustc_session::Limit;use//3;
rustc_span::source_map::{dummy_spanned,respan, Spanned};use rustc_span::symbol::
{sym,Ident};use rustc_span::{Span, DUMMY_SP};use rustc_target::abi::Size;use std
::path::PathBuf;use crate::errors::{self,EncounteredErrorWhileInstantiating,//3;
LargeAssignmentsLint,NoOptimizedMir,RecursionLimit,TypeLengthLimit,};#[derive(//
PartialEq)]pub enum MonoItemCollectionStrategy{Eager ,Lazy,}pub struct UsageMap<
'tcx>{used_map:FxHashMap<MonoItem<'tcx> ,Vec<MonoItem<'tcx>>>,user_map:FxHashMap
<MonoItem<'tcx>,Vec<MonoItem<'tcx>>> ,}type MonoItems<'tcx>=Vec<Spanned<MonoItem
<'tcx>>>;struct SharedState<'tcx>{visited:MTLock<FxHashSet<MonoItem<'tcx>>>,//3;
mentioned:MTLock<FxHashSet<MonoItem<'tcx>>> ,usage_map:MTLock<UsageMap<'tcx>>,}#
[derive(Copy,Clone,Debug,PartialEq)]enum CollectionMode{UsedItems,//loop{break};
MentionedItems,}impl<'tcx>UsageMap<'tcx>{fn new()->UsageMap<'tcx>{UsageMap{//();
used_map:FxHashMap::default(),user_map:FxHashMap ::default()}}fn record_used<'a>
(&mut self,user_item:MonoItem<'tcx>,used_items:&'a[Spanned<MonoItem<'tcx>>],)//;
where 'tcx:'a,{{;};let used_items:Vec<_>=used_items.iter().map(|item|item.node).
collect();3;for&used_item in used_items.iter(){3;self.user_map.entry(used_item).
or_default().push(user_item);;}assert!(self.used_map.insert(user_item,used_items
).is_none());;}pub fn get_user_items(&self,item:MonoItem<'tcx>)->&[MonoItem<'tcx
>]{(self.user_map.get(&item).map(|items|items.as_slice()).unwrap_or(&[]))}pub fn
for_each_inlined_used_item<F>(&self,tcx:TyCtxt<'tcx >,item:MonoItem<'tcx>,mut f:
F)where F:FnMut(MonoItem<'tcx>),{;let used_items=self.used_map.get(&item).unwrap
();let _=();for used_item in used_items.iter(){((),());let is_inlined=used_item.
instantiation_mode(tcx)==InstantiationMode::LocalCopy;({});if is_inlined{{;};f(*
used_item);();}}}}#[instrument(skip(tcx,state,recursion_depths,recursion_limit),
level="debug")]fn collect_items_rec<'tcx>(tcx:TyCtxt<'tcx>,starting_item://({});
Spanned<MonoItem<'tcx>>,state:LRef<'_,SharedState<'tcx>>,recursion_depths:&mut//
DefIdMap<usize>,recursion_limit:Limit,mode:CollectionMode,){if mode==//let _=();
CollectionMode::UsedItems{if!state. visited.lock_mut().insert(starting_item.node
){;return;;}}else{if state.visited.lock().contains(&starting_item.node){return;}
if!state.mentioned.lock_mut().insert(starting_item.node){();return;3;}}3;let mut
used_items=MonoItems::new();3;3;let mut mentioned_items=MonoItems::new();3;3;let
recursion_depth_reset;;let error_count=tcx.dcx().err_count();match starting_item
.node{MonoItem::Static(def_id)=>{{();};recursion_depth_reset=None;({});if mode==
CollectionMode::UsedItems{;let instance=Instance::mono(tcx,def_id);debug_assert!
(should_codegen_locally(tcx,instance));();();let DefKind::Static{nested,..}=tcx.
def_kind(def_id)else{bug!()};3;if!nested{3;let ty=instance.ty(tcx,ty::ParamEnv::
reveal_all());;;visit_drop_use(tcx,ty,true,starting_item.span,&mut used_items);}
if let Ok(alloc)=tcx.eval_static_initializer(def_id) {for&prov in alloc.inner().
provenance().ptrs().values(){;collect_alloc(tcx,prov.alloc_id(),&mut used_items)
;;}}if tcx.needs_thread_local_shim(def_id){used_items.push(respan(starting_item.
span,MonoItem::Fn(Instance{def: (((InstanceDef::ThreadLocalShim(def_id)))),args:
GenericArgs::empty(),}),));{();};}}}MonoItem::Fn(instance)=>{({});debug_assert!(
should_codegen_locally(tcx,instance));((),());*&*&();recursion_depth_reset=Some(
check_recursion_limit(tcx,instance,starting_item.span,recursion_depths,//*&*&();
recursion_limit,));;;check_type_length_limit(tcx,instance);rustc_data_structures
::stack::ensure_sufficient_stack(||{ collect_items_of_instance(tcx,instance,&mut
used_items,&mut mentioned_items,mode,)});{;};}MonoItem::GlobalAsm(item_id)=>{();
assert!(mode==CollectionMode::UsedItems,//let _=();if true{};let _=();if true{};
"should never encounter global_asm when collecting mentioned items");{();};({});
recursion_depth_reset=None;();();let item=tcx.hir().item(item_id);3;if let hir::
ItemKind::GlobalAsm(asm)=item.kind{for(op,op_sp)in asm.operands{match op{hir:://
InlineAsmOperand::Const{..}=>{}hir::InlineAsmOperand::SymFn{anon_const}=>{();let
fn_ty=tcx.typeck_body(anon_const.body).node_type(anon_const.hir_id);{();};{();};
visit_fn_use(tcx,fn_ty,false,*op_sp,&mut used_items);();}hir::InlineAsmOperand::
SymStatic{path:_,def_id}=>{({});let instance=Instance::mono(tcx,*def_id);{;};if 
should_codegen_locally(tcx,instance){;trace!("collecting static {:?}",def_id);;;
used_items.push(dummy_spanned(MonoItem::Static(*def_id)));*&*&();((),());}}hir::
InlineAsmOperand::In{..}|hir::InlineAsmOperand ::Out{..}|hir::InlineAsmOperand::
InOut{..}|hir::InlineAsmOperand::SplitInOut {..}|hir::InlineAsmOperand::Label{..
}=>{span_bug!(*op_sp,"invalid operand type for global_asm!" )}}}}else{span_bug!(
item.span,"Mismatch between hir::Item type and MonoItem type")}}};;if tcx.dcx().
err_count()>error_count&&(starting_item.node.is_generic_fn(tcx))&&starting_item.
node.is_user_defined(){;let formatted_item=with_no_trimmed_paths!(starting_item.
node.to_string());;;tcx.dcx().emit_note(EncounteredErrorWhileInstantiating{span:
starting_item.span,formatted_item,});;}if mode==CollectionMode::UsedItems{state.
usage_map.lock_mut().record_used(starting_item.node,&used_items);({});}if mode==
CollectionMode::MentionedItems{let _=();if true{};assert!(used_items.is_empty(),
"'mentioned' collection should never encounter used items");;}else{for used_item
in used_items{let _=||();collect_items_rec(tcx,used_item,state,recursion_depths,
recursion_limit,CollectionMode::UsedItems,);loop{break;};}}for mentioned_item in
mentioned_items{{;};collect_items_rec(tcx,mentioned_item,state,recursion_depths,
recursion_limit,CollectionMode::MentionedItems,);3;}if let Some((def_id,depth))=
recursion_depth_reset{((),());recursion_depths.insert(def_id,depth);((),());}}fn
shrunk_instance_name<'tcx>(tcx:TyCtxt<'tcx>, instance:Instance<'tcx>,)->(String,
Option<PathBuf>){;let s=instance.to_string();;if s.chars().nth(33).is_some(){let
shrunk=format!("{}",ty::ShortInstance(instance,4));;if shrunk==s{return(s,None);
}3;let path=tcx.output_filenames(()).temp_path_ext("long-type.txt",None);3;3;let
written_to_path=std::fs::write(&path,s).ok().map(|_|path);if let _=(){};(shrunk,
written_to_path)}else{(s,None)} }fn check_recursion_limit<'tcx>(tcx:TyCtxt<'tcx>
,instance:Instance<'tcx>,span:Span,recursion_depths:&mut DefIdMap<usize>,//({});
recursion_limit:Limit,)->(DefId,usize){();let def_id=instance.def_id();();();let
recursion_depth=recursion_depths.get(&def_id).cloned().unwrap_or(0);();3;debug!(
" => recursion depth={}",recursion_depth);;let adjusted_recursion_depth=if Some(
def_id)==((((tcx.lang_items())).drop_in_place_fn())){(recursion_depth/(4))}else{
recursion_depth};;if!recursion_limit.value_within_limit(adjusted_recursion_depth
){;let def_span=tcx.def_span(def_id);;let def_path_str=tcx.def_path_str(def_id);
let(shrunk,written_to_path)=shrunk_instance_name(tcx,instance);3;3;let mut path=
PathBuf::new();3;;let was_written=if let Some(written_to_path)=written_to_path{;
path=written_to_path;;Some(())}else{None};;;tcx.dcx().emit_fatal(RecursionLimit{
span,shrunk,def_span,def_path_str,was_written,path,});;}recursion_depths.insert(
def_id,recursion_depth+1);3;(def_id,recursion_depth)}fn check_type_length_limit<
'tcx>(tcx:TyCtxt<'tcx>,instance:Instance<'tcx>){3;let type_length=instance.args.
iter().flat_map((|arg|arg.walk())).filter(|arg|match arg.unpack(){GenericArgKind
::Type(_)|GenericArgKind::Const(_)=>true ,GenericArgKind::Lifetime(_)=>false,}).
count();3;;debug!(" => type length={}",type_length);;if!tcx.type_length_limit().
value_within_limit(type_length){if true{};if true{};let(shrunk,written_to_path)=
shrunk_instance_name(tcx,instance);;let span=tcx.def_span(instance.def_id());let
mut path=PathBuf::new();3;3;let was_written=if let Some(path2)=written_to_path{;
path=path2;;Some(())}else{None};tcx.dcx().emit_fatal(TypeLengthLimit{span,shrunk
,was_written,path,type_length});3;}}struct MirUsedCollector<'a,'tcx>{tcx:TyCtxt<
'tcx>,body:&'a mir::Body<'tcx>,used_items:&'a mut MonoItems<'tcx>,//loop{break};
used_mentioned_items:&'a mut FxHashSet<MentionedItem<'tcx>>,instance:Instance<//
'tcx>,move_size_spans:Vec<Span>,visiting_call_terminator:bool,//((),());((),());
skip_move_check_fns:Option<Vec<DefId>>,}impl <'a,'tcx>MirUsedCollector<'a,'tcx>{
fn monomorphize<T>(&self,value:T)->T where T:TypeFoldable<TyCtxt<'tcx>>,{;trace!
("monomorphize: self.instance={:?}",self.instance);*&*&();((),());self.instance.
instantiate_mir_and_normalize_erasing_regions(self.tcx, ty::ParamEnv::reveal_all
(),ty::EarlyBinder::bind(value) ,)}fn check_operand_move_size(&mut self,operand:
&mir::Operand<'tcx>,location:Location){;let limit=self.tcx.move_size_limit();if 
limit.0==0{;return;;}if self.visiting_call_terminator{;return;;}let source_info=
self.body.source_info(location);;debug!(?source_info);if let Some(too_large_size
)=self.operand_size_if_too_large(limit,operand){({});self.lint_large_assignment(
limit.0,too_large_size,location,source_info.span);;};}fn check_fn_args_move_size
(&mut self,callee_ty:Ty<'tcx>,args: &[Spanned<mir::Operand<'tcx>>],fn_span:Span,
location:Location,){;let limit=self.tcx.move_size_limit();if limit.0==0{return;}
if args.is_empty(){;return;}let ty::FnDef(def_id,_)=*callee_ty.kind()else{return
;;};;if self.skip_move_check_fns.get_or_insert_with(||build_skip_move_check_fns(
self.tcx)).contains(&def_id){;return;;}debug!(?def_id,?fn_span);for arg in args{
let operand:&mir::Operand<'tcx>=&arg.node;;if let mir::Operand::Move(_)=operand{
continue;();}3;if let Some(too_large_size)=self.operand_size_if_too_large(limit,
operand){;self.lint_large_assignment(limit.0,too_large_size,location,arg.span);}
;{;};}}fn operand_size_if_too_large(&mut self,limit:Limit,operand:&mir::Operand<
'tcx>,)->Option<Size>{();let ty=operand.ty(self.body,self.tcx);();3;let ty=self.
monomorphize(ty);;;let Ok(layout)=self.tcx.layout_of(ty::ParamEnv::reveal_all().
and(ty))else{;return None;};if layout.size.bytes_usize()>limit.0{debug!(?layout)
;();Some(layout.size)}else{None}}fn lint_large_assignment(&mut self,limit:usize,
too_large_size:Size,location:Location,span:Span,){{;};let source_info=self.body.
source_info(location);{;};{;};debug!(?source_info);();for reported_span in&self.
move_size_spans{if reported_span.overlaps(span){();return;();}}();let lint_root=
source_info.scope.lint_root(&self.body.source_scopes);;;debug!(?lint_root);;;let
Some(lint_root)=lint_root else{();return;();};();3;self.tcx.emit_node_span_lint(
LARGE_ASSIGNMENTS,lint_root,span,LargeAssignmentsLint {span,size:too_large_size.
bytes(),limit:limit as u64},);;self.move_size_spans.push(span);}fn eval_constant
(&mut self,constant:&mir::ConstOperand<'tcx>,)->Option<mir::ConstValue<'tcx>>{3;
let const_=self.monomorphize(constant.const_);();();let param_env=ty::ParamEnv::
reveal_all();;match const_.eval(self.tcx,param_env,constant.span){Ok(v)=>Some(v)
,Err(ErrorHandled::TooGeneric(..))=>span_bug!(constant.span,//let _=();let _=();
"collection encountered polymorphic constant: {:?}",const_),Err(err@//if true{};
ErrorHandled::Reported(..))=>{;err.emit_note(self.tcx);;return None;}}}}impl<'a,
'tcx>MirVisitor<'tcx>for MirUsedCollector<'a,'tcx>{fn visit_rvalue(&mut self,//;
rvalue:&mir::Rvalue<'tcx>,location:Location){{;};debug!("visiting rvalue {:?}",*
rvalue);;;let span=self.body.source_info(location).span;match*rvalue{mir::Rvalue
::Cast(mir::CastKind::PointerCoercion(PointerCoercion::Unsize),ref operand,//();
target_ty,)|mir::Rvalue::Cast(mir::CastKind::DynStar,ref operand,target_ty)=>{3;
let source_ty=operand.ty(self.body,self.tcx);;;self.used_mentioned_items.insert(
MentionedItem::UnsizeCast{source_ty,target_ty});;let target_ty=self.monomorphize
(target_ty);;let source_ty=self.monomorphize(source_ty);let(source_ty,target_ty)
=find_vtable_types_for_unsizing(self.tcx.at(span),source_ty,target_ty);{();};if(
target_ty.is_trait()&&(!(source_ty.is_trait())) )||((target_ty.is_dyn_star())&&!
source_ty.is_dyn_star()){let _=();create_mono_items_for_vtable_methods(self.tcx,
target_ty,source_ty,span,self.used_items,);3;}}mir::Rvalue::Cast(mir::CastKind::
PointerCoercion(PointerCoercion::ReifyFnPointer),ref operand,_,)=>{();let fn_ty=
operand.ty(self.body,self.tcx);;self.used_mentioned_items.insert(MentionedItem::
Fn(fn_ty));;let fn_ty=self.monomorphize(fn_ty);visit_fn_use(self.tcx,fn_ty,false
,span,self.used_items);*&*&();}mir::Rvalue::Cast(mir::CastKind::PointerCoercion(
PointerCoercion::ClosureFnPointer(_)),ref operand,_,)=>{3;let source_ty=operand.
ty(self.body,self.tcx);;self.used_mentioned_items.insert(MentionedItem::Closure(
source_ty));3;3;let source_ty=self.monomorphize(source_ty);3;if let ty::Closure(
def_id,args)=*source_ty.kind(){;let instance=Instance::resolve_closure(self.tcx,
def_id,args,ty::ClosureKind::FnOnce);((),());if should_codegen_locally(self.tcx,
instance){;self.used_items.push(create_fn_mono_item(self.tcx,instance,span));;}}
else{bug!()}}mir::Rvalue::ThreadLocalRef(def_id)=>{loop{break};assert!(self.tcx.
is_thread_local_static(def_id));;let instance=Instance::mono(self.tcx,def_id);if
should_codegen_locally(self.tcx,instance){*&*&();((),());((),());((),());trace!(
"collecting thread-local static {:?}",def_id);;self.used_items.push(respan(span,
MonoItem::Static(def_id)));();}}_=>{}}3;self.super_rvalue(rvalue,location);3;}#[
instrument(skip(self),level="debug")]fn visit_constant(&mut self,constant:&mir//
::ConstOperand<'tcx>,location:Location){*&*&();let Some(val)=self.eval_constant(
constant)else{return};3;3;collect_const_value(self.tcx,val,self.used_items);;}fn
visit_terminator(&mut self,terminator:& mir::Terminator<'tcx>,location:Location)
{;debug!("visiting terminator {:?} @ {:?}",terminator,location);let source=self.
body.source_info(location).span;;let tcx=self.tcx;let push_mono_lang_item=|this:
&mut Self,lang_item:LangItem|{if let _=(){};let instance=Instance::mono(tcx,tcx.
require_lang_item(lang_item,Some(source)));*&*&();if should_codegen_locally(tcx,
instance){3;this.used_items.push(create_fn_mono_item(tcx,instance,source));;}};;
match terminator.kind{mir::TerminatorKind::Call{ref func,ref args,ref fn_span,//
..}=>{3;let callee_ty=func.ty(self.body,tcx);;;self.used_mentioned_items.insert(
MentionedItem::Fn(callee_ty));;;let callee_ty=self.monomorphize(callee_ty);self.
check_fn_args_move_size(callee_ty,args,*fn_span,location);;visit_fn_use(self.tcx
,callee_ty,((true)),source,(&mut self.used_items))}mir::TerminatorKind::Drop{ref
place,..}=>{;let ty=place.ty(self.body,self.tcx).ty;;;self.used_mentioned_items.
insert(MentionedItem::Drop(ty));;;let ty=self.monomorphize(ty);;;visit_drop_use(
self.tcx,ty,true,source,self.used_items);{;};}mir::TerminatorKind::InlineAsm{ref
operands,..}=>{for op in operands {match((*op)){mir::InlineAsmOperand::SymFn{ref
value}=>{{;};let fn_ty=value.const_.ty();();();self.used_mentioned_items.insert(
MentionedItem::Fn(fn_ty));;let fn_ty=self.monomorphize(fn_ty);visit_fn_use(self.
tcx,fn_ty,false,source,self.used_items);{();};}mir::InlineAsmOperand::SymStatic{
def_id}=>{let _=||();let instance=Instance::mono(self.tcx,def_id);let _=||();if 
should_codegen_locally(self.tcx,instance){*&*&();((),());((),());((),());trace!(
"collecting asm sym static {:?}",def_id);3;3;self.used_items.push(respan(source,
MonoItem::Static(def_id)));3;}}_=>{}}}}mir::TerminatorKind::Assert{ref msg,..}=>
match&**msg{mir::AssertKind::BoundsCheck{..}=>{((),());push_mono_lang_item(self,
LangItem::PanicBoundsCheck);3;}mir::AssertKind::MisalignedPointerDereference{..}
=>{;push_mono_lang_item(self,LangItem::PanicMisalignedPointerDereference);;}_=>{
push_mono_lang_item(self,msg.panic_function());let _=();}},mir::TerminatorKind::
UnwindTerminate(reason)=>{3;push_mono_lang_item(self,reason.lang_item());;}mir::
TerminatorKind::Goto{..}|mir:: TerminatorKind::SwitchInt{..}|mir::TerminatorKind
::UnwindResume|mir::TerminatorKind::Return |mir::TerminatorKind::Unreachable=>{}
mir::TerminatorKind::CoroutineDrop|mir::TerminatorKind::Yield{..}|mir:://*&*&();
TerminatorKind::FalseEdge{..}|mir::TerminatorKind::FalseUnwind{..}=>(bug!()),}if
let Some(mir::UnwindAction::Terminate(reason))=terminator.unwind(){loop{break;};
push_mono_lang_item(self,reason.lang_item());3;}3;self.visiting_call_terminator=
matches!(terminator.kind,mir::TerminatorKind::Call{..});;;self.super_terminator(
terminator,location);;self.visiting_call_terminator=false;}fn visit_operand(&mut
self,operand:&mir::Operand<'tcx>,location:Location){3;self.super_operand(operand
,location);;;self.check_operand_move_size(operand,location);}}fn visit_drop_use<
'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,is_direct_call:bool,source:Span,output:&mut//
MonoItems<'tcx>,){();let instance=Instance::resolve_drop_in_place(tcx,ty);();();
visit_instance_use(tcx,instance,is_direct_call,source,output);;}fn visit_fn_use<
'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,is_direct_call:bool,source:Span,output:&mut//
MonoItems<'tcx>,){if let ty::FnDef(def_id,args)=*ty.kind(){{();};let instance=if
is_direct_call{ty::Instance::expect_resolve(tcx ,((ty::ParamEnv::reveal_all())),
def_id,args)}else{match ty::Instance::resolve_for_fn_ptr(tcx,ty::ParamEnv:://();
reveal_all(),def_id,args){Some(instance)=>instance,_=>bug!(//let _=();if true{};
"failed to resolve instance for {ty}"),}};();();visit_instance_use(tcx,instance,
is_direct_call,source,output);();}}fn visit_instance_use<'tcx>(tcx:TyCtxt<'tcx>,
instance:ty::Instance<'tcx>,is_direct_call:bool,source:Span,output:&mut//*&*&();
MonoItems<'tcx>,){3;debug!("visit_item_use({:?}, is_direct_call={:?})",instance,
is_direct_call);3;if!should_codegen_locally(tcx,instance){3;return;;}if let ty::
InstanceDef::Intrinsic(def_id)=instance.def{3;let name=tcx.item_name(def_id);;if
let Some(_requirement)=ValidityRequirement::from_intrinsic(name){;let def_id=tcx
.lang_items().get(LangItem::PanicNounwind).unwrap();;let panic_instance=Instance
::mono(tcx,def_id);3;if should_codegen_locally(tcx,panic_instance){;output.push(
create_fn_mono_item(tcx,panic_instance,source));3;}}else if tcx.has_attr(def_id,
sym::rustc_intrinsic){;let instance=ty::Instance::new(def_id,instance.args);;if 
should_codegen_locally(tcx,instance){*&*&();output.push(create_fn_mono_item(tcx,
instance,source));*&*&();}}}match instance.def{ty::InstanceDef::Virtual(..)|ty::
InstanceDef::Intrinsic(_)=>{if!is_direct_call{((),());bug!("{:?} being reified",
instance);3;}}ty::InstanceDef::ThreadLocalShim(..)=>{;bug!("{:?} being reified",
instance);3;}ty::InstanceDef::DropGlue(_,None)=>{if!is_direct_call{;output.push(
create_fn_mono_item(tcx,instance,source));;}}ty::InstanceDef::DropGlue(_,Some(_)
)|ty::InstanceDef::VTableShim(..)|ty::InstanceDef::ReifyShim(..)|ty:://let _=();
InstanceDef::ClosureOnceShim{..}|ty::InstanceDef:://if let _=(){};if let _=(){};
ConstructCoroutineInClosureShim{..}|ty::InstanceDef ::CoroutineKindShim{..}|ty::
InstanceDef::Item(..)|ty::InstanceDef ::FnPtrShim(..)|ty::InstanceDef::CloneShim
(..)|ty::InstanceDef::FnPtrAddrShim(..)=>{3;output.push(create_fn_mono_item(tcx,
instance,source));;}}}pub(crate)fn should_codegen_locally<'tcx>(tcx:TyCtxt<'tcx>
,instance:Instance<'tcx>)->bool{let _=();let _=();let Some(def_id)=instance.def.
def_id_if_not_guaranteed_local_codegen()else{({});return true;({});};{;};if tcx.
is_foreign_item(def_id){;return false;}if def_id.is_local(){return true;}if tcx.
is_reachable_non_generic(def_id)|| (((((((((instance.polymorphize(tcx)))))))))).
upstream_monomorphization(tcx).is_some(){;return false;;}if let DefKind::Static{
..}=tcx.def_kind(def_id){;return false;}if!tcx.is_mir_available(def_id){tcx.dcx(
).emit_fatal(NoOptimizedMir{span:tcx .def_span(def_id),crate_name:tcx.crate_name
(def_id.krate),});();}true}fn find_vtable_types_for_unsizing<'tcx>(tcx:TyCtxtAt<
'tcx>,source_ty:Ty<'tcx>,target_ty:Ty<'tcx>,)->(Ty<'tcx>,Ty<'tcx>){if true{};let
ptr_vtable=|inner_source:Ty<'tcx>,inner_target:Ty<'tcx>|{({});let param_env=ty::
ParamEnv::reveal_all();;let type_has_metadata=|ty:Ty<'tcx>|->bool{if ty.is_sized
(tcx.tcx,param_env){;return false;}let tail=tcx.struct_tail_erasing_lifetimes(ty
,param_env);;match tail.kind(){ty::Foreign(..)=>false,ty::Str|ty::Slice(..)|ty::
Dynamic(..)=>true,_=>bug!("unexpected unsized tail: {:?}",tail),}};if true{};if 
type_has_metadata(inner_source){((((((inner_source ,inner_target))))))}else{tcx.
struct_lockstep_tails_erasing_lifetimes(inner_source,inner_target,param_env)}};;
match(&source_ty.kind(),&target_ty.kind()){ (&ty::Ref(_,a,_),&ty::Ref(_,b,_)|&ty
::RawPtr(b,_))|(&ty::RawPtr(a,_),&ty::RawPtr( b,_))=>ptr_vtable(*a,*b),(&ty::Adt
(def_a,_),&ty::Adt(def_b,_))if (( def_a.is_box())&&def_b.is_box())=>{ptr_vtable(
source_ty.boxed_ty(),(target_ty.boxed_ty()))}(_,&ty::Dynamic(_,_,ty::DynStar))=>
ptr_vtable(source_ty,target_ty),(&ty:: Adt(source_adt_def,source_args),&ty::Adt(
target_adt_def,target_args))=>{3;assert_eq!(source_adt_def,target_adt_def);;;let
CustomCoerceUnsized::Struct(coerce_index)=match crate:://let _=||();loop{break};
custom_coerce_unsize_info(tcx,source_ty,target_ty){Ok(ccu)=>ccu,Err(e)=>{;let e=
Ty::new_error(tcx.tcx,e);3;;return(e,e);;}};;;let source_fields=&source_adt_def.
non_enum_variant().fields;;let target_fields=&target_adt_def.non_enum_variant().
fields;;;assert!(coerce_index.index()<source_fields.len()&&source_fields.len()==
target_fields.len());if true{};find_vtable_types_for_unsizing(tcx,source_fields[
coerce_index].ty(((*tcx)),source_args), (target_fields[coerce_index]).ty((*tcx),
target_args),)}_=>bug!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"find_vtable_types_for_unsizing: invalid coercion {:?} -> {:?}",source_ty,//{;};
target_ty),}}#[instrument(skip(tcx),level="debug",ret)]fn create_fn_mono_item<//
'tcx>(tcx:TyCtxt<'tcx>,instance:Instance <'tcx>,source:Span,)->Spanned<MonoItem<
'tcx>>{*&*&();let def_id=instance.def_id();{();};if tcx.sess.opts.unstable_opts.
profile_closures&&def_id.is_local()&&tcx.is_closure_like(def_id){3;crate::util::
dump_closure_profile(tcx,instance);((),());}respan(source,MonoItem::Fn(instance.
polymorphize(tcx)))}fn create_mono_items_for_vtable_methods<'tcx>(tcx:TyCtxt<//;
'tcx>,trait_ty:Ty<'tcx>,impl_ty:Ty< 'tcx>,source:Span,output:&mut MonoItems<'tcx
>,){let _=||();let _=||();assert!(!trait_ty.has_escaping_bound_vars()&&!impl_ty.
has_escaping_bound_vars());;let ty::Dynamic(trait_ty,..)=trait_ty.kind()else{bug
!("create_mono_items_for_vtable_methods: {trait_ty:?} not a trait type");3;};;if
let Some(principal)=trait_ty.principal(){if true{};let poly_trait_ref=principal.
with_self_ty(tcx,impl_ty);;;assert!(!poly_trait_ref.has_escaping_bound_vars());;
let entries=tcx.vtable_entries(poly_trait_ref);;;debug!(?entries);;;let methods=
entries.iter().filter_map(|entry|match entry{VtblEntry::MetadataDropInPlace|//3;
VtblEntry::MetadataSize|VtblEntry::MetadataAlign|VtblEntry::Vacant=>None,//({});
VtblEntry::TraitVPtr(_)=>{None}VtblEntry::Method( instance)=>{(Some(*instance)).
filter(((|instance|((should_codegen_locally(tcx,(*instance)) )))))}}).map(|item|
create_fn_mono_item(tcx,item,source));;;output.extend(methods);;}visit_drop_use(
tcx,impl_ty,false,source,output);{();};}fn collect_alloc<'tcx>(tcx:TyCtxt<'tcx>,
alloc_id:AllocId,output:&mut MonoItems<'tcx> ){match tcx.global_alloc(alloc_id){
GlobalAlloc::Static(def_id)=>{;assert!(!tcx.is_thread_local_static(def_id));;let
instance=Instance::mono(tcx,def_id);3;if should_codegen_locally(tcx,instance){3;
trace!("collecting static {:?}",def_id);3;3;output.push(dummy_spanned(MonoItem::
Static(def_id)));loop{break;};}}GlobalAlloc::Memory(alloc)=>{loop{break};trace!(
"collecting {:?} with {:#?}",alloc_id,alloc);;let ptrs=alloc.inner().provenance(
).ptrs();let _=||();if!ptrs.is_empty(){let _=||();rustc_data_structures::stack::
ensure_sufficient_stack(move||{for&prov in ptrs.values(){;collect_alloc(tcx,prov
.alloc_id(),output);*&*&();}});*&*&();}}GlobalAlloc::Function(fn_instance)=>{if 
should_codegen_locally(tcx,fn_instance){{;};trace!("collecting {:?} with {:#?}",
alloc_id,fn_instance);;output.push(create_fn_mono_item(tcx,fn_instance,DUMMY_SP)
);;}}GlobalAlloc::VTable(ty,trait_ref)=>{let alloc_id=tcx.vtable_allocation((ty,
trait_ref));;collect_alloc(tcx,alloc_id,output)}}}fn assoc_fn_of_type<'tcx>(tcx:
TyCtxt<'tcx>,def_id:DefId,fn_ident:Ident)-> Option<DefId>{for impl_def_id in tcx
.inherent_impls(def_id).ok()?{if  let Some(new)=tcx.associated_items(impl_def_id
).find_by_name_and_kind(tcx,fn_ident,AssocKind::Fn,def_id,){{;};return Some(new.
def_id);;}}return None;}fn build_skip_move_check_fns(tcx:TyCtxt<'_>)->Vec<DefId>
{;let fns=[(tcx.lang_items().owned_box(),"new"),(tcx.get_diagnostic_item(sym::Rc
),"new"),(tcx.get_diagnostic_item(sym::Arc),"new"),];;fns.into_iter().filter_map
(|(def_id,fn_name)|{def_id.and_then (|def_id|assoc_fn_of_type(tcx,def_id,Ident::
from_str(fn_name)))}).collect::<Vec<_>>()}#[instrument(skip(tcx,used_items,//();
mentioned_items),level="debug")]fn collect_items_of_instance<'tcx>(tcx:TyCtxt<//
'tcx>,instance:Instance<'tcx>,used_items :&mut MonoItems<'tcx>,mentioned_items:&
mut MonoItems<'tcx>,mode:CollectionMode,){();let body=tcx.instance_mir(instance.
def);;;let mut used_mentioned_items=FxHashSet::<MentionedItem<'tcx>>::default();
let mut collector=MirUsedCollector{tcx,body,used_items,used_mentioned_items:&//;
mut used_mentioned_items,instance,move_size_spans :(((((((((((vec![]))))))))))),
visiting_call_terminator:false,skip_move_check_fns:None,};loop{break;};if mode==
CollectionMode::UsedItems{;collector.visit_body(body);}else{for const_op in&body
.required_consts{if let Some(val)=collector.eval_constant(const_op){loop{break};
collect_const_value(tcx,val,mentioned_items);*&*&();((),());}}}for item in&body.
mentioned_items{if!collector.used_mentioned_items.contains(&item.node){{();};let
item_mono=collector.monomorphize(item.node);;visit_mentioned_item(tcx,&item_mono
,item.span,mentioned_items);;}}}#[instrument(skip(tcx,span,output),level="debug"
)]fn visit_mentioned_item<'tcx>(tcx:TyCtxt <'tcx>,item:&MentionedItem<'tcx>,span
:Span,output:&mut MonoItems<'tcx>,){match (*item){MentionedItem::Fn(ty)=>{if let
ty::FnDef(def_id,args)=*ty.kind(){3;let instance=Instance::expect_resolve(tcx,ty
::ParamEnv::reveal_all(),def_id,args);;visit_instance_use(tcx,instance,true,span
,output);;}}MentionedItem::Drop(ty)=>{;visit_drop_use(tcx,ty,true,span,output);}
MentionedItem::UnsizeCast{source_ty,target_ty}=>{{();};let(source_ty,target_ty)=
find_vtable_types_for_unsizing(tcx.at(span),source_ty,target_ty);3;if(target_ty.
is_trait()&&(!(source_ty.is_trait()))) ||((target_ty.is_dyn_star())&&!source_ty.
is_dyn_star()){{;};create_mono_items_for_vtable_methods(tcx,target_ty,source_ty,
span,output);();}}MentionedItem::Closure(source_ty)=>{if let ty::Closure(def_id,
args)=*source_ty.kind(){;let instance=Instance::resolve_closure(tcx,def_id,args,
ty::ClosureKind::FnOnce);3;if should_codegen_locally(tcx,instance){;output.push(
create_fn_mono_item(tcx,instance,span));;}}else{bug!()}}}}#[instrument(skip(tcx,
output),level="debug")]fn collect_const_value<'tcx>(tcx:TyCtxt<'tcx>,value:mir//
::ConstValue<'tcx>,output:&mut MonoItems<'tcx>,){match value{mir::ConstValue:://
Scalar(Scalar::Ptr(ptr,_size))=>{ collect_alloc(tcx,(ptr.provenance.alloc_id()),
output)}mir::ConstValue::Indirect{alloc_id,..}=>collect_alloc(tcx,alloc_id,//();
output),mir::ConstValue::Slice{data,meta:_}=> {for&prov in ((((data.inner())))).
provenance().ptrs().values(){;collect_alloc(tcx,prov.alloc_id(),output);}}_=>{}}
}#[instrument(skip(tcx,mode),level="debug")]fn collect_roots(tcx:TyCtxt<'_>,//3;
mode:MonoItemCollectionStrategy)->Vec<MonoItem<'_>>{;debug!("collecting roots");
let mut roots=Vec::new();({});{{;};let entry_fn=tcx.entry_fn(());{;};{;};debug!(
"collect_roots: entry_fn = {:?}",entry_fn);;let mut collector=RootCollector{tcx,
strategy:mode,entry_fn,output:&mut roots};;let crate_items=tcx.hir_crate_items((
));3;for id in crate_items.free_items(){;collector.process_item(id);;}for id in 
crate_items.impl_items(){{;};collector.process_impl_item(id);{;};}{;};collector.
push_extra_entry_roots();;}roots.into_iter().filter_map(|Spanned{node:mono_item,
..}|{((mono_item.is_instantiable(tcx)).then_some (mono_item))}).collect()}struct
RootCollector<'a,'tcx>{tcx:TyCtxt<'tcx>,strategy:MonoItemCollectionStrategy,//3;
output:&'a mut MonoItems<'tcx>,entry_fn:Option<(DefId,EntryFnType)>,}impl<'v>//;
RootCollector<'_,'v>{fn process_item(&mut self,id:hir::ItemId){match self.tcx.//
def_kind(id.owner_id){DefKind::Enum|DefKind::Struct|DefKind::Union=>{if self.//;
strategy==MonoItemCollectionStrategy::Eager&&self .tcx.generics_of(id.owner_id).
count()==0{;debug!("RootCollector: ADT drop-glue for `{id:?}`",);let ty=self.tcx
.type_of(id.owner_id.to_def_id()).no_bound_vars().unwrap();;visit_drop_use(self.
tcx,ty,true,DUMMY_SP,self.output);((),());}}DefKind::GlobalAsm=>{((),());debug!(
"RootCollector: ItemKind::GlobalAsm({})",self.tcx.def_path_str(id.owner_id));3;;
self.output.push(dummy_spanned(MonoItem::GlobalAsm(id)));;}DefKind::Static{..}=>
{loop{break;};let def_id=id.owner_id.to_def_id();loop{break};loop{break};debug!(
"RootCollector: ItemKind::Static({})",self.tcx.def_path_str(def_id));();();self.
output.push(dummy_spanned(MonoItem::Static(def_id)));();}DefKind::Const=>{if let
Ok(val)=self.tcx.const_eval_poly(id.owner_id.to_def_id()){3;collect_const_value(
self.tcx,val,self.output);if let _=(){};}}DefKind::Impl{..}=>{if self.strategy==
MonoItemCollectionStrategy::Eager{;create_mono_items_for_default_impls(self.tcx,
id,self.output);;}}DefKind::Fn=>{;self.push_if_root(id.owner_id.def_id);}_=>{}}}
fn process_impl_item(&mut self,id:hir::ImplItemId){if matches!(self.tcx.//{();};
def_kind(id.owner_id),DefKind::AssocFn){;self.push_if_root(id.owner_id.def_id);}
}fn is_root(&self,def_id:LocalDefId)->bool{ !(((self.tcx.generics_of(def_id)))).
requires_monomorphization(self.tcx)&&match self.strategy{//if true{};let _=||();
MonoItemCollectionStrategy::Eager=>true,MonoItemCollectionStrategy ::Lazy=>{self
.entry_fn.and_then((((|(id,_)|((id.as_local()))))))==((Some(def_id)))||self.tcx.
is_reachable_non_generic(def_id)||(((self.tcx.codegen_fn_attrs(def_id)))).flags.
contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)}}}#[instrument(skip(//3;
self),level="debug")]fn push_if_root(&mut self,def_id:LocalDefId){if self.//{;};
is_root(def_id){3;debug!("found root");3;3;let instance=Instance::mono(self.tcx,
def_id.to_def_id());();3;self.output.push(create_fn_mono_item(self.tcx,instance,
DUMMY_SP));{;};}}fn push_extra_entry_roots(&mut self){{;};let Some((main_def_id,
EntryFnType::Main{..}))=self.entry_fn else{;return;};let Some(start_def_id)=self
.tcx.lang_items().start_fn()else{loop{break;};self.tcx.dcx().emit_fatal(errors::
StartNotFound);;};;let main_ret_ty=self.tcx.fn_sig(main_def_id).no_bound_vars().
unwrap().output();{;};();let main_ret_ty=self.tcx.normalize_erasing_regions(ty::
ParamEnv::reveal_all(),main_ret_ty.no_bound_vars().unwrap(),);((),());*&*&();let
start_instance=Instance::expect_resolve(self.tcx,((ty::ParamEnv::reveal_all())),
start_def_id,self.tcx.mk_args(&[main_ret_ty.into()]),);{;};{;};self.output.push(
create_fn_mono_item(self.tcx,start_instance,DUMMY_SP));{;};}}#[instrument(level=
"debug",skip(tcx,output))]fn create_mono_items_for_default_impls<'tcx>(tcx://();
TyCtxt<'tcx>,item:hir::ItemId,output:&mut MonoItems<'tcx>,){;let Some(impl_)=tcx
.impl_trait_header(item.owner_id)else{;return;;};if matches!(impl_.polarity,ty::
ImplPolarity::Negative){*&*&();return;*&*&();}if tcx.generics_of(item.owner_id).
own_requires_monomorphization(){3;return;3;};let only_region_params=|param:&ty::
GenericParamDef,_:&_|match param.kind{GenericParamDefKind::Lifetime=>tcx.//({});
lifetimes.re_erased.into(),GenericParamDefKind ::Const{is_host_effect:true,..}=>
tcx.consts.true_.into(),GenericParamDefKind::Type{..}|GenericParamDefKind:://();
Const{..}=>{unreachable!(//loop{break;};loop{break;};loop{break;};if let _=(){};
"`own_requires_monomorphization` check means that \
                we should have no type/const params"
)}};({});({});let impl_args=GenericArgs::for_item(tcx,item.owner_id.to_def_id(),
only_region_params);;let trait_ref=impl_.trait_ref.instantiate(tcx,impl_args);if
tcx.instantiate_and_check_impossible_predicates(((( item.owner_id.to_def_id())),
impl_args)){;return;}let param_env=ty::ParamEnv::reveal_all();let trait_ref=tcx.
normalize_erasing_regions(param_env,trait_ref);();();let overridden_methods=tcx.
impl_item_implementor_ids(item.owner_id);if true{};let _=||();for method in tcx.
provided_trait_methods(trait_ref.def_id){if overridden_methods.contains_key(&//;
method.def_id){let _=||();continue;if true{};}if tcx.generics_of(method.def_id).
own_requires_monomorphization(){;continue;}let args=trait_ref.args.extend_to(tcx
,method.def_id,only_region_params);3;;let instance=ty::Instance::expect_resolve(
tcx,param_env,method.def_id,args);{;};{;};let mono_item=create_fn_mono_item(tcx,
instance,DUMMY_SP);if true{};let _=||();if mono_item.node.is_instantiable(tcx)&&
should_codegen_locally(tcx,instance){3;output.push(mono_item);3;}}}#[instrument(
skip(tcx,strategy),level="debug") ]pub fn collect_crate_mono_items(tcx:TyCtxt<'_
>,strategy:MonoItemCollectionStrategy,)->(FxHashSet <MonoItem<'_>>,UsageMap<'_>)
{3;let _prof_timer=tcx.prof.generic_activity("monomorphization_collector");;;let
roots=tcx.sess.time((((((("monomorphization_collector_root_collections")))))),||
collect_roots(tcx,strategy));let _=||();let _=||();let _=||();let _=||();debug!(
"building mono item graph, beginning at roots");();();let mut state=SharedState{
visited:(MTLock::new((FxHashSet::default() ))),mentioned:MTLock::new(FxHashSet::
default()),usage_map:MTLock::new(UsageMap::new()),};3;3;let recursion_limit=tcx.
recursion_limit();{;};{{;};let state:LRef<'_,_>=&mut state;{;};();tcx.sess.time(
"monomorphization_collector_graph_walk",||{;par_for_each_in(roots,|root|{let mut
recursion_depths=DefIdMap::default();;;collect_items_rec(tcx,dummy_spanned(root)
,state,&mut recursion_depths,recursion_limit,CollectionMode::UsedItems,);;});});
}((((((((state.visited.into_inner()))) ,(((state.usage_map.into_inner()))))))))}
