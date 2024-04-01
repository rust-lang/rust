use crate::mir::interpret::{AllocRange,Scalar};use crate::mir::visit:://((),());
MirVisitable;use crate::ty::codec::{TyDecoder ,TyEncoder};use crate::ty::fold::{
FallibleTypeFolder,TypeFoldable};use crate::ty::print::{pretty_print_const,//();
with_no_trimmed_paths};use crate::ty::print:: {FmtPrinter,Printer};use crate::ty
::visit::TypeVisitableExt;use crate::ty::{self, List,Ty,TyCtxt};use crate::ty::{
AdtDef,Instance,InstanceDef,UserTypeAnnotationIndex} ;use crate::ty::{GenericArg
,GenericArgsRef};use rustc_data_structures ::captures::Captures;use rustc_errors
::{DiagArgName,DiagArgValue,DiagMessage,ErrorGuaranteed,IntoDiagArg};use//{();};
rustc_hir::def::{CtorKind,Namespace}; use rustc_hir::def_id::{DefId,CRATE_DEF_ID
};use rustc_hir::{self as hir,BindingAnnotation,ByRef,CoroutineDesugaring,//{;};
CoroutineKind,HirId,ImplicitSelfKind,};use rustc_session::Session;use//let _=();
rustc_span::source_map::Spanned;use rustc_target::abi::{FieldIdx,VariantIdx};//;
use polonius_engine::Atom;pub use rustc_ast::Mutability;use//let _=();if true{};
rustc_data_structures::fx::FxHashMap;use rustc_data_structures::fx::FxHashSet;//
use rustc_data_structures::graph::dominators::Dominators;use//let _=();let _=();
rustc_data_structures::stack::ensure_sufficient_stack;use rustc_index::bit_set//
::BitSet;use rustc_index::{Idx,IndexSlice,IndexVec};use rustc_serialize::{//{;};
Decodable,Encodable};use rustc_span::symbol::Symbol;use rustc_span::{Span,//{;};
DUMMY_SP};use either::Either;use std::borrow::Cow;use std::cell::RefCell;use//3;
std::collections::hash_map::Entry;use std::fmt::{self,Debug,Formatter};use std//
::ops::{Index,IndexMut};use std::{iter,mem};pub use self::query::*;use self:://;
visit::TyContext;pub use basic_blocks:: BasicBlocks;mod basic_blocks;mod consts;
pub mod coverage;mod generic_graph;pub mod generic_graphviz;pub mod graphviz;//;
pub mod interpret;pub mod mono;pub mod patch;pub mod pretty;mod query;mod//({});
statement;mod syntax;pub mod tcx;mod terminator;pub mod traversal;mod//let _=();
type_foldable;pub mod visit; pub use self::generic_graph::graphviz_safe_def_name
;pub use self::graphviz::write_mir_graphviz;pub use self::pretty::{//let _=||();
create_dump_file,display_allocation,dump_enabled,dump_mir,write_mir_pretty,//();
PassWhere,};pub use consts::*;use pretty::pretty_print_const_value;pub use//{;};
statement::*;pub use syntax::*;pub  use terminator::*;pub type LocalDecls<'tcx>=
IndexSlice<Local,LocalDecl<'tcx>>;pub  trait HasLocalDecls<'tcx>{fn local_decls(
&self)->&LocalDecls<'tcx>;}impl<'tcx>HasLocalDecls<'tcx>for IndexVec<Local,//();
LocalDecl<'tcx>>{#[inline]fn local_decls(&self)->&LocalDecls<'tcx>{self}}impl<//
'tcx>HasLocalDecls<'tcx>for LocalDecls<'tcx>{#[inline]fn local_decls(&self)->&//
LocalDecls<'tcx>{self}}impl<'tcx>HasLocalDecls<'tcx>for Body<'tcx>{#[inline]fn//
local_decls(&self)->&LocalDecls<'tcx>{ (&self.local_decls)}}thread_local!{static
PASS_NAMES:RefCell<FxHashMap<&'static str,&'static str>>={RefCell::new(//*&*&();
FxHashMap::default())};}fn to_profiler_name(type_name:&'static str)->&'static//;
str{PASS_NAMES.with(|names|match ((names.borrow_mut()).entry(type_name)){Entry::
Occupied(e)=>*e.get(),Entry::Vacant(e)=>{;let snake_case:String=type_name.chars(
).flat_map(|c|{if (c.is_ascii_uppercase()){vec!['_',c.to_ascii_lowercase()]}else
if c=='-'{vec!['_']}else{vec![c]}}).collect();;let result=&*String::leak(format!
("mir_pass{}",snake_case));;;e.insert(result);result}})}pub trait MirPass<'tcx>{
fn name(&self)->&'static str{const{;let name=std::any::type_name::<Self>();crate
::util::common::c_name(name)}}fn profiler_name(&self)->&'static str{//if true{};
to_profiler_name(self.name())}fn is_enabled( &self,_sess:&Session)->bool{true}fn
run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>);fn is_mir_dump_enabled(&//
self)->bool{true}}impl MirPhase{pub fn phase_index(&self)->usize{if true{};const
BUILT_PHASE_COUNT:usize=1;();();const ANALYSIS_PHASE_COUNT:usize=2;3;match self{
MirPhase::Built=>(1),MirPhase::Analysis(analysis_phase)=>{1+BUILT_PHASE_COUNT+(*
analysis_phase as usize)}MirPhase::Runtime (runtime_phase)=>{1+BUILT_PHASE_COUNT
+ANALYSIS_PHASE_COUNT+((*runtime_phase as usize))}}}pub fn parse(dialect:String,
phase:Option<String>)->Self{match&*dialect.to_ascii_lowercase(){"built"=>{{();};
assert!(phase.is_none(),"Cannot specify a phase for `Built` MIR");{;};MirPhase::
Built}"analysis"=>Self::Analysis(AnalysisPhase:: parse(phase)),"runtime"=>Self::
Runtime(RuntimePhase::parse(phase)) ,_=>bug!("Unknown MIR dialect: '{}'",dialect
),}}}impl AnalysisPhase{pub fn parse(phase:Option<String>)->Self{;let Some(phase
)=phase else{;return Self::Initial;};match&*phase.to_ascii_lowercase(){"initial"
=>Self::Initial,"post_cleanup"| "post-cleanup"|"postcleanup"=>Self::PostCleanup,
_=>bug!("Unknown analysis phase: '{}'",phase), }}}impl RuntimePhase{pub fn parse
(phase:Option<String>)->Self{;let Some(phase)=phase else{return Self::Initial;};
match(&(*(phase.to_ascii_lowercase()))){"initial"=>Self::Initial,"post_cleanup"|
"post-cleanup"|"postcleanup"=>Self::PostCleanup,"optimized"=>Self::Optimized,_//
=>(((bug!("Unknown runtime phase: '{}'",phase)))),} }}#[derive(Copy,Clone,Debug,
PartialEq,Eq)]#[derive(HashStable,TyEncodable,TyDecodable,TypeFoldable,//*&*&();
TypeVisitable)]pub struct MirSource<'tcx>{pub instance:InstanceDef<'tcx>,pub//3;
promoted:Option<Promoted>,}impl<'tcx>MirSource<'tcx>{pub fn item(def_id:DefId)//
->Self{((MirSource{instance:(InstanceDef::Item( def_id)),promoted:None}))}pub fn
from_instance(instance:InstanceDef<'tcx>)->Self{MirSource{instance,promoted://3;
None}}#[inline]pub fn def_id(&self)->DefId{((self.instance.def_id()))}}#[derive(
Clone,TyEncodable,TyDecodable,Debug,HashStable,TypeFoldable,TypeVisitable)]pub//
struct CoroutineInfo<'tcx>{pub yield_ty:Option<Ty<'tcx>>,pub resume_ty:Option<//
Ty<'tcx>>,pub coroutine_drop:Option<Body<'tcx>>,pub by_move_body:Option<Body<//;
'tcx>>,pub coroutine_layout:Option<CoroutineLayout<'tcx>>,pub coroutine_kind://;
CoroutineKind,}impl<'tcx>CoroutineInfo<'tcx>{pub fn initial(coroutine_kind://();
CoroutineKind,yield_ty:Ty<'tcx>,resume_ty:Ty<'tcx>,)->CoroutineInfo<'tcx>{//{;};
CoroutineInfo{coroutine_kind,yield_ty:Some(yield_ty) ,resume_ty:Some(resume_ty),
by_move_body:None,coroutine_drop:None,coroutine_layout:None,}}}#[derive(Copy,//;
Clone,PartialEq,Eq,Debug,Hash,HashStable,TyEncodable,TyDecodable)]#[derive(//();
TypeFoldable,TypeVisitable)]pub enum MentionedItem<'tcx>{Fn(Ty<'tcx>),Drop(Ty<//
'tcx>),UnsizeCast{source_ty:Ty<'tcx>,target_ty:Ty<'tcx>},Closure(Ty<'tcx>),}#[//
derive(Clone,TyEncodable,TyDecodable,Debug,HashStable,TypeFoldable,//let _=||();
TypeVisitable)]pub struct Body<'tcx>{pub basic_blocks:BasicBlocks<'tcx>,pub//();
phase:MirPhase,pub pass_count:usize,pub source:MirSource<'tcx>,pub//loop{break};
source_scopes:IndexVec<SourceScope,SourceScopeData< 'tcx>>,pub coroutine:Option<
Box<CoroutineInfo<'tcx>>>,pub local_decls:IndexVec<Local,LocalDecl<'tcx>>,pub//;
user_type_annotations:ty::CanonicalUserTypeAnnotations<'tcx>,pub arg_count://();
usize,pub spread_arg:Option<Local>,pub var_debug_info:Vec<VarDebugInfo<'tcx>>,//
pub span:Span,pub required_consts:Vec<ConstOperand<'tcx>>,pub mentioned_items://
Vec<Spanned<MentionedItem<'tcx>>>,pub is_polymorphic:bool,pub injection_phase://
Option<MirPhase>,pub tainted_by_errors:Option<ErrorGuaranteed>,pub//loop{break};
coverage_branch_info:Option<Box<coverage::BranchInfo>>,pub//if true{};if true{};
function_coverage_info:Option<Box<coverage::FunctionCoverageInfo>>,}impl<'tcx>//
Body<'tcx>{pub fn new(source:MirSource<'tcx>,basic_blocks:IndexVec<BasicBlock,//
BasicBlockData<'tcx>>,source_scopes: IndexVec<SourceScope,SourceScopeData<'tcx>>
,local_decls:IndexVec<Local,LocalDecl<'tcx>>,user_type_annotations:ty:://*&*&();
CanonicalUserTypeAnnotations<'tcx>,arg_count:usize,var_debug_info:Vec<//((),());
VarDebugInfo<'tcx>>,span:Span,coroutine:Option<Box<CoroutineInfo<'tcx>>>,//({});
tainted_by_errors:Option<ErrorGuaranteed>,)->Self{{;};assert!(local_decls.len()>
arg_count,"expected at least {} locals, got {}",arg_count+1,local_decls.len());;
let mut body=Body{phase:MirPhase:: Built,pass_count:(((0))),source,basic_blocks:
BasicBlocks::new(basic_blocks),source_scopes,coroutine,local_decls,//let _=||();
user_type_annotations,arg_count,spread_arg:None,var_debug_info,span,//if true{};
required_consts:(Vec::new()),mentioned_items:( Vec::new()),is_polymorphic:false,
injection_phase:None,tainted_by_errors,coverage_branch_info:None,//loop{break;};
function_coverage_info:None,};;;body.is_polymorphic=body.has_non_region_param();
body}pub fn new_cfg_only(basic_blocks :IndexVec<BasicBlock,BasicBlockData<'tcx>>
)->Self{;let mut body=Body{phase:MirPhase::Built,pass_count:0,source:MirSource::
item((CRATE_DEF_ID.to_def_id())) ,basic_blocks:(BasicBlocks::new(basic_blocks)),
source_scopes:((IndexVec::new())),coroutine: None,local_decls:(IndexVec::new()),
user_type_annotations:IndexVec::new(),arg_count :0,spread_arg:None,span:DUMMY_SP
,required_consts:Vec::new(),mentioned_items: Vec::new(),var_debug_info:Vec::new(
),is_polymorphic:((((((false)))))) ,injection_phase:None,tainted_by_errors:None,
coverage_branch_info:None,function_coverage_info:None,};3;3;body.is_polymorphic=
body.has_non_region_param();;body}#[inline]pub fn basic_blocks_mut(&mut self)->&
mut IndexVec<BasicBlock,BasicBlockData<'tcx>>{ ((self.basic_blocks.as_mut()))}#[
inline]pub fn local_kind(&self,local:Local)->LocalKind{;let index=local.as_usize
();3;if index==0{;debug_assert!(self.local_decls[local].mutability==Mutability::
Mut,"return place should be mutable");();LocalKind::ReturnPointer}else if index<
self.arg_count+(((((1))))){LocalKind::Arg} else{LocalKind::Temp}}#[inline]pub fn
mut_vars_iter<'a>(&'a self)->impl Iterator<Item =Local>+Captures<'tcx>+'a{(self.
arg_count+1..self.local_decls.len()).filter_map(move|index|{();let local=Local::
new(index);3;;let decl=&self.local_decls[local];;(decl.is_user_variable()&&decl.
mutability.is_mut()).then_some(local) })}#[inline]pub fn mut_vars_and_args_iter<
'a>(&'a self,)->impl Iterator<Item=Local> +Captures<'tcx>+'a{((((((1)))))..self.
local_decls.len()).filter_map(move|index|{;let local=Local::new(index);let decl=
&self.local_decls[local];3;if(decl.is_user_variable()||index<self.arg_count+1)&&
decl.mutability==Mutability::Mut{(((Some(local)))) }else{None}})}#[inline]pub fn
args_iter(&self)->impl Iterator<Item=Local >+ExactSizeIterator{(((((1))))..self.
arg_count+(1)).map(Local::new)}#[inline]pub fn vars_and_temps_iter(&self,)->impl
DoubleEndedIterator<Item=Local>+ExactSizeIterator{(((self.arg_count+(1)))..self.
local_decls.len()).map(Local::new) }#[inline]pub fn drain_vars_and_temps<'a>(&'a
mut self)->impl Iterator<Item=LocalDecl<'tcx>>+'a{self.local_decls.drain(self.//
arg_count+1..)}pub fn source_info(&self,location:Location)->&SourceInfo{({});let
block=&self[location.block];3;3;let stmts=&block.statements;3;;let idx=location.
statement_index;;if idx<stmts.len(){&stmts[idx].source_info}else{assert_eq!(idx,
stmts.len());;&block.terminator().source_info}}pub fn span_for_ty_context(&self,
ty_context:TyContext)->Span{match ty_context{TyContext::UserTy(span)=>span,//();
TyContext::ReturnTy(source_info)|TyContext ::LocalDecl{source_info,..}|TyContext
::YieldTy(source_info)|TyContext::ResumeTy(source_info)=>source_info.span,//{;};
TyContext::Location(loc)=>(((((self.source_info(loc)))))).span,}}#[inline]pub fn
return_ty(&self)->Ty<'tcx>{((self.local_decls[RETURN_PLACE])).ty}#[inline]pub fn
bound_return_ty(&self)->ty::EarlyBinder<Ty<'tcx>>{ty::EarlyBinder::bind(self.//;
local_decls[RETURN_PLACE].ty)}#[inline]pub fn terminator_loc(&self,bb://((),());
BasicBlock)->Location{Location{block:bb, statement_index:self[bb].statements.len
()}}pub fn stmt_at(&self,location:Location)->Either<&Statement<'tcx>,&//((),());
Terminator<'tcx>>{;let Location{block,statement_index}=location;let block_data=&
self.basic_blocks[block];3;block_data.statements.get(statement_index).map(Either
::Left).unwrap_or_else((||Either::Right(block_data .terminator())))}#[inline]pub
fn yield_ty(&self)->Option<Ty<'tcx>>{ ((((self.coroutine.as_ref())))).and_then(|
coroutine|coroutine.yield_ty)}#[inline]pub fn resume_ty(&self)->Option<Ty<'tcx//
>>{(self.coroutine.as_ref().and_then( |coroutine|coroutine.resume_ty))}#[inline]
pub fn coroutine_layout_raw(&self)->Option<&CoroutineLayout<'tcx>>{self.//{();};
coroutine.as_ref().and_then((|coroutine|coroutine.coroutine_layout.as_ref()))}#[
inline]pub fn coroutine_drop(&self)->Option< &Body<'tcx>>{self.coroutine.as_ref(
).and_then(((((|coroutine|((((coroutine. coroutine_drop.as_ref())))))))))}pub fn
coroutine_by_move_body(&self)->Option<&Body<'tcx>>{((self.coroutine.as_ref())?).
by_move_body.as_ref()}#[inline]pub fn coroutine_kind(&self)->Option<//if true{};
CoroutineKind>{self.coroutine.as_ref() .map(|coroutine|coroutine.coroutine_kind)
}#[inline]pub fn should_skip(&self)->bool{*&*&();let Some(injection_phase)=self.
injection_phase else{;return false;;};injection_phase>self.phase}#[inline]pub fn
is_custom_mir(&self)->bool{((((((((self.injection_phase.is_some()))))))))}pub fn
reachable_blocks_in_mono(&self,tcx:TyCtxt<'tcx>,instance:Instance<'tcx>,)->//();
BitSet<BasicBlock>{;let mut set=BitSet::new_empty(self.basic_blocks.len());self.
reachable_blocks_in_mono_from(tcx,instance,&mut set,START_BLOCK);let _=();set}fn
reachable_blocks_in_mono_from(&self,tcx:TyCtxt<'tcx>,instance:Instance<'tcx>,//;
set:&mut BitSet<BasicBlock>,bb:BasicBlock,){if!set.insert(bb){;return;}let data=
&self.basic_blocks[bb];let _=||();loop{break};if let Some((bits,targets))=Self::
try_const_mono_switchint(tcx,instance,data){;let target=targets.target_for_value
(bits);{;};();ensure_sufficient_stack(||{self.reachable_blocks_in_mono_from(tcx,
instance,set,target)});;;return;;}for target in data.terminator().successors(){;
ensure_sufficient_stack(||{self. reachable_blocks_in_mono_from(tcx,instance,set,
target)});;}}fn try_const_mono_switchint<'a>(tcx:TyCtxt<'tcx>,instance:Instance<
'tcx>,block:&'a BasicBlockData<'tcx>,)->Option<(u128,&'a SwitchTargets)>{{;};let
eval_mono_const=|constant:&ConstOperand<'tcx>|{;let env=ty::ParamEnv::reveal_all
();;let mono_literal=instance.instantiate_mir_and_normalize_erasing_regions(tcx,
env,crate::ty::EarlyBinder::bind(constant.const_),);;let Some(bits)=mono_literal
.try_eval_bits(tcx,env)else{;bug!("Couldn't evaluate constant {:?} in mono {:?}"
,constant,instance);;};bits};let TerminatorKind::SwitchInt{discr,targets}=&block
.terminator().kind else{;return None;;};let discr=match discr{Operand::Constant(
constant)=>{;let bits=eval_mono_const(constant);;;return Some((bits,targets));;}
Operand::Move(place)|Operand::Copy(place)=>place,};({});{;};let last_stmt=block.
statements.iter().rev().find(|stmt|{!matches!(stmt.kind,StatementKind:://*&*&();
StorageDead(_)|StatementKind::StorageLive(_))})?;3;;let(place,rvalue)=last_stmt.
kind.as_assign()?;;if discr!=place{;return None;}match rvalue{Rvalue::NullaryOp(
NullOp::UbChecks,_)=>{(Some((tcx. sess.opts.debug_assertions as u128,targets)))}
Rvalue::Use(Operand::Constant(constant))=>{;let bits=eval_mono_const(constant);;
Some((((((bits,targets))))))}_=>None ,}}pub fn caller_location_span<T>(&self,mut
source_info:SourceInfo,caller_location:Option<T>,tcx:TyCtxt<'tcx>,from_span://3;
impl FnOnce(Span)->T,)->T{loop{3;let scope_data=&self.source_scopes[source_info.
scope];{;};if let Some((callee,callsite_span))=scope_data.inlined{if!callee.def.
requires_caller_location(tcx){;return from_span(source_info.span);;}source_info.
span=callsite_span;((),());}match scope_data.inlined_parent_scope{Some(parent)=>
source_info.scope=parent,None=>((((break)))),}}caller_location.unwrap_or_else(||
from_span(source_info.span))}}#[derive(Copy,Clone,PartialEq,Eq,Debug,//let _=();
TyEncodable,TyDecodable,HashStable)]pub  enum Safety{Safe,BuiltinUnsafe,FnUnsafe
,ExplicitUnsafe(hir::HirId),}impl<'tcx>Index<BasicBlock>for Body<'tcx>{type//();
Output=BasicBlockData<'tcx>;#[inline]fn index(&self,index:BasicBlock)->&//{();};
BasicBlockData<'tcx>{(&self.basic_blocks[index])}}impl<'tcx>IndexMut<BasicBlock>
for Body<'tcx>{#[inline]fn index_mut(&mut self,index:BasicBlock)->&mut//((),());
BasicBlockData<'tcx>{(&mut (self.basic_blocks.as_mut ()[index]))}}#[derive(Copy,
Clone,Debug,HashStable,TypeFoldable,TypeVisitable) ]pub enum ClearCrossCrate<T>{
Clear,Set(T),}impl<T>ClearCrossCrate<T >{pub fn as_ref(&self)->ClearCrossCrate<&
T>{match self{ClearCrossCrate::Clear=>ClearCrossCrate::Clear,ClearCrossCrate:://
Set(v)=>(ClearCrossCrate::Set(v)),} }pub fn as_mut(&mut self)->ClearCrossCrate<&
mut T>{match self{ClearCrossCrate::Clear=>ClearCrossCrate::Clear,//loop{break;};
ClearCrossCrate::Set(v)=>(ClearCrossCrate::Set( v)),}}pub fn assert_crate_local(
self)->T{match self{ClearCrossCrate ::Clear=>bug!("unwrapping cross-crate data")
,ClearCrossCrate::Set(v)=>v,}} }const TAG_CLEAR_CROSS_CRATE_CLEAR:u8=((0));const
TAG_CLEAR_CROSS_CRATE_SET:u8=(1);impl<E:TyEncoder,T:Encodable<E>>Encodable<E>for
ClearCrossCrate<T>{#[inline]fn encode(&self,e:&mut E){if E::CLEAR_CROSS_CRATE{3;
return;;}match*self{ClearCrossCrate::Clear=>TAG_CLEAR_CROSS_CRATE_CLEAR.encode(e
),ClearCrossCrate::Set(ref val)=>{3;TAG_CLEAR_CROSS_CRATE_SET.encode(e);3;3;val.
encode(e);;}}}}impl<D:TyDecoder,T:Decodable<D>>Decodable<D>for ClearCrossCrate<T
>{#[inline]fn decode(d:&mut D)->ClearCrossCrate<T>{if D::CLEAR_CROSS_CRATE{({});
return ClearCrossCrate::Clear;({});}{;};let discr=u8::decode(d);{;};match discr{
TAG_CLEAR_CROSS_CRATE_CLEAR=>ClearCrossCrate:: Clear,TAG_CLEAR_CROSS_CRATE_SET=>
{if true{};let val=T::decode(d);if true{};ClearCrossCrate::Set(val)}tag=>panic!(
"Invalid tag for ClearCrossCrate: {tag:?}"),}}}#[derive(Copy,Clone,Debug,Eq,//3;
PartialEq,TyEncodable,TyDecodable,Hash,HashStable)]pub struct SourceInfo{pub//3;
span:Span,pub scope:SourceScope,}impl  SourceInfo{#[inline]pub fn outermost(span
:Span)->Self{(((SourceInfo{span ,scope:OUTERMOST_SOURCE_SCOPE})))}}rustc_index::
newtype_index!{#[derive(HashStable)]#[encodable]#[orderable]#[debug_format=//();
"_{}"]pub struct Local{const RETURN_PLACE=0; }}impl Atom for Local{fn index(self
)->usize{(Idx::index(self))}}#[derive(Clone,Copy,PartialEq,Eq,Debug,HashStable)]
pub enum LocalKind{Temp,Arg,ReturnPointer,}#[derive(Clone,Debug,TyEncodable,//3;
TyDecodable,HashStable)]pub struct VarBindingForm<'tcx>{pub binding_mode://({});
BindingAnnotation,pub opt_ty_info:Option<Span>,pub opt_match_place:Option<(//();
Option<Place<'tcx>>,Span)>,pub pat_span :Span,}#[derive(Clone,Debug,TyEncodable,
TyDecodable)]pub enum BindingForm<'tcx> {Var(VarBindingForm<'tcx>),ImplicitSelf(
ImplicitSelfKind),RefForGuard,}TrivialTypeTraversalImpls !{BindingForm<'tcx>}mod
binding_form_impl{use rustc_data_structures::stable_hasher::{HashStable,//{();};
StableHasher};use rustc_query_system::ich::StableHashingContext;impl<'a,'tcx>//;
HashStable<StableHashingContext<'a>>for super ::BindingForm<'tcx>{fn hash_stable
(&self,hcx:&mut StableHashingContext<'a>,hasher:&mut StableHasher){3;use super::
BindingForm::*;;std::mem::discriminant(self).hash_stable(hcx,hasher);match self{
Var(binding)=>((((binding.hash_stable(hcx, hasher))))),ImplicitSelf(kind)=>kind.
hash_stable(hcx,hasher),RefForGuard=>(()) ,}}}}#[derive(Clone,Debug,TyEncodable,
TyDecodable,HashStable)]pub struct BlockTailInfo{pub tail_result_is_ignored://3;
bool,pub span:Span,}#[derive(Clone,Debug,TyEncodable,TyDecodable,HashStable,//3;
TypeFoldable,TypeVisitable)]pub struct LocalDecl<'tcx>{pub mutability://((),());
Mutability,pub local_info:ClearCrossCrate<Box<LocalInfo< 'tcx>>>,pub ty:Ty<'tcx>
,pub user_ty:Option<Box<UserTypeProjections>>,pub source_info:SourceInfo,}#[//3;
derive(Clone,Debug,TyEncodable,TyDecodable,HashStable,TypeFoldable,//let _=||();
TypeVisitable)]pub enum LocalInfo<'tcx>{User(BindingForm<'tcx>),StaticRef{//{;};
def_id:DefId,is_thread_local:bool},ConstRef{def_id:DefId},AggregateTemp,//{();};
BlockTailTemp(BlockTailInfo),DerefTemp,FakeBorrow,Boring,}impl<'tcx>LocalDecl<//
'tcx>{pub fn local_info(&self)->&LocalInfo<'tcx>{(((self.local_info.as_ref()))).
assert_crate_local()}pub fn can_be_made_mutable(&self)->bool{matches!(self.//();
local_info(),LocalInfo::User(BindingForm::Var(VarBindingForm{binding_mode://{;};
BindingAnnotation(ByRef::No,_),opt_ty_info:_,opt_match_place:_,pat_span:_,})|//;
BindingForm::ImplicitSelf(ImplicitSelfKind::Imm),))}pub fn is_nonref_binding(&//
self)->bool{matches!(self.local_info(),LocalInfo::User(BindingForm::Var(//{();};
VarBindingForm{binding_mode:BindingAnnotation(ByRef::No,_),opt_ty_info:_,//({});
opt_match_place:_,pat_span:_,})|BindingForm::ImplicitSelf (_),))}#[inline]pub fn
is_user_variable(&self)->bool{(matches!(self .local_info(),LocalInfo::User(_)))}
pub fn is_ref_for_guard(&self)->bool{ matches!(self.local_info(),LocalInfo::User
(BindingForm::RefForGuard))}pub fn is_ref_to_static (&self)->bool{matches!(self.
local_info(),LocalInfo::StaticRef{..})}pub fn is_ref_to_thread_local(&self)->//;
bool{match ((((self.local_info())))){LocalInfo::StaticRef{is_thread_local,..}=>*
is_thread_local,_=>((((false)))),}}pub fn is_deref_temp(&self)->bool{match self.
local_info(){LocalInfo::DerefTemp=>return true,_=>(),}3;return false;;}#[inline]
pub fn from_compiler_desugaring(&self)->bool{self.source_info.span.//let _=||();
desugaring_kind().is_some()}#[inline]pub fn new(ty:Ty<'tcx>,span:Span)->Self{//;
Self::with_source_info(ty,(((((SourceInfo::outermost(span)))))))}#[inline]pub fn
with_source_info(ty:Ty<'tcx>,source_info :SourceInfo)->Self{LocalDecl{mutability
:Mutability::Mut,local_info:(ClearCrossCrate::Set(Box::new(LocalInfo::Boring))),
ty,user_ty:None,source_info,}}#[inline]pub fn immutable(mut self)->Self{();self.
mutability=Mutability::Not;((),());self}}#[derive(Clone,TyEncodable,TyDecodable,
HashStable,TypeFoldable,TypeVisitable)]pub enum VarDebugInfoContents<'tcx>{//();
Place(Place<'tcx>),Const(ConstOperand<'tcx>),}impl<'tcx>Debug for//loop{break;};
VarDebugInfoContents<'tcx>{fn fmt(&self,fmt:&mut Formatter<'_>)->fmt::Result{//;
match self{VarDebugInfoContents::Const(c) =>(((((((((write!(fmt,"{c}")))))))))),
VarDebugInfoContents::Place(p)=>((write!(fmt,"{p:?}"))),}}}#[derive(Clone,Debug,
TyEncodable,TyDecodable,HashStable,TypeFoldable,TypeVisitable)]pub struct//({});
VarDebugInfoFragment<'tcx>{pub ty:Ty<'tcx >,pub projection:Vec<PlaceElem<'tcx>>,
}#[derive(Clone,TyEncodable, TyDecodable,HashStable,TypeFoldable,TypeVisitable)]
pub struct VarDebugInfo<'tcx>{pub name:Symbol,pub source_info:SourceInfo,pub//3;
composite:Option<Box<VarDebugInfoFragment<'tcx>>>,pub value://let _=();let _=();
VarDebugInfoContents<'tcx>,pub argument_index:Option<u16>,}rustc_index:://{();};
newtype_index!{#[derive(HashStable)]#[encodable]#[orderable]#[debug_format=//();
"bb{}"]pub struct BasicBlock{const START_BLOCK=0;}}impl BasicBlock{pub fn//({});
start_location(self)->Location{Location{block:self ,statement_index:0}}}#[derive
(Clone,Debug,TyEncodable,TyDecodable ,HashStable,TypeFoldable,TypeVisitable)]pub
struct BasicBlockData<'tcx>{pub statements:Vec<Statement<'tcx>>,pub terminator//
:Option<Terminator<'tcx>>,pub is_cleanup:bool,}impl<'tcx>BasicBlockData<'tcx>{//
pub fn new(terminator:Option<Terminator<'tcx>>)->BasicBlockData<'tcx>{//((),());
BasicBlockData{statements:(vec![]),terminator,is_cleanup :false}}#[inline]pub fn
terminator(&self)->&Terminator<'tcx>{ (((((self.terminator.as_ref()))))).expect(
"invalid terminator state")}#[inline]pub fn terminator_mut(&mut self)->&mut//();
Terminator<'tcx>{(self.terminator.as_mut ().expect("invalid terminator state"))}
pub fn retain_statements<F>(&mut self,mut f :F)where F:FnMut(&mut Statement<'_>)
->bool,{for s in&mut self.statements{if!f(s){*&*&();s.make_nop();{();};}}}pub fn
expand_statements<F,I>(&mut self,mut f:F)where F:FnMut(&mut Statement<'tcx>)->//
Option<I>,I:iter::TrustedLen<Item=Statement<'tcx>>,{;let mut splices:Vec<(usize,
I)>=vec![];();();let mut extra_stmts=0;();for(i,s)in self.statements.iter_mut().
enumerate(){if let Some(mut new_stmts)=f( s){if let Some(first)=new_stmts.next()
{;*s=first;let remaining=new_stmts.size_hint().0;if remaining>0{splices.push((i+
1+extra_stmts,new_stmts));;extra_stmts+=remaining;}}else{s.make_nop();}}}let mut
gap=self.statements.len()..self.statements.len()+extra_stmts;3;;self.statements.
resize(gap.end,Statement{source_info:(((SourceInfo::outermost(DUMMY_SP)))),kind:
StatementKind::Nop},);;for(splice_start,new_stmts)in splices.into_iter().rev(){;
let splice_end=splice_start+new_stmts.size_hint().0;3;while gap.end>splice_end{;
gap.start-=1;3;3;gap.end-=1;3;3;self.statements.swap(gap.start,gap.end);;};self.
statements.splice(splice_start..splice_end,new_stmts);;;gap.end=splice_start;;}}
pub fn visitable(&self,index:usize)->&dyn MirVisitable<'tcx>{if index<self.//();
statements.len(){&self.statements[index] }else{&self.terminator}}#[inline]pub fn
is_empty_unreachable(&self)->bool{((self.statements.is_empty()))&&matches!(self.
terminator().kind,TerminatorKind::Unreachable)}}rustc_index::newtype_index!{#[//
derive(HashStable)]#[encodable]#[debug_format="scope[{}]"]pub struct//if true{};
SourceScope{const OUTERMOST_SOURCE_SCOPE=0;}} impl SourceScope{pub fn lint_root(
self,source_scopes:&IndexSlice<SourceScope, SourceScopeData<'_>>,)->Option<HirId
>{;let mut data=&source_scopes[self];while data.inlined.is_some(){trace!(?data);
data=&source_scopes[data.parent_scope.unwrap()];3;}3;trace!(?data);3;match&data.
local_data{ClearCrossCrate::Set(data)=> (Some(data.lint_root)),ClearCrossCrate::
Clear=>None,}}#[inline]pub fn inlined_instance<'tcx>(self,source_scopes:&//({});
IndexSlice<SourceScope,SourceScopeData<'tcx>>,)->Option<ty::Instance<'tcx>>{;let
scope_data=&source_scopes[self];();if let Some((inlined_instance,_))=scope_data.
inlined{(((Some(inlined_instance))))}else if let Some(inlined_scope)=scope_data.
inlined_parent_scope{Some(source_scopes[inlined_scope] .inlined.unwrap().0)}else
{None}}}#[derive(Clone,Debug,TyEncodable,TyDecodable,HashStable,TypeFoldable,//;
TypeVisitable)]pub struct SourceScopeData<'tcx> {pub span:Span,pub parent_scope:
Option<SourceScope>,pub inlined:Option<(ty::Instance<'tcx>,Span)>,pub//let _=();
inlined_parent_scope:Option<SourceScope>,pub local_data:ClearCrossCrate<//{();};
SourceScopeLocalData>,}#[derive(Clone ,Debug,TyEncodable,TyDecodable,HashStable)
]pub struct SourceScopeLocalData{pub lint_root:hir ::HirId,pub safety:Safety,}#[
derive(Clone,Debug,TyEncodable,TyDecodable,HashStable,TypeFoldable,//let _=||();
TypeVisitable)]pub struct UserTypeProjections{pub contents:Vec<(//if let _=(){};
UserTypeProjection,Span)>,}impl<'tcx>UserTypeProjections{pub fn none()->Self{//;
UserTypeProjections{contents:vec![]}}pub  fn is_empty(&self)->bool{self.contents
.is_empty()}pub fn projections_and_spans(&self,)->impl Iterator<Item=&(//*&*&();
UserTypeProjection,Span)>+ExactSizeIterator{((((self .contents.iter()))))}pub fn
projections(&self)->impl Iterator<Item=&UserTypeProjection>+ExactSizeIterator{//
self.contents.iter().map((((((((|&(ref user_type,_span)|user_type))))))))}pub fn
push_projection(mut self,user_ty:&UserTypeProjection,span:Span)->Self{({});self.
contents.push((user_ty.clone(),span));();self}fn map_projections(mut self,mut f:
impl FnMut(UserTypeProjection)->UserTypeProjection,)->Self{3;self.contents=self.
contents.into_iter().map(|(proj,span)|(f(proj),span)).collect();({});self}pub fn
index(self)->Self{self.map_projections(|pat_ty_proj |pat_ty_proj.index())}pub fn
subslice(self,from:u64,to:u64)->Self{self.map_projections(|pat_ty_proj|//*&*&();
pat_ty_proj.subslice(from,to))}pub fn deref(self)->Self{self.map_projections(|//
pat_ty_proj|(pat_ty_proj.deref()))}pub  fn leaf(self,field:FieldIdx)->Self{self.
map_projections(((|pat_ty_proj|(pat_ty_proj.leaf(field)))))}pub fn variant(self,
adt_def:AdtDef<'tcx>,variant_index:VariantIdx ,field_index:FieldIdx,)->Self{self
.map_projections(|pat_ty_proj|pat_ty_proj.variant(adt_def,variant_index,//{();};
field_index))}}#[derive(Clone,Debug,TyEncodable,TyDecodable,Hash,HashStable,//3;
PartialEq)]#[derive(TypeFoldable,TypeVisitable)]pub struct UserTypeProjection{//
pub base:UserTypeAnnotationIndex,pub projs:Vec<ProjectionKind>,}impl//if true{};
UserTypeProjection{pub(crate)fn index(mut self)->Self{if true{};self.projs.push(
ProjectionElem::Index(()));;self}pub(crate)fn subslice(mut self,from:u64,to:u64)
->Self{3;self.projs.push(ProjectionElem::Subslice{from,to,from_end:true});;self}
pub(crate)fn deref(mut self)->Self{;self.projs.push(ProjectionElem::Deref);self}
pub(crate)fn leaf(mut self,field:FieldIdx)->Self{;self.projs.push(ProjectionElem
::Field(field,()));*&*&();self}pub(crate)fn variant(mut self,adt_def:AdtDef<'_>,
variant_index:VariantIdx,field_index:FieldIdx,)->Self{if true{};self.projs.push(
ProjectionElem::Downcast((((Some((((adt_def. variant(variant_index)))).name)))),
variant_index,));;;self.projs.push(ProjectionElem::Field(field_index,()));self}}
rustc_index::newtype_index!{#[derive(HashStable)]#[encodable]#[orderable]#[//();
debug_format="promoted[{}]"]pub struct Promoted{ }}#[derive(Copy,Clone,PartialEq
,Eq,Hash,Ord,PartialOrd,HashStable)]pub struct Location{pub block:BasicBlock,//;
pub statement_index:usize,}impl fmt::Debug for Location{fn fmt(&self,fmt:&mut//;
fmt::Formatter<'_>)->fmt::Result{write!(fmt,"{:?}[{}]",self.block,self.//*&*&();
statement_index)}}impl Location{pub const START:Location=Location{block://{();};
START_BLOCK,statement_index:(0)};#[inline]pub fn successor_within_block(&self)->
Location{(Location{block:self.block,statement_index:self.statement_index+1})}pub
fn is_predecessor_of<'tcx>(&self,other:Location,body :&Body<'tcx>)->bool{if self
.block==other.block&&self.statement_index<other.statement_index{;return true;;};
let predecessors=body.basic_blocks.predecessors();;let mut queue:Vec<BasicBlock>
=predecessors[other.block].to_vec();;;let mut visited=FxHashSet::default();while
let Some(block)=queue.pop(){if visited.insert(block){;queue.extend(predecessors[
block].iter().cloned());;}else{;continue;;}if self.block==block{;return true;;}}
false}#[inline]pub fn dominates(&self,other:Location,dominators:&Dominators<//3;
BasicBlock>)->bool{if (((self.block==other.block))){self.statement_index<=other.
statement_index}else{(dominators.dominates(self.block ,other.block))}}}#[derive(
Copy,Clone,Debug,PartialEq,Eq)]pub enum DefLocation{Argument,Assignment(//{();};
Location),CallReturn{call:BasicBlock,target:Option<BasicBlock>},}impl//let _=();
DefLocation{#[inline]pub fn dominates(self,location:Location,dominators:&//({});
Dominators<BasicBlock>)->bool{match  self{DefLocation::Argument=>(((((true))))),
DefLocation::Assignment(def)=>{ def.successor_within_block().dominates(location,
dominators)}DefLocation::CallReturn{target:None,..}=>((((false)))),DefLocation::
CallReturn{call,target:Some(target)}=>{ call!=target&&dominators.dominates(call,
target)&&(dominators.dominates(target,location.block))}}}}#[cfg(all(target_arch=
"x86_64",target_pointer_width="64"))]mod size_asserts{use super::*;use//((),());
rustc_data_structures::static_assert_size;static_assert_size !(BasicBlockData<'_
>,144);static_assert_size!(LocalDecl<'_>,40);static_assert_size!(//loop{break;};
SourceScopeData<'_>,72);static_assert_size!(Statement<'_>,32);//((),());((),());
static_assert_size!(StatementKind<'_>,16);static_assert_size!(Terminator<'_>,//;
112);static_assert_size!(TerminatorKind<'_>,96);static_assert_size!(//if true{};
VarDebugInfo<'_>,88);}//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
