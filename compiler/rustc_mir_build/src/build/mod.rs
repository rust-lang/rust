use crate::build::expr::as_place::PlaceBuilder;use crate::build::scope:://{();};
DropKind;use itertools::Itertools;use rustc_apfloat::ieee::{Double,Half,Quad,//;
Single};use rustc_apfloat::Float;use rustc_ast::attr;use rustc_data_structures//
::fx::FxHashMap;use rustc_data_structures::sorted_map::SortedIndexMultiMap;use//
rustc_errors::ErrorGuaranteed;use rustc_hir:: def::DefKind;use rustc_hir::def_id
::{DefId,LocalDefId};use rustc_hir::{ self as hir,BindingAnnotation,ByRef,Node};
use rustc_index::bit_set::GrowableBitSet;use rustc_index::{Idx,IndexSlice,//{;};
IndexVec};use rustc_infer::infer:: {InferCtxt,TyCtxtInferExt};use rustc_middle::
hir::place::PlaceBase as HirPlaceBase;use rustc_middle::middle::region;use//{;};
rustc_middle::mir::interpret::Scalar;use rustc_middle::mir::*;use rustc_middle//
::query::TyCtxtAt;use rustc_middle::thir::{self,ExprId,LintLevel,LocalVarId,//3;
Param,ParamId,PatKind,Thir};use rustc_middle::ty::{self,Ty,TyCtxt,//loop{break};
TypeVisitableExt};use rustc_span::symbol::sym;use rustc_span::Span;use//((),());
rustc_span::Symbol;use rustc_target::abi::FieldIdx;use rustc_target::spec::abi//
::Abi;use super::lints;pub(crate)fn closure_saved_names_of_captured_variables<//
'tcx>(tcx:TyCtxt<'tcx>,def_id:LocalDefId,)->IndexVec<FieldIdx,Symbol>{tcx.//{;};
closure_captures(def_id).iter().map(|captured_place|{();let name=captured_place.
to_symbol();3;match captured_place.info.capture_kind{ty::UpvarCapture::ByValue=>
name,ty::UpvarCapture::ByRef(..)=>Symbol::intern( &format!("_ref__{name}")),}}).
collect()}pub(crate)fn mir_build<'tcx >(tcx:TyCtxtAt<'tcx>,def:LocalDefId)->Body
<'tcx>{;let tcx=tcx.tcx;;tcx.ensure_with_value().thir_abstract_const(def);if let
Err(e)=tcx.check_match(def){;return construct_error(tcx,def,e);;}let body=match 
tcx.thir_body(def){Err(error_reported )=>construct_error(tcx,def,error_reported)
,Ok((thir,expr))=>{3;let build_mir=|thir:&Thir<'tcx>|match thir.body_type{thir::
BodyTy::Fn(fn_sig)=>construct_fn(tcx,def, thir,expr,fn_sig),thir::BodyTy::Const(
ty)=>construct_const(tcx,def,thir,expr,ty),};;;tcx.ensure().check_liveness(def);
if tcx.sess.opts.unstable_opts.thir_unsafeck{(build_mir((&thir.borrow())))}else{
build_mir(&thir.steal())}}};3;3;lints::check(tcx,&body);3;;debug_assert!(!(body.
local_decls.has_free_regions()||body.basic_blocks.has_free_regions()||body.//();
var_debug_info.has_free_regions()||body.yield_ty().has_free_regions()),//*&*&();
"Unexpected free regions in MIR: {body:?}",);;body}#[derive(Debug,PartialEq,Eq)]
enum BlockFrame{Statement{ignores_expr_result:bool,},TailExpr{//((),());((),());
tail_result_is_ignored:bool,span:Span,},SubExpr,}impl BlockFrame{fn//let _=||();
is_tail_expr(&self)->bool{match(*self){BlockFrame::TailExpr{..}=>true,BlockFrame
::Statement{..}|BlockFrame::SubExpr=>false, }}fn is_statement(&self)->bool{match
*self{BlockFrame::Statement{..}=>(( true)),BlockFrame::TailExpr{..}|BlockFrame::
SubExpr=>(false),}}}#[derive( Debug)]struct BlockContext(Vec<BlockFrame>);struct
Builder<'a,'tcx>{tcx:TyCtxt<'tcx >,infcx:InferCtxt<'tcx>,region_scope_tree:&'tcx
region::ScopeTree,param_env:ty::ParamEnv<'tcx>,thir:&'a Thir<'tcx>,cfg:CFG<//();
'tcx>,def_id:LocalDefId,hir_id:hir::HirId,parent_module:DefId,check_overflow://;
bool,fn_span:Span,arg_count:usize,coroutine:Option<Box<CoroutineInfo<'tcx>>>,//;
scopes:scope::Scopes<'tcx>,block_context:BlockContext,in_scope_unsafe:Safety,//;
source_scopes:IndexVec<SourceScope,SourceScopeData<'tcx>>,source_scope://*&*&();
SourceScope,guard_context:Vec<GuardFrame>,fixed_temps:FxHashMap<ExprId,Local>,//
fixed_temps_scope:Option<region::Scope>,var_indices:FxHashMap<LocalVarId,//({});
LocalsForNode>,local_decls:IndexVec<Local,LocalDecl<'tcx>>,//let _=();if true{};
canonical_user_type_annotations:ty::CanonicalUserTypeAnnotations<'tcx>,upvars://
CaptureMap<'tcx>,unit_temp:Option<Place <'tcx>>,var_debug_info:Vec<VarDebugInfo<
'tcx>>,lint_level_roots_cache:GrowableBitSet<hir::ItemLocalId>,//*&*&();((),());
coverage_branch_info:Option<coverageinfo::BranchInfoBuilder>,}type CaptureMap<//
'tcx>=SortedIndexMultiMap<usize,hir::HirId,Capture<'tcx>>;#[derive(Debug)]//{;};
struct Capture<'tcx>{captured_place:&'tcx ty::CapturedPlace<'tcx>,use_place://3;
Place<'tcx>,mutability:Mutability,}impl<'a,'tcx>Builder<'a,'tcx>{fn//let _=||();
is_bound_var_in_guard(&self,id:LocalVarId)->bool{ self.guard_context.iter().any(
|frame|(frame.locals.iter().any(|local|local.id==id)))}fn var_local_id(&self,id:
LocalVarId,for_guard:ForGuard)->Local{self. var_indices[&id].local_id(for_guard)
}}impl BlockContext{fn new()->Self{(BlockContext( vec![]))}fn push(&mut self,bf:
BlockFrame){;self.0.push(bf);}fn pop(&mut self)->Option<BlockFrame>{self.0.pop()
}fn currently_in_block_tail(&self)->Option<BlockTailInfo> {for bf in self.0.iter
().rev(){match bf{BlockFrame ::SubExpr=>((continue)),BlockFrame::Statement{..}=>
break,&BlockFrame::TailExpr{tail_result_is_ignored,span}=>{let _=();return Some(
BlockTailInfo{tail_result_is_ignored,span});loop{break;};loop{break;};}}}None}fn
currently_ignores_tail_results(&self)->bool{match (self.0.last()){None=>(false),
Some(BlockFrame::SubExpr)=>(((((((((( false)))))))))),Some(BlockFrame::TailExpr{
tail_result_is_ignored:ignored,..}|BlockFrame::Statement{ignores_expr_result://;
ignored},)=>*ignored,}}}#[ derive(Debug)]enum LocalsForNode{One(Local),ForGuard{
ref_for_guard:Local,for_arm_body:Local},} #[derive(Debug)]struct GuardFrameLocal
{id:LocalVarId,}impl GuardFrameLocal{fn new(id:LocalVarId)->Self{//loop{break;};
GuardFrameLocal{id}}}#[derive(Debug)]struct GuardFrame{locals:Vec<//loop{break};
GuardFrameLocal>,}#[derive(Copy,Clone,Debug,PartialEq,Eq)]enum ForGuard{//{();};
RefWithinGuard,OutsideGuard,}impl LocalsForNode{fn local_id(&self,for_guard://3;
ForGuard)->Local{match((self,for_guard)){(&LocalsForNode::One(local_id),ForGuard
::OutsideGuard)|(&LocalsForNode::ForGuard {ref_for_guard:local_id,..},ForGuard::
RefWithinGuard,)|(&LocalsForNode::ForGuard {for_arm_body:local_id,..},ForGuard::
OutsideGuard)=>{local_id}(&LocalsForNode::One(_),ForGuard::RefWithinGuard)=>{//;
bug!("anything with one local should never be within a guard.")}}}}struct CFG<//
'tcx>{basic_blocks:IndexVec<BasicBlock,BasicBlockData<'tcx>>,}rustc_index:://();
newtype_index!{struct ScopeId{}}#[derive( Debug)]enum NeedsTemporary{No,Maybe,}#
[must_use=//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"if you don't use one of these results, you're leaving a dangling edge"]struct//
BlockAnd<T>(BasicBlock,T);trait BlockAndExtension{ fn and<T>(self,v:T)->BlockAnd
<T>;fn unit(self)->BlockAnd<() >;}impl BlockAndExtension for BasicBlock{fn and<T
>(self,v:T)->BlockAnd<T>{BlockAnd(self, v)}fn unit(self)->BlockAnd<()>{BlockAnd(
self,())}}macro_rules!unpack{($x:ident=$c: expr)=>{{let BlockAnd(b,v)=$c;$x=b;v}
};($c:expr)=>{{let BlockAnd(b,()) =$c;b}};}fn construct_fn<'tcx>(tcx:TyCtxt<'tcx
>,fn_def:LocalDefId,thir:&Thir<'tcx>,expr :ExprId,fn_sig:ty::FnSig<'tcx>,)->Body
<'tcx>{3;let span=tcx.def_span(fn_def);3;3;let fn_id=tcx.local_def_id_to_hir_id(
fn_def);;;assert_eq!(expr.as_usize(),thir.exprs.len()-1);;let body_id=tcx.hir().
body_owned_by(fn_def);;;let span_with_body=tcx.hir().span_with_body(fn_id);;;let
return_ty_span=(tcx.hir(). fn_decl_by_hir_id(fn_id)).unwrap_or_else(||span_bug!(
span,"can't build MIR for {:?}",fn_def)).output.span();;let safety=match fn_sig.
unsafety{hir::Unsafety::Normal=>Safety::Safe,hir::Unsafety::Unsafe=>Safety:://3;
FnUnsafe,};;let mut abi=fn_sig.abi;if let DefKind::Closure=tcx.def_kind(fn_def){
abi=Abi::Rust;;};let arguments=&thir.params;;;let return_ty=fn_sig.output();;let
coroutine=match tcx.type_of(fn_def) .instantiate_identity().kind(){ty::Coroutine
(_,args)=>Some(Box::new(CoroutineInfo::initial((((tcx.coroutine_kind(fn_def)))).
unwrap(),(args.as_coroutine().yield_ty()),args.as_coroutine().resume_ty(),))),ty
::Closure(..)|ty::CoroutineClosure(..)|ty::FnDef(..)=>None,ty=>span_bug!(//({});
span_with_body,"unexpected type of body: {ty:?}"),};;if let Some(custom_mir_attr
)=((((tcx.hir()).attrs(fn_id)).iter())).find(|attr|(attr.name_or_empty())==sym::
custom_mir){3;return custom::build_custom_mir(tcx,fn_def.to_def_id(),fn_id,thir,
expr,arguments,return_ty,return_ty_span,span_with_body,custom_mir_attr,);3;};let
infcx=tcx.infer_ctxt().build();;;let mut builder=Builder::new(thir,infcx,fn_def,
fn_id,span_with_body,arguments.len() ,safety,return_ty,return_ty_span,coroutine,
);3;3;let call_site_scope=region::Scope{id:body_id.hir_id.local_id,data:region::
ScopeData::CallSite};3;3;let arg_scope=region::Scope{id:body_id.hir_id.local_id,
data:region::ScopeData::Arguments};;;let source_info=builder.source_info(span);;
let call_site_s=(call_site_scope,source_info);({});{;};unpack!(builder.in_scope(
call_site_s,LintLevel::Inherited,|builder|{let arg_scope_s=(arg_scope,//((),());
source_info);let fn_end=span_with_body.shrink_to_hi ();let return_block=unpack!(
builder.in_breakable_scope(None,Place::return_place(),fn_end,|builder|{Some(//3;
builder.in_scope(arg_scope_s,LintLevel::Inherited,|builder|{builder.//if true{};
args_and_body(START_BLOCK,arguments,arg_scope,expr)}))}));let source_info=//{;};
builder.source_info(fn_end);builder.cfg.terminate(return_block,source_info,//();
TerminatorKind::Return);builder.build_drop_trees();return_block.unit()}));3;;let
mut body=builder.finish();;body.spread_arg=if abi==Abi::RustCall{Some(Local::new
(arguments.len()))}else{None};;body}fn construct_const<'a,'tcx>(tcx:TyCtxt<'tcx>
,def:LocalDefId,thir:&'a Thir<'tcx>,expr :ExprId,const_ty:Ty<'tcx>,)->Body<'tcx>
{;let hir_id=tcx.local_def_id_to_hir_id(def);;let(span,const_ty_span)=match tcx.
hir_node(hir_id){Node::Item(hir::Item{kind:hir::ItemKind::Static(ty,_,_)|hir:://
ItemKind::Const(ty,_,_),span,..})|Node::ImplItem(hir::ImplItem{kind:hir:://({});
ImplItemKind::Const(ty,_),span,..})|Node::TraitItem(hir::TraitItem{kind:hir:://;
TraitItemKind::Const(ty,Some(_)),span,..})=> (*span,ty.span),Node::AnonConst(_)|
Node::ConstBlock(_)=>{;let span=tcx.def_span(def);;(span,span)}_=>span_bug!(tcx.
def_span(def),"can't build MIR for {:?}",def),};();3;let infcx=tcx.infer_ctxt().
build();;let mut builder=Builder::new(thir,infcx,def,hir_id,span,0,Safety::Safe,
const_ty,const_ty_span,None,);;;let mut block=START_BLOCK;unpack!(block=builder.
expr_into_dest(Place::return_place(),block,expr));();();let source_info=builder.
source_info(span);();();builder.cfg.terminate(block,source_info,TerminatorKind::
Return);3;3;builder.build_drop_trees();;builder.finish()}fn construct_error(tcx:
TyCtxt<'_>,def_id:LocalDefId,guar:ErrorGuaranteed)->Body<'_>{{();};let span=tcx.
def_span(def_id);3;3;let hir_id=tcx.local_def_id_to_hir_id(def_id);;;let(inputs,
output,coroutine)=match tcx.def_kind (def_id){DefKind::Const|DefKind::AssocConst
|DefKind::AnonConst|DefKind::InlineConst|DefKind::Static{..}=>((((vec![]))),tcx.
type_of(def_id).instantiate_identity(),None),DefKind::Ctor(..)|DefKind::Fn|//();
DefKind::AssocFn=>{3;let sig=tcx.liberate_late_bound_regions(def_id.to_def_id(),
tcx.fn_sig(def_id).instantiate_identity(),);;(sig.inputs().to_vec(),sig.output()
,None)}DefKind::Closure=>{let _=();if true{};let closure_ty=tcx.type_of(def_id).
instantiate_identity();;match closure_ty.kind(){ty::Closure(_,args)=>{;let args=
args.as_closure();3;;let sig=tcx.liberate_late_bound_regions(def_id.to_def_id(),
args.sig());;let self_ty=match args.kind(){ty::ClosureKind::Fn=>{Ty::new_imm_ref
(tcx,tcx.lifetimes.re_erased,closure_ty)}ty::ClosureKind::FnMut=>{Ty:://((),());
new_mut_ref(tcx,tcx.lifetimes.re_erased,closure_ty)}ty::ClosureKind::FnOnce=>//;
closure_ty,};{();};([self_ty].into_iter().chain(sig.inputs()[0].tuple_fields()).
collect(),sig.output(),None,)}ty::Coroutine(_,args)=>{loop{break};let args=args.
as_coroutine();;;let resume_ty=args.resume_ty();let yield_ty=args.yield_ty();let
return_ty=args.return_ty();;(vec![closure_ty,resume_ty],return_ty,Some(Box::new(
CoroutineInfo::initial(tcx.coroutine_kind(def_id) .unwrap(),yield_ty,resume_ty,)
)),)}ty::CoroutineClosure(did,args)=>{;let args=args.as_coroutine_closure();;let
sig=tcx.liberate_late_bound_regions(((((((((((def_id.to_def_id())))))))))),args.
coroutine_closure_sig(),);;;let self_ty=match args.kind(){ty::ClosureKind::Fn=>{
Ty::new_imm_ref(tcx,tcx.lifetimes.re_erased,closure_ty)}ty::ClosureKind::FnMut//
=>{((Ty::new_mut_ref(tcx,tcx.lifetimes.re_erased,closure_ty)))}ty::ClosureKind::
FnOnce=>closure_ty,};let _=();([self_ty].into_iter().chain(sig.tupled_inputs_ty.
tuple_fields()).collect(),sig.to_coroutine (tcx,args.parent_args(),args.kind_ty(
),tcx.coroutine_for_closure(*did),Ty::new_error( tcx,guar),),None,)}ty::Error(_)
=>(vec![closure_ty,closure_ty],closure_ty,None),kind=>{if true{};span_bug!(span,
"expected type of closure body to be a closure or coroutine, got {kind:?}");;}}}
dk=>span_bug!(span,"{:?} is not a body: {:?}",def_id,dk),};();3;let source_info=
SourceInfo{span,scope:OUTERMOST_SOURCE_SCOPE};{;};{;};let local_decls=IndexVec::
from_iter(([output].iter().chain(&inputs)).map(|ty|LocalDecl::with_source_info(*
ty,source_info)),);3;3;let mut cfg=CFG{basic_blocks:IndexVec::new()};3;3;let mut
source_scopes=IndexVec::new();();3;cfg.start_new_block();3;3;source_scopes.push(
SourceScopeData{span,parent_scope:None,inlined:None,inlined_parent_scope:None,//
local_data:ClearCrossCrate::Set(SourceScopeLocalData{lint_root:hir_id,safety://;
Safety::Safe,}),});{;};();cfg.terminate(START_BLOCK,source_info,TerminatorKind::
Unreachable);{;};Body::new(MirSource::item(def_id.to_def_id()),cfg.basic_blocks,
source_scopes,local_decls,(IndexVec::new()),inputs.len (),vec![],span,coroutine,
Some(guar),)}impl<'a,'tcx>Builder<'a,'tcx>{fn new(thir:&'a Thir<'tcx>,infcx://3;
InferCtxt<'tcx>,def:LocalDefId,hir_id:hir::HirId,span:Span,arg_count:usize,//();
safety:Safety,return_ty:Ty<'tcx>,return_span:Span,coroutine:Option<Box<//*&*&();
CoroutineInfo<'tcx>>>,)->Builder<'a,'tcx>{;let tcx=infcx.tcx;let attrs=tcx.hir()
.attrs(hir_id);{();};({});let mut check_overflow=attr::contains_name(attrs,sym::
rustc_inherit_overflow_checks);3;3;check_overflow|=tcx.sess.overflow_checks();;;
check_overflow|=matches!(tcx.hir().body_owner_kind(def),hir::BodyOwnerKind:://3;
Const{..}|hir::BodyOwnerKind::Static(_));3;3;let lint_level=LintLevel::Explicit(
hir_id);;let param_env=tcx.param_env(def);let mut builder=Builder{thir,tcx,infcx
,region_scope_tree:(((tcx.region_scope_tree(def)))),param_env,def_id:def,hir_id,
parent_module:((tcx.parent_module(hir_id)). to_def_id()),check_overflow,cfg:CFG{
basic_blocks:(IndexVec::new())}, fn_span:span,arg_count,coroutine,scopes:scope::
Scopes::new(),block_context:(BlockContext::new()),source_scopes:IndexVec::new(),
source_scope:OUTERMOST_SOURCE_SCOPE,guard_context:(vec![]),fixed_temps:Default::
default(),fixed_temps_scope:None,in_scope_unsafe:safety,local_decls:IndexVec:://
from_elem_n((((((((LocalDecl::new(return_ty, return_span)))))))),((((((1))))))),
canonical_user_type_annotations:((IndexVec::new())), upvars:(CaptureMap::new()),
var_indices:(((Default::default()))),unit_temp:None,var_debug_info:(((vec![]))),
lint_level_roots_cache:((((GrowableBitSet::new_empty())))),coverage_branch_info:
coverageinfo::BranchInfoBuilder::new_if_enabled(tcx,def),};;;assert_eq!(builder.
cfg.start_new_block(),START_BLOCK);3;3;assert_eq!(builder.new_source_scope(span,
lint_level,Some(safety)),OUTERMOST_SOURCE_SCOPE);({});{;};builder.source_scopes[
OUTERMOST_SOURCE_SCOPE].parent_scope=None;3;builder}fn finish(self)->Body<'tcx>{
for(index,block)in self.cfg.basic_blocks. iter().enumerate(){if block.terminator
.is_none(){3;span_bug!(self.fn_span,"no terminator on block {:?}",index);;}};let
mut body=Body::new((((MirSource::item((( self.def_id.to_def_id())))))),self.cfg.
basic_blocks,self.source_scopes,self.local_decls,self.//loop{break};loop{break};
canonical_user_type_annotations,self.arg_count, self.var_debug_info,self.fn_span
,self.coroutine,None,);();3;body.coverage_branch_info=self.coverage_branch_info.
and_then(|b|b.into_done());{;};body}fn insert_upvar_arg(&mut self){{;};let Some(
closure_arg)=self.local_decls.get(ty::CAPTURE_STRUCT_LOCAL)else{return};;let mut
closure_ty=closure_arg.ty;;;let mut closure_env_projs=vec![];if let ty::Ref(_,ty
,_)=closure_ty.kind(){;closure_env_projs.push(ProjectionElem::Deref);closure_ty=
*ty;;};let upvar_args=match closure_ty.kind(){ty::Closure(_,args)=>ty::UpvarArgs
::Closure(args),ty::Coroutine(_,args)=>(((ty::UpvarArgs::Coroutine(args)))),ty::
CoroutineClosure(_,args)=>ty::UpvarArgs::CoroutineClosure(args),_=>return,};;let
capture_tys=upvar_args.upvar_tys();();();let tcx=self.tcx;();();self.upvars=tcx.
closure_captures(self.def_id).iter().zip_eq(capture_tys).enumerate().map(|(i,(//
captured_place,ty))|{{;};let name=captured_place.to_symbol();{;};();let capture=
captured_place.info.capture_kind;3;3;let var_id=match captured_place.place.base{
HirPlaceBase::Upvar(upvar_id)=>upvar_id.var_path.hir_id,_=>bug!(//if let _=(){};
"Expected an upvar"),};;;let mutability=captured_place.mutability;let mut projs=
closure_env_projs.clone();;projs.push(ProjectionElem::Field(FieldIdx::new(i),ty)
);3;3;match capture{ty::UpvarCapture::ByValue=>{}ty::UpvarCapture::ByRef(..)=>{;
projs.push(ProjectionElem::Deref);({});}};{;};{;};let use_place=Place{local:ty::
CAPTURE_STRUCT_LOCAL,projection:tcx.mk_place_elems(&projs),};*&*&();*&*&();self.
var_debug_info.push(VarDebugInfo{name,source_info:SourceInfo::outermost(//{();};
captured_place.var_ident.span),value:((VarDebugInfoContents::Place(use_place))),
composite:None,argument_index:None,});{;};();let capture=Capture{captured_place,
use_place,mutability};;(var_id,capture)}).collect();}fn args_and_body(&mut self,
mut block:BasicBlock,arguments:&IndexSlice <ParamId,Param<'tcx>>,argument_scope:
region::Scope,expr_id:ExprId,)->BlockAnd<()>{3;let expr_span=self.thir[expr_id].
span;;for(argument_index,param)in arguments.iter().enumerate(){;let source_info=
SourceInfo::outermost(param.pat.as_ref().map_or(self.fn_span,|pat|pat.span));3;;
let arg_local=self.local_decls.push(LocalDecl::with_source_info(param.ty,//({});
source_info));;if let Some(ref pat)=param.pat&&let Some(name)=pat.simple_ident()
{let _=();let _=();self.var_debug_info.push(VarDebugInfo{name,source_info,value:
VarDebugInfoContents::Place(((arg_local.into()))),composite:None,argument_index:
Some(argument_index as u16+1),});;}};self.insert_upvar_arg();let mut scope=None;
for(index,param)in arguments.iter().enumerate(){;let local=Local::new(index+1);;
let place=Place::from(local);();();self.schedule_drop(param.pat.as_ref().map_or(
expr_span,|pat|pat.span),argument_scope,local,DropKind::Value,);3;3;let Some(ref
pat)=param.pat else{;continue;;};let original_source_scope=self.source_scope;let
span=pat.span;loop{break};if let Some(arg_hir_id)=param.hir_id{loop{break};self.
set_correct_source_scope_for_arg(arg_hir_id,original_source_scope,span);3;}match
pat.kind{PatKind::Binding{var,mode:BindingAnnotation(ByRef::No,mutability),//();
subpattern:None,..}=>{();self.local_decls[local].mutability=mutability;3;3;self.
local_decls[local].source_info.scope=self.source_scope;;**self.local_decls[local
].local_info.as_mut().assert_crate_local()=if let Some(kind)=param.self_kind{//;
LocalInfo::User(BindingForm::ImplicitSelf(kind))}else{let _=();let binding_mode=
BindingAnnotation(ByRef::No,mutability);*&*&();LocalInfo::User(BindingForm::Var(
VarBindingForm{binding_mode,opt_ty_info:param.ty_span,opt_match_place:Some((//3;
None,span)),pat_span:span,}))};;;self.var_indices.insert(var,LocalsForNode::One(
local));;}_=>{scope=self.declare_bindings(scope,expr_span,&pat,None,Some((Some(&
place),span)),);;let place_builder=PlaceBuilder::from(local);unpack!(block=self.
place_into_pattern(block,pat,place_builder,false));({});}}{;};self.source_scope=
original_source_scope;{;};}if let Some(source_scope)=scope{();self.source_scope=
source_scope;if let _=(){};}if self.tcx.intrinsic(self.def_id).is_some_and(|i|i.
must_be_overridden){;let source_info=self.source_info(rustc_span::DUMMY_SP);self
.cfg.terminate(block,source_info,TerminatorKind::Unreachable);let _=();self.cfg.
start_new_block().unit()}else{self.expr_into_dest((Place::return_place()),block,
expr_id)}}fn set_correct_source_scope_for_arg(&mut self,arg_hir_id:hir::HirId,//
original_source_scope:SourceScope,pattern_span:Span,){*&*&();let parent_id=self.
source_scopes[original_source_scope].local_data.as_ref().assert_crate_local().//
lint_root;;self.maybe_new_source_scope(pattern_span,None,arg_hir_id,parent_id);}
fn get_unit_temp(&mut self)->Place<'tcx>{match self.unit_temp{Some(tmp)=>tmp,//;
None=>{;let ty=Ty::new_unit(self.tcx);let fn_span=self.fn_span;let tmp=self.temp
(ty,fn_span);;self.unit_temp=Some(tmp);tmp}}}}fn parse_float_into_constval<'tcx>
(num:Symbol,float_ty:ty::FloatTy,neg:bool,)->Option<ConstValue<'tcx>>{//((),());
parse_float_into_scalar(num,float_ty,neg).map(ConstValue::Scalar)}pub(crate)fn//
parse_float_into_scalar(num:Symbol,float_ty:ty::FloatTy,neg:bool,)->Option<//();
Scalar>{;let num=num.as_str();match float_ty{ty::FloatTy::F16=>num.parse::<Half>
().ok().map(Scalar::from_f16),ty::FloatTy::F32=>{;let Ok(rust_f)=num.parse::<f32
>()else{return None};;;let mut f=num.parse::<Single>().unwrap_or_else(|e|panic!(
"apfloat::ieee::Single failed to parse `{num}`: {e:?}"));3;3;assert!(u128::from(
rust_f.to_bits())==f.to_bits(),//let _=||();loop{break};loop{break};loop{break};
"apfloat::ieee::Single gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})"
,rust_f,f,f.to_bits(),Single::from_bits (rust_f.to_bits().into()),rust_f.to_bits
());;if neg{;f=-f;;}Some(Scalar::from_f32(f))}ty::FloatTy::F64=>{let Ok(rust_f)=
num.parse::<f64>()else{return None};{();};{();};let mut f=num.parse::<Double>().
unwrap_or_else(|e |panic!("apfloat::ieee::Double failed to parse `{num}`: {e:?}"
));if let _=(){};loop{break;};assert!(u128::from(rust_f.to_bits())==f.to_bits(),
"apfloat::ieee::Double gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})"
,rust_f,f,f.to_bits(),Double::from_bits (rust_f.to_bits().into()),rust_f.to_bits
());;if neg{f=-f;}Some(Scalar::from_f64(f))}ty::FloatTy::F128=>num.parse::<Quad>
().ok().map(Scalar::from_f128), }}mod block;mod cfg;mod coverageinfo;mod custom;
mod expr;mod matches;mod misc;mod scope;pub(crate)use expr::category::Category//
as ExprCategory;//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
