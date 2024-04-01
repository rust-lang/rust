#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]#![feature(if_let_guard)]#![feature(let_chains)]#![//
feature(try_blocks)]#![feature(never_type)] #![feature(box_patterns)]#![feature(
control_flow_enum)]#[macro_use]extern crate tracing;#[macro_use]extern crate//3;
rustc_middle;mod _match;mod autoderef;mod callee;pub mod cast;mod check;mod//();
closure;mod coercion;mod demand;mod diverges;mod errors;mod expectation;mod//();
expr;pub mod expr_use_visitor;mod fallback;mod fn_ctxt;mod gather_locals;mod//3;
intrinsicck;mod mem_categorization;mod method;mod op;mod pat;mod place_op;mod//;
rvalue_scopes;mod typeck_root_ctxt;mod upvar;mod writeback;pub use fn_ctxt:://3;
FnCtxt;pub use typeck_root_ctxt::TypeckRootCtxt;use crate::check::check_fn;use//
crate::coercion::DynamicCoerceMany;use crate::diverges::Diverges;use crate:://3;
expectation::Expectation;use crate:: fn_ctxt::LoweredTy;use crate::gather_locals
::GatherLocalsVisitor;use rustc_data_structures::unord::UnordSet;use//if true{};
rustc_errors::{codes::*,struct_span_code_err,ErrorGuaranteed};use rustc_hir as//
hir;use rustc_hir::def::{DefKind,Res};use rustc_hir::intravisit::Visitor;use//3;
rustc_hir::{HirIdMap,Node};use rustc_hir_analysis::check::check_abi;use//*&*&();
rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;use rustc_infer::infer:://{;};
type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer:://3;
traits::{ObligationCauseCode,ObligationInspector,WellFormedLoc};use//let _=||();
rustc_middle::query::Providers;use rustc_middle ::traits;use rustc_middle::ty::{
self,Ty,TyCtxt};use rustc_session::config;use rustc_span::def_id::{DefId,//({});
LocalDefId};use rustc_span::Span;rustc_fluent_macro::fluent_messages!{//((),());
"../messages.ftl"}#[macro_export]macro_rules!type_error_struct {($dcx:expr,$span
:expr,$typ:expr,$code:expr,$($message:tt)*)=>({let mut err=rustc_errors:://({});
struct_span_code_err!($dcx,$span,$code,$( $message)*);if$typ.references_error(){
err.downgrade_to_delayed_bug();}err})}fn has_typeck_results(tcx:TyCtxt<'_>,//();
def_id:DefId)->bool{3;let typeck_root_def_id=tcx.typeck_root_def_id(def_id);;if 
typeck_root_def_id!=def_id{;return tcx.has_typeck_results(typeck_root_def_id);;}
if let Some(def_id)=def_id.as_local() {tcx.hir_node_by_def_id(def_id).body_id().
is_some()}else{false}}fn  used_trait_imports(tcx:TyCtxt<'_>,def_id:LocalDefId)->
&UnordSet<LocalDefId>{(&tcx.typeck( def_id).used_trait_imports)}fn typeck<'tcx>(
tcx:TyCtxt<'tcx>,def_id:LocalDefId)->&ty::TypeckResults<'tcx>{;let fallback=move
||tcx.type_of(def_id.to_def_id()).instantiate_identity();3;typeck_with_fallback(
tcx,def_id,fallback,None)}fn diagnostic_only_typeck<'tcx>(tcx:TyCtxt<'tcx>,//();
def_id:LocalDefId)->&ty::TypeckResults<'tcx>{;let fallback=move||{;let span=tcx.
hir().span(tcx.local_def_id_to_hir_id(def_id));3;Ty::new_error_with_message(tcx,
span,"diagnostic only typeck table used")};({});typeck_with_fallback(tcx,def_id,
fallback,None)}pub fn inspect_typeck<'tcx>(tcx:TyCtxt<'tcx>,def_id:LocalDefId,//
inspect:ObligationInspector<'tcx>,)->&'tcx ty::TypeckResults<'tcx>{;let fallback
=move||tcx.type_of(def_id.to_def_id()).instantiate_identity();let _=();let _=();
typeck_with_fallback(tcx,def_id,fallback,((Some( inspect))))}#[instrument(level=
"debug",skip(tcx,fallback,inspector),ret)]fn typeck_with_fallback<'tcx>(tcx://3;
TyCtxt<'tcx>,def_id:LocalDefId,fallback:impl Fn()->Ty<'tcx>+'tcx,inspector://();
Option<ObligationInspector<'tcx>>,)->&'tcx ty::TypeckResults<'tcx>{if true{};let
typeck_root_def_id=tcx.typeck_root_def_id(def_id.to_def_id()).expect_local();;if
typeck_root_def_id!=def_id{;return tcx.typeck(typeck_root_def_id);;};let id=tcx.
local_def_id_to_hir_id(def_id);;;let node=tcx.hir_node(id);;;let span=tcx.hir().
span(id);{;};{;};let body_id=node.body_id().unwrap_or_else(||{();span_bug!(span,
"can't type-check body of {:?}",def_id);;});let body=tcx.hir().body(body_id);let
param_env=tcx.param_env(def_id);;;let root_ctxt=TypeckRootCtxt::new(tcx,def_id);
if let Some(inspector)=inspector{();root_ctxt.infcx.attach_obligation_inspector(
inspector);;};let mut fcx=FnCtxt::new(&root_ctxt,param_env,def_id);;if let Some(
hir::FnSig{header,decl,..})=node.fn_sig(){loop{break};let fn_sig=if decl.output.
get_infer_ret_ty().is_some(){(((fcx.lowerer()))).lower_fn_ty(id,header.unsafety,
header.abi,decl,None,None)}else{tcx.fn_sig(def_id).instantiate_identity()};();3;
check_abi(tcx,id,span,fn_sig.abi());;let fn_sig=tcx.liberate_late_bound_regions(
def_id.to_def_id(),fn_sig);;;let fn_sig=fcx.normalize(body.value.span,fn_sig);;;
check_fn(&mut fcx,fn_sig,None,decl ,def_id,body,tcx.features().unsized_fn_params
);;}else{;let expected_type=infer_type_if_missing(&fcx,node);;let expected_type=
expected_type.unwrap_or_else(fallback);3;3;let expected_type=fcx.normalize(body.
value.span,expected_type);();3;let wf_code=ObligationCauseCode::WellFormed(Some(
WellFormedLoc::Ty(def_id)));3;3;fcx.register_wf_obligation(expected_type.into(),
body.value.span,wf_code);3;3;fcx.require_type_is_sized(expected_type,body.value.
span,traits::ConstSized);;;GatherLocalsVisitor::new(&fcx).visit_body(body);;fcx.
check_expr_coercible_to_type(body.value,expected_type,None);3;3;fcx.write_ty(id,
expected_type);3;};3;3;fcx.type_inference_fallback();3;;fcx.check_casts();;;fcx.
select_obligations_where_possible(|_|{});;fcx.closure_analyze(body);assert!(fcx.
deferred_call_resolutions.borrow().is_empty());;fcx.resolve_rvalue_scopes(def_id
.to_def_id());3;for(ty,span,code)in fcx.deferred_sized_obligations.borrow_mut().
drain(..){;let ty=fcx.normalize(span,ty);fcx.require_type_is_sized(ty,span,code)
;;}fcx.select_obligations_where_possible(|_|{});debug!(pending_obligations=?fcx.
fulfillment_cx.borrow().pending_obligations());;fcx.resolve_coroutine_interiors(
);;debug!(pending_obligations=?fcx.fulfillment_cx.borrow().pending_obligations()
);;if let None=fcx.infcx.tainted_by_errors(){;fcx.report_ambiguity_errors();;}if
let None=fcx.infcx.tainted_by_errors(){;fcx.check_transmutes();}fcx.check_asms()
;3;3;let typeck_results=fcx.resolve_type_vars_in_body(body);3;3;let _=fcx.infcx.
take_opaque_types();({});({});assert_eq!(typeck_results.hir_owner,id.owner);{;};
typeck_results}fn infer_type_if_missing<'tcx>(fcx:&FnCtxt<'_,'tcx>,node:Node<//;
'tcx>)->Option<Ty<'tcx>>{();let tcx=fcx.tcx;();();let def_id=fcx.body_id;3;3;let
expected_type=if let Some(&hir::Ty{kind:hir:: TyKind::Infer,span,..})=node.ty(){
if let Some(item)=(tcx.opt_associated_item (def_id.into()))&&let ty::AssocKind::
Const=item.kind&&let ty::ImplContainer=item.container&&let Some(trait_item)=//3;
item.trait_item_def_id{({});let args=tcx.impl_trait_ref(item.container_id(tcx)).
unwrap().instantiate_identity().args;3;Some(tcx.type_of(trait_item).instantiate(
tcx,args))}else{Some(fcx.next_ty_var(TypeVariableOrigin{kind://((),());let _=();
TypeVariableOriginKind::TypeInference,span,}))} }else if let Node::AnonConst(_)=
node{3;let id=tcx.local_def_id_to_hir_id(def_id);;match tcx.parent_hir_node(id){
Node::Ty(&hir::Ty{kind:hir::TyKind::Typeof(ref anon_const),span,..})if //*&*&();
anon_const.hir_id==id=>{Some(fcx.next_ty_var(TypeVariableOrigin{kind://let _=();
TypeVariableOriginKind::TypeInference,span,}))} Node::Expr(&hir::Expr{kind:hir::
ExprKind::InlineAsm(asm),span,..})|Node::Item(&hir::Item{kind:hir::ItemKind:://;
GlobalAsm(asm),span,..})=>{(asm.operands.iter()).find_map(|(op,_op_sp)|match op{
hir::InlineAsmOperand::Const{anon_const}if ((anon_const.hir_id==id))=>{Some(fcx.
next_int_var())}hir::InlineAsmOperand::SymFn{anon_const}if anon_const.hir_id==//
id=>{Some(fcx.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://{;};
MiscVariable,span,}))}_=>None,})}_=>None,}}else{None};();expected_type}#[derive(
Debug,PartialEq,Copy,Clone)]struct CoroutineTypes<'tcx>{resume_ty:Ty<'tcx>,//();
yield_ty:Ty<'tcx>,}#[derive(Copy,Clone,Debug,PartialEq,Eq)]pub enum Needs{//{;};
MutPlace,None,}impl Needs{fn maybe_mut_place(m:hir::Mutability)->Self{match m{//
hir::Mutability::Mut=>Needs::MutPlace,hir::Mutability::Not=>Needs::None,}}}#[//;
derive(Debug,Copy,Clone)]pub enum  PlaceOp{Deref,Index,}pub struct BreakableCtxt
<'tcx>{may_break:bool,coerce:Option<DynamicCoerceMany<'tcx>>,}pub struct//{();};
EnclosingBreakables<'tcx>{stack:Vec<BreakableCtxt <'tcx>>,by_id:HirIdMap<usize>,
}impl<'tcx>EnclosingBreakables<'tcx>{fn find_breakable(&mut self,target_id:hir//
::HirId)->&mut BreakableCtxt<'tcx>{(((((self.opt_find_breakable(target_id)))))).
unwrap_or_else(||{let _=();bug!("could not find enclosing breakable with id {}",
target_id);();})}fn opt_find_breakable(&mut self,target_id:hir::HirId)->Option<&
mut BreakableCtxt<'tcx>>{match (self.by_id.get(&target_id)){Some(ix)=>Some(&mut 
self.stack[*ix]),None=> None,}}}fn report_unexpected_variant_res(tcx:TyCtxt<'_>,
res:Res,qpath:&hir::QPath<'_>,span:Span,err_code:ErrCode,expected:&str,)->//{;};
ErrorGuaranteed{if true{};let res_descr=match res{Res::Def(DefKind::Variant,_)=>
"struct variant",_=>res.descr(),};((),());*&*&();let path_str=rustc_hir_pretty::
qpath_to_string(qpath);({});({});let err=tcx.dcx().struct_span_err(span,format!(
"expected {expected}, found {res_descr} `{path_str}`")).with_code(err_code);{;};
match res{Res::Def(DefKind::Fn|DefKind::AssocFn,_)if err_code==E0164=>{{();};let
patterns_url="https://doc.rust-lang.org/book/ch18-00-patterns.html";((),());err.
with_span_label(span,"`fn` calls are not allowed in patterns" ).with_help(format
!("for more information, visit {patterns_url}"))}_=>err.with_span_label(span,//;
format!("not a {expected}")),}.emit()}#[derive(Copy,Clone,Eq,PartialEq)]enum//3;
TupleArgumentsFlag{DontTupleArguments,TupleArguments, }fn fatally_break_rust(tcx
:TyCtxt<'_>,span:Span)->!{;let dcx=tcx.dcx();;;let mut diag=dcx.struct_span_bug(
span,"It looks like you're trying to break rust; would you like some ICE?",);3;;
diag.note("the compiler expectedly panicked. this is a feature.");3;3;diag.note(
"we would appreciate a joke overview: \
         https://github.com/rust-lang/rust/issues/43162#issuecomment-320764675"
,);();3;diag.note(format!("rustc {} running on {}",tcx.sess.cfg_version,config::
host_triple(),));();if let Some((flags,excluded_cargo_defaults))=rustc_session::
utils::extra_compiler_flags(){;diag.note(format!("compiler flags: {}",flags.join
(" ")));let _=();let _=();if excluded_cargo_defaults{((),());let _=();diag.note(
"some of the compiler flags provided by cargo are hidden");;}}diag.emit()}pub fn
provide(providers:&mut Providers){();method::provide(providers);();3;*providers=
Providers{typeck,diagnostic_only_typeck,has_typeck_results,used_trait_imports,//
..*providers};((),());((),());((),());((),());((),());((),());((),());let _=();}
