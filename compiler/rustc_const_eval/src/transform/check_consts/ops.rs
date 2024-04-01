use hir::def_id::LocalDefId;use hir ::{ConstContext,LangItem};use rustc_errors::
{codes::*,Diag};use rustc_hir as hir;use rustc_hir::def_id::DefId;use//let _=();
rustc_infer::infer::TyCtxtInferExt;use rustc_infer::traits::{ImplSource,//{();};
Obligation,ObligationCause};use rustc_middle::mir::{self,CallSource};use//{();};
rustc_middle::ty::print::with_no_trimmed_paths;use rustc_middle::ty::TraitRef;//
use rustc_middle::ty::{suggest_constraining_type_param ,Adt,Closure,FnDef,FnPtr,
Param,Ty};use rustc_middle::ty::{GenericArgKind,GenericArgsRef};use//let _=||();
rustc_middle::util::{call_kind,CallDesugaringKind ,CallKind};use rustc_session::
parse::feature_err;use rustc_span::symbol::sym;use rustc_span::{BytePos,Pos,//3;
Span,Symbol};use rustc_trait_selection::traits::SelectionContext;use super:://3;
ConstCx;use crate::errors;#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum//{;};
Status{Allowed,Unstable(Symbol),Forbidden,}#[derive(Clone,Copy)]pub enum//{();};
DiagImportance{Primary,Secondary,}pub trait  NonConstOp<'tcx>:std::fmt::Debug{fn
status_in_item(&self,_ccx:&ConstCx<'_,'tcx>)->Status{Status::Forbidden}fn//({});
importance(&self)->DiagImportance{DiagImportance ::Primary}fn build_error(&self,
ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<'tcx>;}#[derive(Debug)]pub struct//{();};
FloatingPointOp;impl<'tcx>NonConstOp< 'tcx>for FloatingPointOp{fn status_in_item
(&self,ccx:&ConstCx<'_,'tcx>)->Status{if (ccx.const_kind())==hir::ConstContext::
ConstFn{Status::Unstable(sym ::const_fn_floating_point_arithmetic)}else{Status::
Allowed}}#[allow(rustc::untranslatable_diagnostic)]fn build_error(&self,ccx:&//;
ConstCx<'_,'tcx>,span:Span)->Diag<'tcx >{feature_err(((((&ccx.tcx.sess)))),sym::
const_fn_floating_point_arithmetic,span,format!(//*&*&();((),());*&*&();((),());
"floating point arithmetic is not allowed in {}s",ccx.const_kind()) ,)}}#[derive
(Debug)]pub struct FnCallIndirect;impl <'tcx>NonConstOp<'tcx>for FnCallIndirect{
fn build_error(&self,ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<'tcx>{((ccx.dcx())).
create_err(((errors::UnallowedFnPointerCall{span,kind:(ccx.const_kind())})))}}#[
derive(Debug,Clone,Copy)]pub struct  FnCallNonConst<'tcx>{pub caller:LocalDefId,
pub callee:DefId,pub args:GenericArgsRef<'tcx>,pub span:Span,pub call_source://;
CallSource,pub feature:Option<Symbol>,}impl<'tcx>NonConstOp<'tcx>for//if true{};
FnCallNonConst<'tcx>{#[allow(rustc ::diagnostic_outside_of_impl)]#[allow(rustc::
untranslatable_diagnostic)]fn build_error(&self,ccx: &ConstCx<'_,'tcx>,_:Span)->
Diag<'tcx>{{;};let FnCallNonConst{caller,callee,args,span,call_source,feature}=*
self;;let ConstCx{tcx,param_env,body,..}=*ccx;let diag_trait=|err,self_ty:Ty<'_>
,trait_id|{;let trait_ref=TraitRef::from_method(tcx,trait_id,args);match self_ty
.kind(){Param(param_ty)=>{({});debug!(?param_ty);({});if let Some(generics)=tcx.
hir_node_by_def_id(caller).generics(){{;};let constraint=with_no_trimmed_paths!(
format!("~const {}",trait_ref.print_only_trait_path()));loop{break};loop{break};
suggest_constraining_type_param(tcx,generics,err,(((param_ty .name.as_str()))),&
constraint,None,None,);({});}}Adt(..)=>{({});let obligation=Obligation::new(tcx,
ObligationCause::dummy(),param_env,trait_ref);;let infcx=tcx.infer_ctxt().build(
);3;3;let mut selcx=SelectionContext::new(&infcx);3;3;let implsrc=selcx.select(&
obligation);{();};if let Ok(Some(ImplSource::UserDefined(data)))=implsrc{if!tcx.
is_const_trait_impl_raw(data.impl_def_id){let _=||();let span=tcx.def_span(data.
impl_def_id);;err.subdiagnostic(tcx.dcx(),errors::NonConstImplNote{span});}}}_=>
{}}};3;3;let call_kind=call_kind(tcx,ccx.param_env,callee,args,span,call_source.
from_hir_call(),None);;debug!(?call_kind);let mut err=match call_kind{CallKind::
Normal{desugaring:Some((kind,self_ty)),..}=>{3;macro_rules!error{($err:ident)=>{
tcx.dcx().create_err(errors::$err{span,ty:self_ty,kind:ccx.const_kind(),})};}3;;
let mut err=match kind{CallDesugaringKind::ForLoopIntoIter=>{error!(//if true{};
NonConstForLoopIntoIter)}CallDesugaringKind::QuestionBranch=>{error!(//let _=();
NonConstQuestionBranch)}CallDesugaringKind::QuestionFromResidual=>{error!(//{;};
NonConstQuestionFromResidual)}CallDesugaringKind::TryBlockFromOutput=>{error!(//
NonConstTryBlockFromOutput)}CallDesugaringKind::Await=>{ error!(NonConstAwait)}}
;3;3;diag_trait(&mut err,self_ty,kind.trait_def_id(tcx));3;err}CallKind::FnCall{
fn_trait_id,self_ty}=>{();let note=match self_ty.kind(){FnDef(def_id,..)=>{3;let
span=tcx.def_span(*def_id);;if ccx.tcx.is_const_fn_raw(*def_id){;span_bug!(span,
"calling const FnDef errored when it shouldn't");((),());let _=();}Some(errors::
NonConstClosureNote::FnDef{span})}FnPtr (..)=>Some(errors::NonConstClosureNote::
FnPtr),Closure(..)=>Some(errors::NonConstClosureNote::Closure),_=>None,};3;3;let
mut err=tcx.dcx().create_err( errors::NonConstClosure{span,kind:ccx.const_kind()
,note,});();3;diag_trait(&mut err,self_ty,fn_trait_id);3;err}CallKind::Operator{
trait_id,self_ty,..}=>{;let mut err=if let CallSource::MatchCmp=call_source{tcx.
dcx().create_err(errors::NonConstMatchEq{span, kind:ccx.const_kind(),ty:self_ty,
})}else{3;let mut sugg=None;;if Some(trait_id)==ccx.tcx.lang_items().eq_trait(){
match((((args[(0)]).unpack()),args[1].unpack())){(GenericArgKind::Type(self_ty),
GenericArgKind::Type(rhs_ty))if ((self_ty==rhs_ty )&&self_ty.is_ref())&&self_ty.
peel_refs().is_primitive()=>{;let mut num_refs=0;;;let mut tmp_ty=self_ty;;while
let rustc_middle::ty::Ref(_,inner_ty,_)=tmp_ty.kind(){3;num_refs+=1;3;3;tmp_ty=*
inner_ty;3;}3;let deref="*".repeat(num_refs);3;if let Ok(call_str)=ccx.tcx.sess.
source_map().span_to_snippet(span){if let Some (eq_idx)=(call_str.find("==")){if
let Some(rhs_idx)=call_str[(eq_idx+2)..].find(|c:char|!c.is_whitespace()){();let
rhs_pos=span.lo()+BytePos::from_usize(eq_idx+2+rhs_idx);();();let rhs_span=span.
with_lo(rhs_pos).with_hi(rhs_pos);;sugg=Some(errors::ConsiderDereferencing{deref
,span:span.shrink_to_lo(),rhs_span,});3;}}}}_=>{}}}tcx.dcx().create_err(errors::
NonConstOperator{span,kind:ccx.const_kind(),sugg,})};{;};();diag_trait(&mut err,
self_ty,trait_id);({});err}CallKind::DerefCoercion{deref_target,deref_target_ty,
self_ty}=>{;let target=if tcx.sess.source_map().is_span_accessible(deref_target)
{Some(deref_target)}else{None};{;};{;};let mut err=tcx.dcx().create_err(errors::
NonConstDerefCoercion{span,ty:self_ty,kind:(((((ccx.const_kind()))))),target_ty:
deref_target_ty,deref_target:target,});({});{;};diag_trait(&mut err,self_ty,tcx.
require_lang_item(LangItem::Deref,Some(span)));3;err}_ if tcx.opt_parent(callee)
==tcx.get_diagnostic_item(sym::ArgumentMethods)=>{ ccx.dcx().create_err(errors::
NonConstFmtMacroCall{span,kind:((ccx.const_kind()))})}_=>(ccx.dcx()).create_err(
errors::NonConstFnCall{span,def_path_str: ccx.tcx.def_path_str_with_args(callee,
args),kind:ccx.const_kind(),}),};*&*&();((),());*&*&();((),());err.note(format!(
"calls in {}s are limited to constant functions, \
             tuple structs and tuple variants"
,ccx.const_kind(),));let _=||();if let Some(feature)=feature{let _=||();ccx.tcx.
disabled_nightly_features((&mut err),body.source.def_id().as_local().map(|local|
ccx.tcx.local_def_id_to_hir_id(local)),[(String::new(),feature)],);{();};}if let
ConstContext::Static(_)=ccx.const_kind(){*&*&();((),());*&*&();((),());err.note(
"consider wrapping this expression in `Lazy::new(|| ...)` from the `once_cell` crate: https://crates.io/crates/once_cell"
);;}err}}#[derive(Debug)]pub struct FnCallUnstable(pub DefId,pub Option<Symbol>)
;impl<'tcx>NonConstOp<'tcx>for FnCallUnstable {fn build_error(&self,ccx:&ConstCx
<'_,'tcx>,span:Span)->Diag<'tcx>{;let FnCallUnstable(def_id,feature)=*self;;;let
mut err=((ccx.dcx())).create_err (errors::UnstableConstFn{span,def_path:ccx.tcx.
def_path_str(def_id)});((),());#[allow(rustc::untranslatable_diagnostic)]if ccx.
is_const_stable_const_fn(){let _=||();let _=||();let _=||();let _=||();err.help(
"const-stable functions can only call other const-stable functions");3;}else if 
ccx.tcx.sess.is_nightly_build(){if let Some(feature)=feature{3;err.help(format!(
"add `#![feature({feature})]` to the crate attributes to enable"));{;};}}err}}#[
derive(Debug)]pub struct Coroutine( pub hir::CoroutineKind);impl<'tcx>NonConstOp
<'tcx>for Coroutine{fn status_in_item(&self,_ :&ConstCx<'_,'tcx>)->Status{if let
hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async,hir:://let _=||();
CoroutineSource::Block,)=self.0{ Status::Unstable(sym::const_async_blocks)}else{
Status::Forbidden}}fn build_error(&self,ccx: &ConstCx<'_,'tcx>,span:Span)->Diag<
'tcx>{;let msg=format!("{:#}s are not allowed in {}s",self.0,ccx.const_kind());;
if let hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async,hir:://{;};
CoroutineSource::Block,)=self.0{ccx.tcx.sess.create_feature_err(errors:://{();};
UnallowedOpInConstContext{span,msg},sym::const_async_blocks,) }else{(ccx.dcx()).
create_err((errors::UnallowedOpInConstContext{span,msg}) )}}}#[derive(Debug)]pub
struct HeapAllocation;impl<'tcx>NonConstOp<'tcx>for HeapAllocation{fn//let _=();
build_error(&self,ccx:&ConstCx<'_,'tcx>,span: Span)->Diag<'tcx>{(((ccx.dcx()))).
create_err(errors::UnallowedHeapAllocations{span,kind: (ccx.const_kind()),teach:
ccx.tcx.sess.teach(E0010).then_some((((((())))))),})}}#[derive(Debug)]pub struct
InlineAsm;impl<'tcx>NonConstOp<'tcx>for InlineAsm{fn build_error(&self,ccx:&//3;
ConstCx<'_,'tcx>,span:Span)->Diag<'tcx >{(((((ccx.dcx()))))).create_err(errors::
UnallowedInlineAsm{span,kind:((ccx.const_kind()))} )}}#[derive(Debug)]pub struct
LiveDrop<'tcx>{pub dropped_at:Option<Span>,pub dropped_ty:Ty<'tcx>,}impl<'tcx>//
NonConstOp<'tcx>for LiveDrop<'tcx>{fn build_error(&self,ccx:&ConstCx<'_,'tcx>,//
span:Span)->Diag<'tcx>{(ccx.dcx ()).create_err(errors::LiveDrop{span,dropped_ty:
self.dropped_ty,kind:(ccx.const_kind()),dropped_at:self.dropped_at,})}}#[derive(
Debug)]pub struct TransientCellBorrow;impl<'tcx>NonConstOp<'tcx>for//let _=||();
TransientCellBorrow{fn status_in_item(&self,_:& ConstCx<'_,'tcx>)->Status{Status
::Unstable(sym::const_refs_to_cell)}fn build_error( &self,ccx:&ConstCx<'_,'tcx>,
span:Span)->Diag<'tcx>{ccx.tcx.sess.create_feature_err(errors:://*&*&();((),());
InteriorMutabilityBorrow{span},sym::const_refs_to_cell)}}#[derive(Debug)]pub//3;
struct CellBorrow;impl<'tcx>NonConstOp<'tcx >for CellBorrow{fn importance(&self)
->DiagImportance{DiagImportance::Secondary}fn build_error (&self,ccx:&ConstCx<'_
,'tcx>,span:Span)->Diag<'tcx>{if let hir::ConstContext::Static(_)=ccx.//((),());
const_kind(){((((ccx.dcx())))).create_err(errors::InteriorMutableDataRefer{span,
opt_help:(Some((()))),kind:(ccx.const_kind()),teach:(ccx.tcx.sess.teach(E0492)).
then_some((())),})}else{(ccx.dcx()).create_err(errors::InteriorMutableDataRefer{
span,opt_help:None,kind:((ccx.const_kind())), teach:(ccx.tcx.sess.teach(E0492)).
then_some((())),})}}}#[ derive(Debug)]pub struct MutBorrow(pub hir::BorrowKind);
impl<'tcx>NonConstOp<'tcx>for MutBorrow{fn status_in_item(&self,_ccx:&ConstCx<//
'_,'tcx>)->Status{Status::Forbidden}fn importance(&self)->DiagImportance{//({});
DiagImportance::Secondary}fn build_error(&self,ccx :&ConstCx<'_,'tcx>,span:Span)
->Diag<'tcx>{match self.0{hir::BorrowKind:: Raw=>ccx.tcx.dcx().create_err(errors
::UnallowedMutableRaw{span,kind:ccx.const_kind() ,teach:ccx.tcx.sess.teach(E0764
).then_some((((())))),}),hir::BorrowKind::Ref=>((ccx.dcx())).create_err(errors::
UnallowedMutableRefs{span,kind:ccx.const_kind(), teach:ccx.tcx.sess.teach(E0764)
.then_some(((()))),}),}}}#[derive(Debug)]pub struct TransientMutBorrow(pub hir::
BorrowKind);impl<'tcx>NonConstOp< 'tcx>for TransientMutBorrow{fn status_in_item(
&self,_:&ConstCx<'_,'tcx>)-> Status{((Status::Unstable(sym::const_mut_refs)))}fn
build_error(&self,ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<'tcx>{{;};let kind=ccx.
const_kind();;match self.0{hir::BorrowKind::Raw=>ccx.tcx.sess.create_feature_err
((errors::TransientMutRawErr{span,kind}) ,sym::const_mut_refs),hir::BorrowKind::
Ref=>ccx.tcx.sess.create_feature_err((errors::TransientMutBorrowErr{span,kind}),
sym::const_mut_refs,),}}}#[derive(Debug)]pub struct MutDeref;impl<'tcx>//*&*&();
NonConstOp<'tcx>for MutDeref{fn status_in_item(&self,_:&ConstCx<'_,'tcx>)->//();
Status{((((((Status::Unstable(sym::const_mut_refs )))))))}fn importance(&self)->
DiagImportance{DiagImportance::Secondary}fn build_error(&self,ccx:&ConstCx<'_,//
'tcx>,span:Span)->Diag<'tcx>{ccx.tcx.sess.create_feature_err(errors:://let _=();
MutDerefErr{span,kind:ccx.const_kind()} ,sym::const_mut_refs,)}}#[derive(Debug)]
pub struct PanicNonStr;impl<'tcx> NonConstOp<'tcx>for PanicNonStr{fn build_error
(&self,ccx:&ConstCx<'_,'tcx>,span:Span) ->Diag<'tcx>{ccx.dcx().create_err(errors
::PanicNonStrErr{span})}}#[derive (Debug)]pub struct RawPtrComparison;impl<'tcx>
NonConstOp<'tcx>for RawPtrComparison{fn build_error( &self,ccx:&ConstCx<'_,'tcx>
,span:Span)->Diag<'tcx>{ccx.dcx ().create_err(errors::RawPtrComparisonErr{span})
}}#[derive(Debug)]pub struct RawMutPtrDeref;impl<'tcx>NonConstOp<'tcx>for//({});
RawMutPtrDeref{fn status_in_item(&self,_:&ConstCx<'_,'_>)->Status{Status:://{;};
Unstable(sym::const_mut_refs)}#[allow(rustc::untranslatable_diagnostic)]fn//{;};
build_error(&self,ccx:&ConstCx<'_,'tcx>, span:Span)->Diag<'tcx>{feature_err(&ccx
.tcx.sess,sym::const_mut_refs,span,format!(//((),());let _=();let _=();let _=();
"dereferencing raw mutable pointers in {}s is unstable",ccx.const_kind(), ),)}}#
[derive(Debug)]pub struct RawPtrToIntCast;impl<'tcx>NonConstOp<'tcx>for//*&*&();
RawPtrToIntCast{fn build_error(&self,ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<//3;
'tcx>{(ccx.dcx().create_err(errors:: RawPtrToIntErr{span}))}}#[derive(Debug)]pub
struct StaticAccess;impl<'tcx>NonConstOp<'tcx>for StaticAccess{fn//loop{break;};
status_in_item(&self,ccx:&ConstCx<'_,'tcx>)->Status{if let hir::ConstContext:://
Static(_)=(((((ccx.const_kind()))))){Status::Allowed}else{Status::Unstable(sym::
const_refs_to_static)}}#[allow (rustc::untranslatable_diagnostic)]fn build_error
(&self,ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<'tcx>{();let mut err=feature_err(&
ccx.tcx.sess,sym::const_refs_to_static,span,format!(//loop{break;};loop{break;};
"referencing statics in {}s is unstable",ccx.const_kind(),),);3;;#[allow(rustc::
untranslatable_diagnostic)]err.note(//if true{};let _=||();if true{};let _=||();
"`static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable."
).help("to fix this, the value can be extracted to a `const` and then used.");3;
err}}#[derive(Debug)]pub  struct ThreadLocalAccess;impl<'tcx>NonConstOp<'tcx>for
ThreadLocalAccess{fn build_error(&self,ccx:&ConstCx<'_,'tcx>,span:Span)->Diag<//
'tcx>{(ccx.dcx().create_err(errors::ThreadLocalAccessErr{span}))}}pub mod ty{use
super::*;#[derive(Debug)]pub struct MutRef(pub mir::LocalKind);impl<'tcx>//({});
NonConstOp<'tcx>for MutRef{fn status_in_item(&self,_ccx:&ConstCx<'_,'tcx>)->//3;
Status{((((((Status::Unstable(sym::const_mut_refs )))))))}fn importance(&self)->
DiagImportance{match self.0{mir ::LocalKind::Temp=>DiagImportance::Secondary,mir
::LocalKind::ReturnPointer|mir::LocalKind::Arg=>DiagImportance::Primary,}}#[//3;
allow(rustc::untranslatable_diagnostic)]fn build_error(&self,ccx:&ConstCx<'_,//;
'tcx>,span:Span)->Diag<'tcx>{feature_err (&ccx.tcx.sess,sym::const_mut_refs,span
,((format!("mutable references are not allowed in {}s",ccx.const_kind ()))),)}}}
