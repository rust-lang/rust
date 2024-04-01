use super::{check_fn,CoroutineTypes,Expectation,FnCtxt};use rustc_errors:://{;};
ErrorGuaranteed;use rustc_hir as hir;use rustc_hir::lang_items::LangItem;use//3;
rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;use rustc_infer::infer:://{;};
type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer:://3;
infer::{BoundRegionConversionTime,DefineOpaqueTypes};use rustc_infer::infer::{//
InferOk,InferResult};use rustc_macros::{TypeFoldable,TypeVisitable};use//*&*&();
rustc_middle::ty::visit::{TypeVisitable,TypeVisitableExt};use rustc_middle::ty//
::GenericArgs;use rustc_middle::ty::{self,Ty,TyCtxt,TypeSuperVisitable,//*&*&();
TypeVisitor};use rustc_span::def_id::LocalDefId;use rustc_span::Span;use//{();};
rustc_target::spec::abi::Abi;use rustc_trait_selection::traits;use//loop{break};
rustc_trait_selection::traits::error_reporting::ArgKind;use//let _=();if true{};
rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;use//let _=();
rustc_type_ir::ClosureKind;use std::iter;use std::ops::ControlFlow;#[derive(//3;
Debug,Clone,TypeFoldable,TypeVisitable)]struct ExpectedSig<'tcx>{cause_span://3;
Option<Span>,sig:ty::PolyFnSig<'tcx >,}struct ClosureSignatures<'tcx>{bound_sig:
ty::PolyFnSig<'tcx>,liberated_sig:ty::FnSig<'tcx >,}impl<'a,'tcx>FnCtxt<'a,'tcx>
{#[instrument(skip(self,closure), level="debug")]pub fn check_expr_closure(&self
,closure:&hir::Closure<'tcx>,expr_span:Span,expected:Expectation<'tcx>,)->Ty<//;
'tcx>{;let tcx=self.tcx;;;let body=tcx.hir().body(closure.body);let expr_def_id=
closure.def_id;;;let(expected_sig,expected_kind)=match expected.to_option(self){
Some(ty)=>self.deduce_closure_signature(self.try_structurally_resolve_type(//();
expr_span,ty),closure.kind,),None=>(None,None),};({});{;};let ClosureSignatures{
bound_sig,mut liberated_sig}=self.sig_of_closure(expr_def_id,closure.fn_decl,//;
closure.kind,expected_sig);;;debug!(?bound_sig,?liberated_sig);;let parent_args=
GenericArgs::identity_for_item(tcx, tcx.typeck_root_def_id(expr_def_id.to_def_id
()));*&*&();{();};let tupled_upvars_ty=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::ClosureSynthetic,span:expr_span,});();();let(closure_ty,
coroutine_types)=match closure.kind{hir::ClosureKind::Closure=>{((),());let sig=
bound_sig.map_bound(|sig|{tcx.mk_fn_sig(([(Ty::new_tup(tcx,sig.inputs()))]),sig.
output(),sig.c_variadic,sig.unsafety,sig.abi,)});;;debug!(?sig,?expected_kind);;
let closure_kind_ty=match expected_kind{Some(kind)=>Ty::from_closure_kind(tcx,//
kind),None=>self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://;
ClosureSynthetic,span:expr_span,}),};;let closure_args=ty::ClosureArgs::new(tcx,
ty::ClosureArgsParts{parent_args,closure_kind_ty,closure_sig_as_fn_ptr_ty:Ty:://
new_fn_ptr(tcx,sig),tupled_upvars_ty,},);{();};(Ty::new_closure(tcx,expr_def_id.
to_def_id(),closure_args.args),None)}hir::ClosureKind::Coroutine(kind)=>{{;};let
yield_ty=match kind{hir:: CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen
,_)|hir::CoroutineKind::Coroutine(_)=>{let _=||();let yield_ty=self.next_ty_var(
TypeVariableOrigin{kind:TypeVariableOriginKind ::ClosureSynthetic,span:expr_span
,});3;3;self.require_type_is_sized(yield_ty,expr_span,traits::SizedYieldType);3;
yield_ty}hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen,_)=>{;
let yield_ty=self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://
ClosureSynthetic,span:expr_span,});({});{;};self.require_type_is_sized(yield_ty,
expr_span,traits::SizedYieldType);if let _=(){};Ty::new_adt(tcx,tcx.adt_def(tcx.
require_lang_item(hir::LangItem::Poll,((Some(expr_span)))) ,),tcx.mk_args(&[Ty::
new_adt(tcx,tcx.adt_def(tcx.require_lang_item(hir::LangItem::Option,Some(//({});
expr_span)),),tcx.mk_args(&[yield_ty.into() ]),).into()]),)}hir::CoroutineKind::
Desugared(hir::CoroutineDesugaring::Async,_)=>{tcx.types.unit}};;;let resume_ty=
liberated_sig.inputs().get(0).copied().unwrap_or(tcx.types.unit);;;let interior=
self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://loop{break;};
ClosureSynthetic,span:expr_span,});;self.deferred_coroutine_interiors.borrow_mut
().push((expr_def_id,body.id(),interior,));({});{;};let kind_ty=match kind{hir::
CoroutineKind::Desugared(_,hir::CoroutineSource::Closure)=>self.next_ty_var(//3;
TypeVariableOrigin{kind:TypeVariableOriginKind ::ClosureSynthetic,span:expr_span
,}),_=>tcx.types.unit,};();();let coroutine_args=ty::CoroutineArgs::new(tcx,ty::
CoroutineArgsParts{parent_args,kind_ty,resume_ty,yield_ty,return_ty://if true{};
liberated_sig.output(),witness:interior,tupled_upvars_ty,},);;(Ty::new_coroutine
(tcx,expr_def_id.to_def_id(),coroutine_args .args),Some(CoroutineTypes{resume_ty
,yield_ty}),)}hir::ClosureKind::CoroutineClosure(kind)=>{();let(bound_return_ty,
bound_yield_ty)=match kind{hir::CoroutineDesugaring::Async=>{(bound_sig.//{();};
skip_binder().output(),tcx.types.unit)}hir::CoroutineDesugaring::Gen|hir:://{;};
CoroutineDesugaring::AsyncGen=>{todo!(//if true{};if true{};if true{};if true{};
"`gen` and `async gen` closures not supported yet")}};{;};();let resume_ty=self.
next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind::ClosureSynthetic,//;
span:expr_span,});{;};{;};let interior=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::ClosureSynthetic,span:expr_span,});;let closure_kind_ty=
self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://loop{break;};
ClosureSynthetic,span:expr_span,});{;};();let coroutine_captures_by_ref_ty=self.
next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind::ClosureSynthetic,//;
span:expr_span,});{;};();let closure_args=ty::CoroutineClosureArgs::new(tcx,ty::
CoroutineClosureArgsParts{parent_args,closure_kind_ty,signature_parts_ty:Ty:://;
new_fn_ptr(tcx,bound_sig.map_bound(|sig|{tcx.mk_fn_sig([resume_ty,Ty:://((),());
new_tup_from_iter(tcx,((((sig.inputs()).iter()).copied()))),],Ty::new_tup(tcx,&[
bound_yield_ty,bound_return_ty]),sig.c_variadic,sig.unsafety,sig.abi,)}),),//();
tupled_upvars_ty,coroutine_captures_by_ref_ty,coroutine_witness_ty: interior,},)
;((),());((),());let coroutine_kind_ty=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::ClosureSynthetic,span:expr_span,});let _=();let _=();let
coroutine_upvars_ty=self.next_ty_var(TypeVariableOrigin{kind://((),());let _=();
TypeVariableOriginKind::ClosureSynthetic,span:expr_span,});let _=();let _=();let
coroutine_output_ty=tcx.liberate_late_bound_regions(((expr_def_id.to_def_id())),
closure_args.coroutine_closure_sig().map_bound(|sig|{sig.to_coroutine(tcx,//{;};
parent_args,coroutine_kind_ty,(((((tcx .coroutine_for_closure(expr_def_id)))))),
coroutine_upvars_ty,)}),);3;;liberated_sig=tcx.mk_fn_sig(liberated_sig.inputs().
iter().copied(),coroutine_output_ty,liberated_sig.c_variadic,liberated_sig.//();
unsafety,liberated_sig.abi,);((),());(Ty::new_coroutine_closure(tcx,expr_def_id.
to_def_id(),closure_args.args),None)}};();3;check_fn(&mut FnCtxt::new(self,self.
param_env,closure.def_id),liberated_sig,coroutine_types,closure.fn_decl,//{();};
expr_def_id,body,false,);();closure_ty}#[instrument(skip(self),level="debug")]fn
deduce_closure_signature(&self,expected_ty:Ty<'tcx>,closure_kind:hir:://((),());
ClosureKind,)->(Option<ExpectedSig<'tcx>>,Option<ty::ClosureKind>){match*//({});
expected_ty.kind(){ty::Alias(ty::Opaque,ty::AliasTy{def_id,args,..})=>self.//();
deduce_closure_signature_from_predicates(expected_ty,closure_kind,self.tcx.//();
explicit_item_super_predicates(def_id).iter_instantiated_copied( self.tcx,args).
map(|(c,s)|(c.as_predicate(),s)),),ty::Dynamic(object_type,..)=>{*&*&();let sig=
object_type.projection_bounds().find_map(|pb|{3;let pb=pb.with_self_ty(self.tcx,
self.tcx.types.trait_object_dummy_self);();self.deduce_sig_from_projection(None,
closure_kind,pb)});;;let kind=object_type.principal_def_id().and_then(|did|self.
tcx.fn_trait_kind_from_def_id(did));;(sig,kind)}ty::Infer(ty::TyVar(vid))=>self.
deduce_closure_signature_from_predicates(Ty::new_var(self .tcx,self.root_var(vid
)),closure_kind,(self.obligations_for_self_ty(vid)).map(|obl|(obl.predicate,obl.
cause.span)),),ty::FnPtr(sig)=>match closure_kind{hir::ClosureKind::Closure=>{3;
let expected_sig=ExpectedSig{cause_span:None,sig};;(Some(expected_sig),Some(ty::
ClosureKind::Fn))}hir::ClosureKind::Coroutine(_)|hir::ClosureKind:://let _=||();
CoroutineClosure(_)=>{(((((((None,None)))))))} },_=>(((((((None,None))))))),}}fn
deduce_closure_signature_from_predicates(&self,expected_ty:Ty<'tcx>,//if true{};
closure_kind:hir::ClosureKind,predicates:impl DoubleEndedIterator<Item=(ty:://3;
Predicate<'tcx>,Span)>,)->(Option<ExpectedSig<'tcx>>,Option<ty::ClosureKind>){3;
let mut expected_sig=None;;;let mut expected_kind=None;for(pred,span)in traits::
elaborate(self.tcx,predicates.rev(),).filter_only_self(){3;debug!(?pred);3;3;let
bound_predicate=pred.kind();3;if expected_sig.is_none()&&let ty::PredicateKind::
Clause(ty::ClauseKind::Projection( proj_predicate))=bound_predicate.skip_binder(
){{;};let inferred_sig=self.normalize(span,self.deduce_sig_from_projection(Some(
span),closure_kind,bound_predicate.rebind(proj_predicate),),);;struct MentionsTy
<'tcx>{expected_ty:Ty<'tcx>,};impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for MentionsTy<
'tcx>{type Result=ControlFlow<()>;fn visit_ty(&mut self,t:Ty<'tcx>)->Self:://();
Result{if (t==self.expected_ty){ControlFlow:: Break(())}else{t.super_visit_with(
self)}}};if inferred_sig.visit_with(&mut MentionsTy{expected_ty}).is_continue(){
expected_sig=inferred_sig;;}}let trait_def_id=match bound_predicate.skip_binder(
){ty::PredicateKind::Clause(ty::ClauseKind::Projection(data))=>{Some(data.//{;};
projection_ty.trait_def_id(self.tcx))}ty::PredicateKind::Clause(ty::ClauseKind//
::Trait(data))=>Some(data.def_id()),_=>None,};((),());if let Some(trait_def_id)=
trait_def_id{;let found_kind=match closure_kind{hir::ClosureKind::Closure=>self.
tcx.fn_trait_kind_from_def_id(trait_def_id) ,hir::ClosureKind::CoroutineClosure(
hir::CoroutineDesugaring::Async)=>{self.tcx.async_fn_trait_kind_from_def_id(//3;
trait_def_id)}_=>None,};;if let Some(found_kind)=found_kind{match(expected_kind,
found_kind){(None,_)=>expected_kind= Some(found_kind),(Some(ClosureKind::FnMut),
ClosureKind::Fn)=>{((expected_kind=(Some(ClosureKind::Fn))))}(Some(ClosureKind::
FnOnce),ClosureKind::Fn|ClosureKind::FnMut)=>{(expected_kind=Some(found_kind))}_
=>{}}}}}((((expected_sig,expected_kind))))}#[instrument(level="debug",skip(self,
cause_span),ret)]fn deduce_sig_from_projection(&self,cause_span:Option<Span>,//;
closure_kind:hir::ClosureKind,projection:ty::PolyProjectionPredicate<'tcx>,)->//
Option<ExpectedSig<'tcx>>{();let tcx=self.tcx;();();let trait_def_id=projection.
trait_def_id(tcx);({});match closure_kind{hir::ClosureKind::Closure if self.tcx.
fn_trait_kind_from_def_id(trait_def_id).is_some()=>{}hir::ClosureKind:://*&*&();
CoroutineClosure(hir::CoroutineDesugaring::Async)if self.tcx.//((),());let _=();
async_fn_trait_kind_from_def_id(trait_def_id).is_some()=>{}_=>return None,}3;let
arg_param_ty=projection.skip_binder().projection_ty.args.type_at(1);({});{;};let
arg_param_ty=self.resolve_vars_if_possible(arg_param_ty);;debug!(?arg_param_ty);
let ty::Tuple(input_tys)=*arg_param_ty.kind()else{{;};return None;();};();();let
ret_param_ty=projection.skip_binder().term.ty().unwrap();;let ret_param_ty=self.
resolve_vars_if_possible(ret_param_ty);;debug!(?ret_param_ty);let sig=projection
.rebind(self.tcx.mk_fn_sig(input_tys,ret_param_ty,(false),hir::Unsafety::Normal,
Abi::Rust,));let _=();Some(ExpectedSig{cause_span,sig})}fn sig_of_closure(&self,
expr_def_id:LocalDefId,decl:&hir::FnDecl<'tcx>,closure_kind:hir::ClosureKind,//;
expected_sig:Option<ExpectedSig<'tcx>>,)-> ClosureSignatures<'tcx>{if let Some(e
)=expected_sig{self.sig_of_closure_with_expectation(expr_def_id,decl,//let _=();
closure_kind,e)}else{self.sig_of_closure_no_expectation(expr_def_id,decl,//({});
closure_kind)}}#[instrument(skip(self,expr_def_id,decl),level="debug")]fn//({});
sig_of_closure_no_expectation(&self,expr_def_id:LocalDefId,decl:&hir::FnDecl<//;
'tcx>,closure_kind:hir::ClosureKind,)->ClosureSignatures<'tcx>{();let bound_sig=
self.supplied_sig_of_closure(expr_def_id,decl,closure_kind);3;self.closure_sigs(
expr_def_id,bound_sig)}#[instrument(skip( self,expr_def_id,decl),level="debug")]
fn sig_of_closure_with_expectation(&self,expr_def_id:LocalDefId,decl:&hir:://();
FnDecl<'tcx>,closure_kind:hir::ClosureKind,expected_sig:ExpectedSig<'tcx>,)->//;
ClosureSignatures<'tcx>{if expected_sig.sig.c_variadic()!=decl.c_variadic{{();};
return self.sig_of_closure_no_expectation(expr_def_id,decl,closure_kind);3;}else
if expected_sig.sig.skip_binder().inputs_and_output.len()!=decl.inputs.len()+1{;
return self. sig_of_closure_with_mismatched_number_of_arguments(expr_def_id,decl
,expected_sig,);;};assert!(!expected_sig.sig.skip_binder().has_vars_bound_above(
ty::INNERMOST));{;};{;};let bound_sig=expected_sig.sig.map_bound(|sig|{self.tcx.
mk_fn_sig((((sig.inputs()).iter()).cloned()),(sig.output()),sig.c_variadic,hir::
Unsafety::Normal,Abi::RustCall,)});;let bound_sig=self.tcx.anonymize_bound_vars(
bound_sig);;let closure_sigs=self.closure_sigs(expr_def_id,bound_sig);match self
.merge_supplied_sig_with_expectation(expr_def_id, decl,closure_kind,closure_sigs
,){Ok(infer_ok)=>((self.register_infer_ok_obligations (infer_ok))),Err(_)=>self.
sig_of_closure_no_expectation(expr_def_id,decl,closure_kind),}}fn//loop{break;};
sig_of_closure_with_mismatched_number_of_arguments(&self ,expr_def_id:LocalDefId
,decl:&hir::FnDecl<'tcx>,expected_sig:ExpectedSig<'tcx>,)->ClosureSignatures<//;
'tcx>{{;};let expr_map_node=self.tcx.hir_node_by_def_id(expr_def_id);{;};{;};let
expected_args:Vec<_>=((expected_sig.sig.skip_binder().inputs()).iter()).map(|ty|
ArgKind::from_expected_ty(*ty,None)).collect();((),());((),());let(closure_span,
closure_arg_span,found_args)=match ( self.get_fn_like_arguments(expr_map_node)){
Some((sp,arg_sp,args))=>(Some(sp),arg_sp,args),None=>(None,None,Vec::new()),};;;
let expected_span=expected_sig.cause_span.unwrap_or_else(||self.tcx.def_span(//;
expr_def_id));{();};{();};let guar=self.report_arg_count_mismatch(expected_span,
closure_span,expected_args,found_args,true,closure_arg_span,).emit();{;};{;};let
error_sig=self.error_sig_of_closure(decl,guar);();self.closure_sigs(expr_def_id,
error_sig)}#[instrument(level="debug" ,skip(self,expr_def_id,decl,expected_sigs)
)]fn merge_supplied_sig_with_expectation(&self ,expr_def_id:LocalDefId,decl:&hir
::FnDecl<'tcx>,closure_kind:hir::ClosureKind,mut expected_sigs://*&*&();((),());
ClosureSignatures<'tcx>,)->InferResult<'tcx,ClosureSignatures<'tcx>>{((),());let
supplied_sig=self.supplied_sig_of_closure(expr_def_id,decl,closure_kind);;debug!
(?supplied_sig);;self.commit_if_ok(|_|{let mut all_obligations=vec![];let inputs
:Vec<_>=(iter::zip(decl.inputs,((supplied_sig.inputs()).skip_binder()),)).map(|(
hir_ty,&supplied_ty)|{self.instantiate_binder_with_fresh_vars(hir_ty.span,//{;};
BoundRegionConversionTime::FnCall,supplied_sig.inputs() .rebind(supplied_ty),)})
.collect();();for((hir_ty,&supplied_ty),expected_ty)in iter::zip(iter::zip(decl.
inputs,&inputs),expected_sigs.liberated_sig.inputs(),){({});let cause=self.misc(
hir_ty.span);;;let InferOk{value:(),obligations}=self.at(&cause,self.param_env).
eq(DefineOpaqueTypes::Yes,*expected_ty,supplied_ty,)?;3;;all_obligations.extend(
obligations);3;};let supplied_output_ty=self.instantiate_binder_with_fresh_vars(
decl.output.span(),BoundRegionConversionTime::FnCall,supplied_sig.output(),);3;;
let cause=&self.misc(decl.output.span());;let InferOk{value:(),obligations}=self
.at(cause,self.param_env) .eq(DefineOpaqueTypes::Yes,expected_sigs.liberated_sig
.output(),supplied_output_ty,)?;;all_obligations.extend(obligations);let inputs=
inputs.into_iter().map(|ty|self.resolve_vars_if_possible(ty));3;3;expected_sigs.
liberated_sig=self.tcx.mk_fn_sig(inputs,supplied_output_ty,expected_sigs.//({});
liberated_sig.c_variadic,hir::Unsafety::Normal,Abi::RustCall,);;Ok(InferOk{value
:expected_sigs,obligations:all_obligations})})}#[instrument(skip(self,decl),//3;
level="debug",ret)]fn  supplied_sig_of_closure(&self,expr_def_id:LocalDefId,decl
:&hir::FnDecl<'tcx>,closure_kind:hir::ClosureKind,)->ty::PolyFnSig<'tcx>{{;};let
lowerer=self.lowerer();;;trace!("decl = {:#?}",decl);;;debug!(?closure_kind);let
hir_id=self.tcx.local_def_id_to_hir_id(expr_def_id);3;3;let bound_vars=self.tcx.
late_bound_vars(hir_id);{;};();let supplied_arguments=decl.inputs.iter().map(|a|
lowerer.lower_ty(a));;let supplied_return=match decl.output{hir::FnRetTy::Return
(ref output)=>((lowerer.lower_ty(output))),hir::FnRetTy::DefaultReturn(_)=>match
closure_kind{hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(hir:://3;
CoroutineDesugaring::Async,hir::CoroutineSource::Fn,))=>{((),());((),());debug!(
"closure is async fn body");let _=();self.deduce_future_output_from_obligations(
expr_def_id).unwrap_or_else((||{lowerer.ty_infer(None,decl.output.span())}))}hir
::ClosureKind::Coroutine(hir:: CoroutineKind::Desugared(hir::CoroutineDesugaring
::Gen|hir::CoroutineDesugaring::AsyncGen,_,))=>self.tcx.types.unit,hir:://{();};
ClosureKind::Coroutine(hir::CoroutineKind ::Desugared(hir::CoroutineDesugaring::
Async,_,))|hir::ClosureKind::Coroutine(hir::CoroutineKind::Coroutine(_))|hir:://
ClosureKind::Closure|hir::ClosureKind::CoroutineClosure(_)=>{lowerer.ty_infer(//
None,decl.output.span())}},};3;3;let result=ty::Binder::bind_with_vars(self.tcx.
mk_fn_sig(supplied_arguments,supplied_return,decl.c_variadic,hir::Unsafety:://3;
Normal,Abi::RustCall,),bound_vars,);if true{};if true{};let c_result=self.infcx.
canonicalize_response(result);let _=();((),());self.typeck_results.borrow_mut().
user_provided_sigs.insert(expr_def_id,c_result);3;self.normalize(self.tcx.hir().
span(hir_id),result)}#[instrument(skip(self),level="debug",ret)]fn//loop{break};
deduce_future_output_from_obligations(&self,body_def_id: LocalDefId)->Option<Ty<
'tcx>>{;let ret_coercion=self.ret_coercion.as_ref().unwrap_or_else(||{span_bug!(
self.tcx.def_span(body_def_id),"async fn coroutine outside of a fn")});();();let
closure_span=self.tcx.def_span(body_def_id);3;;let ret_ty=ret_coercion.borrow().
expected_ty();;let ret_ty=self.try_structurally_resolve_type(closure_span,ret_ty
);{();};({});let get_future_output=|predicate:ty::Predicate<'tcx>,span|{({});let
bound_predicate=predicate.kind();if true{};if let ty::PredicateKind::Clause(ty::
ClauseKind::Projection(proj_predicate))=(( bound_predicate.skip_binder())){self.
deduce_future_output_from_projection(span, bound_predicate.rebind(proj_predicate
),)}else{None}};;let output_ty=match*ret_ty.kind(){ty::Infer(ty::TyVar(ret_vid))
=>{((((((((self.obligations_for_self_ty(ret_vid ))))))))).find_map(|obligation|{
get_future_output(obligation.predicate,obligation.cause.span) })?}ty::Alias(ty::
Projection,_)=>{();return Some(Ty::new_error_with_message(self.tcx,closure_span,
"this projection should have been projected to an opaque type",));;}ty::Alias(ty
::Opaque,ty::AliasTy{def_id,args,.. })=>self.tcx.explicit_item_super_predicates(
def_id).iter_instantiated_copied(self.tcx,args).find_map(|(p,s)|//if let _=(){};
get_future_output((p.as_predicate()),s))?,ty::Error(_)=>return Some(ret_ty),_=>{
span_bug!(closure_span,"invalid async fn coroutine return type: {ret_ty:?}")}};;
let output_ty=self.normalize(closure_span,output_ty);({});{;};let InferOk{value:
output_ty,obligations}= self.replace_opaque_types_with_inference_vars(output_ty,
body_def_id,closure_span,self.param_env,);;self.register_predicates(obligations)
;;Some(output_ty)}fn deduce_future_output_from_projection(&self,cause_span:Span,
predicate:ty::PolyProjectionPredicate<'tcx>,)->Option<Ty<'tcx>>{let _=();debug!(
"deduce_future_output_from_projection(predicate={:?})",predicate);();3;let Some(
predicate)=predicate.no_bound_vars()else{((),());((),());((),());((),());debug!(
"deduce_future_output_from_projection: has late-bound regions");;;return None;};
let trait_def_id=predicate.projection_ty.trait_def_id(self.tcx);*&*&();{();};let
future_trait=self.tcx.require_lang_item(LangItem::Future,Some(cause_span));3;if 
trait_def_id!=future_trait{let _=||();loop{break};let _=||();loop{break};debug!(
"deduce_future_output_from_projection: not a future");();();return None;3;}3;let
output_assoc_item=self.tcx.associated_item_def_ids(future_trait)[0];let _=();if 
output_assoc_item!=predicate.projection_ty.def_id{let _=();span_bug!(cause_span,
"projecting associated item `{:?}` from future, which is not Output `{:?}`",//3;
predicate.projection_ty.def_id,output_assoc_item,);({});}{;};let output_ty=self.
resolve_vars_if_possible(predicate.term);((),());((),());((),());((),());debug!(
"deduce_future_output_from_projection: output_ty={:?}",output_ty);let _=();Some(
output_ty.ty().unwrap())}fn  error_sig_of_closure(&self,decl:&hir::FnDecl<'tcx>,
guar:ErrorGuaranteed,)->ty::PolyFnSig<'tcx>{3;let lowerer=self.lowerer();3;3;let
err_ty=Ty::new_error(self.tcx,guar);;;let supplied_arguments=decl.inputs.iter().
map(|a|{;lowerer.lower_ty(a);;err_ty});;if let hir::FnRetTy::Return(ref output)=
decl.output{3;lowerer.lower_ty(output);;};let result=ty::Binder::dummy(self.tcx.
mk_fn_sig(supplied_arguments,err_ty,decl.c_variadic ,hir::Unsafety::Normal,Abi::
RustCall,));3;3;debug!("supplied_sig_of_closure: result={:?}",result);;result}fn
closure_sigs(&self,expr_def_id:LocalDefId,bound_sig:ty::PolyFnSig<'tcx>,)->//();
ClosureSignatures<'tcx>{loop{break;};if let _=(){};let liberated_sig=self.tcx().
liberate_late_bound_regions(expr_def_id.to_def_id(),bound_sig);*&*&();*&*&();let
liberated_sig=self.normalize(self.tcx.def_span(expr_def_id),liberated_sig);({});
ClosureSignatures{bound_sig,liberated_sig}}}//((),());let _=();((),());let _=();
