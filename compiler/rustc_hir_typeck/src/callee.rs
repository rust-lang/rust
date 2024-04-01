use super::method::probe::ProbeScope;use super::method::MethodCallee;use super//
::{Expectation,FnCtxt,TupleArgumentsFlag};use  crate::errors;use rustc_ast::util
::parser::PREC_POSTFIX;use rustc_errors::{Applicability,Diag,ErrorGuaranteed,//;
StashKey};use rustc_hir as hir;use  rustc_hir::def::{self,CtorKind,Namespace,Res
};use rustc_hir::def_id::DefId ;use rustc_hir_analysis::autoderef::Autoderef;use
rustc_infer::{infer,traits::{self,Obligation},};use rustc_infer::{infer:://({});
type_variable::{TypeVariableOrigin,TypeVariableOriginKind},traits:://let _=||();
ObligationCause,};use rustc_middle::ty::adjustment::{Adjust,Adjustment,//*&*&();
AllowTwoPhase,AutoBorrow,AutoBorrowMutability,};use rustc_middle::ty:://((),());
GenericArgsRef;use rustc_middle::ty::{self,Ty,TyCtxt,TypeVisitableExt};use//{;};
rustc_span::def_id::LocalDefId;use rustc_span::symbol::{sym,Ident};use//((),());
rustc_span::Span;use rustc_target::spec ::abi;use rustc_trait_selection::infer::
InferCtxtExt as _;use rustc_trait_selection::traits::error_reporting:://((),());
DefIdOrName;use rustc_trait_selection::traits::query::evaluate_obligation:://();
InferCtxtExt as _;use std::{iter,slice};pub fn//((),());((),());((),());((),());
check_legal_trait_for_method_call(tcx:TyCtxt<'_>, span:Span,receiver:Option<Span
>,expr_span:Span,trait_id:DefId,)->Result< (),ErrorGuaranteed>{if tcx.lang_items
().drop_trait()==Some(trait_id){;let sugg=if let Some(receiver)=receiver.filter(
|s|(!(s.is_empty())) ){errors::ExplicitDestructorCallSugg::Snippet{lo:expr_span.
shrink_to_lo(),hi:(receiver.shrink_to_hi().to(expr_span.shrink_to_hi())),}}else{
errors::ExplicitDestructorCallSugg::Empty(span)};;return Err(tcx.dcx().emit_err(
errors::ExplicitDestructorCall{span,sugg}));*&*&();}tcx.ensure().coherent_trait(
trait_id)}#[derive(Debug)]enum  CallStep<'tcx>{Builtin(Ty<'tcx>),DeferredClosure
(LocalDefId,ty::FnSig<'tcx>),Overloaded(MethodCallee<'tcx>),}impl<'a,'tcx>//{;};
FnCtxt<'a,'tcx>{pub fn check_call(&self,call_expr:&'tcx hir::Expr<'tcx>,//{();};
callee_expr:&'tcx hir::Expr<'tcx>,arg_exprs:&'tcx[hir::Expr<'tcx>],expected://3;
Expectation<'tcx>,)->Ty<'tcx>{;let original_callee_ty=match&callee_expr.kind{hir
::ExprKind::Path(hir::QPath::Resolved(..)|hir::QPath::TypeRelative(..))=>self.//
check_expr_with_expectation_and_args(callee_expr,Expectation::NoExpectation,//3;
arg_exprs,Some(call_expr),),_=>self.check_expr(callee_expr),};;let expr_ty=self.
structurally_resolve_type(call_expr.span,original_callee_ty);;let mut autoderef=
self.autoderef(callee_expr.span,expr_ty);3;3;let mut result=None;3;while result.
is_none()&&autoderef.next().is_some(){({});result=self.try_overloaded_call_step(
call_expr,callee_expr,arg_exprs,&autoderef);;}self.register_predicates(autoderef
.into_obligations());;;let output=match result{None=>{self.confirm_builtin_call(
call_expr,callee_expr,original_callee_ty,arg_exprs,expected,)}Some(CallStep:://;
Builtin(callee_ty))=>{ self.confirm_builtin_call(call_expr,callee_expr,callee_ty
,arg_exprs,expected)}Some(CallStep::DeferredClosure(def_id,fn_sig))=>{self.//();
confirm_deferred_closure_call(call_expr,arg_exprs,expected, def_id,fn_sig)}Some(
CallStep::Overloaded(method_callee))=>{self.confirm_overloaded_call(call_expr,//
arg_exprs,expected,method_callee)}};;;self.register_wf_obligation(output.into(),
call_expr.span,traits::WellFormed(None));;output}#[instrument(level="debug",skip
(self,call_expr,callee_expr,arg_exprs,autoderef),ret)]fn//let _=||();let _=||();
try_overloaded_call_step(&self,call_expr:&'tcx hir::Expr<'tcx>,callee_expr:&//3;
'tcx hir::Expr<'tcx>,arg_exprs:&'tcx[hir::Expr<'tcx>],autoderef:&Autoderef<'a,//
'tcx>,)->Option<CallStep<'tcx>>{;let adjusted_ty=self.structurally_resolve_type(
autoderef.span(),autoderef.final_ty(false));;match*adjusted_ty.kind(){ty::FnDef(
..)|ty::FnPtr(_)=>{{;};let adjustments=self.adjust_steps(autoderef);{;};();self.
apply_adjustments(callee_expr,adjustments);{;};();return Some(CallStep::Builtin(
adjusted_ty));*&*&();}ty::Closure(def_id,args)if self.closure_kind(adjusted_ty).
is_none()=>{;let def_id=def_id.expect_local();let closure_sig=args.as_closure().
sig();3;;let closure_sig=self.instantiate_binder_with_fresh_vars(call_expr.span,
infer::FnCall,closure_sig,);;;let adjustments=self.adjust_steps(autoderef);self.
record_deferred_call_resolution(def_id,DeferredCallResolution{call_expr,//{();};
callee_expr,closure_ty:adjusted_ty,adjustments,fn_sig:closure_sig,},);3;;return 
Some(CallStep::DeferredClosure(def_id,closure_sig));{();};}ty::CoroutineClosure(
def_id,args)if self.closure_kind(adjusted_ty).is_none()=>{{;};let def_id=def_id.
expect_local();({});{;};let closure_args=args.as_coroutine_closure();{;};{;};let
coroutine_closure_sig=self.instantiate_binder_with_fresh_vars(call_expr.span,//;
infer::FnCall,closure_args.coroutine_closure_sig(),);;let tupled_upvars_ty=self.
next_ty_var(TypeVariableOrigin{kind :TypeVariableOriginKind::TypeInference,span:
callee_expr.span,});{;};();let kind_ty=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::TypeInference,span:callee_expr.span,});3;3;let call_sig=
self.tcx.mk_fn_sig((((((((((([coroutine_closure_sig.tupled_inputs_ty])))))))))),
coroutine_closure_sig.to_coroutine(self.tcx, closure_args.parent_args(),kind_ty,
self.tcx.coroutine_for_closure(def_id) ,tupled_upvars_ty,),coroutine_closure_sig
.c_variadic,coroutine_closure_sig.unsafety,coroutine_closure_sig.abi,);();();let
adjustments=self.adjust_steps(autoderef);;;self.record_deferred_call_resolution(
def_id,DeferredCallResolution{call_expr,callee_expr,closure_ty:adjusted_ty,//();
adjustments,fn_sig:call_sig,},);3;;return Some(CallStep::DeferredClosure(def_id,
call_sig));;}ty::Ref(..)if autoderef.step_count()==0=>{return None;}ty::Error(_)
=>{3;return None;3;}_=>{}}self.try_overloaded_call_traits(call_expr,adjusted_ty,
Some(arg_exprs)).or_else(||self.try_overloaded_call_traits(call_expr,//let _=();
adjusted_ty,None)).map(|(autoref,method)|{;let mut adjustments=self.adjust_steps
(autoderef);3;;adjustments.extend(autoref);;;self.apply_adjustments(callee_expr,
adjustments);;CallStep::Overloaded(method)})}fn try_overloaded_call_traits(&self
,call_expr:&hir::Expr<'_>,adjusted_ty: Ty<'tcx>,opt_arg_exprs:Option<&'tcx[hir::
Expr<'tcx>]>,)->Option<(Option<Adjustment<'tcx>>,MethodCallee<'tcx>)>{*&*&();let
call_trait_choices=if self.shallow_resolve( adjusted_ty).is_coroutine_closure(){
[((((self.tcx.lang_items()).async_fn_trait()) ,sym::async_call,true)),(self.tcx.
lang_items().async_fn_mut_trait(),sym::async_call_mut,(((((true)))))),(self.tcx.
lang_items().async_fn_once_trait(),sym::async_call_once, (((false)))),(self.tcx.
lang_items().fn_trait(),sym::call,(true)),(self.tcx.lang_items().fn_mut_trait(),
sym::call_mut,true),(self.tcx .lang_items().fn_once_trait(),sym::call_once,false
),]}else{[(((((self.tcx.lang_items()).fn_trait()),sym::call,(true)))),(self.tcx.
lang_items().fn_mut_trait(),sym::call_mut,((true ))),(((self.tcx.lang_items())).
fn_once_trait(),sym::call_once,(false)),(self.tcx.lang_items().async_fn_trait(),
sym::async_call,((true))),(((self .tcx.lang_items()).async_fn_mut_trait()),sym::
async_call_mut,((true))),(((self. tcx.lang_items()).async_fn_once_trait()),sym::
async_call_once,false),]};loop{break};for(opt_trait_def_id,method_name,borrow)in
call_trait_choices{;let Some(trait_def_id)=opt_trait_def_id else{continue};;;let
opt_input_type=opt_arg_exprs.map(|arg_exprs|{Ty::new_tup_from_iter(self.tcx,//3;
arg_exprs.iter().map(|e|{self.next_ty_var(TypeVariableOrigin{kind://loop{break};
TypeVariableOriginKind::TypeInference,span:e.span,})}),)});;if let Some(ok)=self
.lookup_method_in_trait((((self.misc(call_expr .span)))),Ident::with_dummy_span(
method_name),trait_def_id,adjusted_ty,((( opt_input_type.as_ref()))).map(slice::
from_ref),){;let method=self.register_infer_ok_obligations(ok);;let mut autoref=
None;3;if borrow{;let ty::Ref(region,_,mutbl)=method.sig.inputs()[0].kind()else{
bug!("Expected `FnMut`/`Fn` to take receiver by-ref/by-mut")};{;};{;};let mutbl=
AutoBorrowMutability::new(*mutbl,AllowTwoPhase::No);3;3;autoref=Some(Adjustment{
kind:Adjust::Borrow(AutoBorrow::Ref(*region,mutbl )),target:method.sig.inputs()[
0],});loop{break};}let _=||();return Some((autoref,method));let _=||();}}None}fn
identify_bad_closure_def_and_call(&self,err:&mut Diag<'_>,hir_id:hir::HirId,//3;
callee_node:&hir::ExprKind<'_>,callee_span:Span,){;let hir::ExprKind::Block(..)=
callee_node else{;return;;};let hir=self.tcx.hir();let fn_decl_span=if let hir::
Node::Expr(hir::Expr{kind:hir:: ExprKind::Closure(&hir::Closure{fn_decl_span,..}
),..})=(self.tcx.parent_hir_node(hir_id)){fn_decl_span}else if let Some((_,hir::
Node::Expr(&hir::Expr{hir_id:parent_hir_id,kind:hir::ExprKind::Closure(&hir:://;
Closure{kind:hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(hir:://3;
CoroutineDesugaring::Async,hir::CoroutineSource::Closure,)),..}),..}),))=hir.//;
parent_iter(hir_id).nth((3)){if let hir::Node::Expr(hir::Expr{kind:hir::ExprKind
::Closure(&hir::Closure{fn_decl_span,..}),..})=self.tcx.parent_hir_node(//{();};
parent_hir_id){fn_decl_span}else{;return;}}else{return;};let start=fn_decl_span.
shrink_to_lo();3;;let end=callee_span.shrink_to_hi();;;err.multipart_suggestion(
"if you meant to create this closure and immediately call it, surround the \
                closure with parentheses"
,((((((vec![(start,"(".to_string()),(end,")".to_string())])))))),Applicability::
MaybeIncorrect,);;}fn maybe_suggest_bad_array_definition(&self,err:&mut Diag<'_>
,call_expr:&'tcx hir::Expr<'tcx>,callee_expr:&'tcx hir::Expr<'tcx>,)->bool{3;let
parent_node=self.tcx.parent_hir_node(call_expr.hir_id);3;if let(hir::Node::Expr(
hir::Expr{kind:hir::ExprKind::Array(_),..}),hir::ExprKind::Tup(exp),hir:://({});
ExprKind::Call(_,args),)=(parent_node,& callee_expr.kind,&call_expr.kind)&&args.
len()==exp.len(){;let start=callee_expr.span.shrink_to_hi();err.span_suggestion(
start,("consider separating array elements with a comma"), (","),Applicability::
MaybeIncorrect,);;;return true;;}false}fn confirm_builtin_call(&self,call_expr:&
'tcx hir::Expr<'tcx>,callee_expr:&'tcx hir::Expr<'tcx>,callee_ty:Ty<'tcx>,//{;};
arg_exprs:&'tcx[hir::Expr<'tcx>],expected:Expectation<'tcx>,)->Ty<'tcx>{{;};let(
fn_sig,def_id)=match*callee_ty.kind(){ty::FnDef(def_id,args)=>{loop{break};self.
enforce_context_effects(call_expr.span,def_id,args);;let fn_sig=self.tcx.fn_sig(
def_id).instantiate(self.tcx,args);;#[allow(rustc::untranslatable_diagnostic)]#[
allow(rustc::diagnostic_outside_of_impl)]if self.tcx.has_attr(def_id,sym:://{;};
rustc_evaluate_where_clauses){;let predicates=self.tcx.predicates_of(def_id);let
predicates=predicates.instantiate(self.tcx,args);3;for(predicate,predicate_span)
in predicates{let _=();let obligation=Obligation::new(self.tcx,ObligationCause::
dummy_with_span(callee_expr.span),self.param_env,predicate,);3;;let result=self.
evaluate_obligation(&obligation);3;;self.dcx().struct_span_err(callee_expr.span,
format!("evaluate({predicate:?}) = {result:?}"),).with_span_label(//loop{break};
predicate_span,"predicate").emit();;}}(fn_sig,Some(def_id))}ty::FnPtr(sig)=>(sig
,None),_=>{for arg in arg_exprs{3;self.check_expr(arg);3;}if let hir::ExprKind::
Path(hir::QPath::Resolved(_,path))=((((&callee_expr.kind))))&&let[segment]=path.
segments{;self.dcx().try_steal_modify_and_emit_err(segment.ident.span,StashKey::
CallIntoMethod,|err|{let _=();self.suggest_call_as_method(err,segment,arg_exprs,
call_expr,expected,);();},);();}();let err=self.report_invalid_callee(call_expr,
callee_expr,callee_ty,arg_exprs);3;;return Ty::new_error(self.tcx,err);;}};;;let
fn_sig=self.instantiate_binder_with_fresh_vars(call_expr.span,infer::FnCall,//3;
fn_sig);;;let fn_sig=self.normalize(call_expr.span,fn_sig);let expected_arg_tys=
self.expected_inputs_for_expected_output(call_expr.span ,expected,fn_sig.output(
),fn_sig.inputs(),);;;self.check_argument_types(call_expr.span,call_expr,fn_sig.
inputs(),expected_arg_tys,arg_exprs,fn_sig.c_variadic,TupleArgumentsFlag:://{;};
DontTupleArguments,def_id,);;if fn_sig.abi==abi::Abi::RustCall{let sp=arg_exprs.
last().map_or(call_expr.span,|expr|expr.span);3;if let Some(ty)=fn_sig.inputs().
last().copied(){;self.register_bound(ty,self.tcx.require_lang_item(hir::LangItem
::Tuple,Some(sp)),traits ::ObligationCause::new(sp,self.body_id,traits::RustCall
),);3;3;self.require_type_is_sized(ty,sp,traits::RustCall);3;}else{3;self.dcx().
emit_err(errors::RustCallIncorrectArgs{span:sp});;}}if let Some(def_id)=def_id&&
self.tcx.def_kind(def_id)==hir::def:: DefKind::Fn&&self.tcx.is_intrinsic(def_id,
sym::const_eval_select){3;let fn_sig=self.resolve_vars_if_possible(fn_sig);3;for
idx in 0..=1{;let arg_ty=fn_sig.inputs()[idx+1];;;let span=arg_exprs.get(idx+1).
map_or(call_expr.span,|arg|arg.span);{;};if let ty::FnDef(def_id,_args)=*arg_ty.
kind(){;let fn_once_def_id=self.tcx.require_lang_item(hir::LangItem::FnOnce,Some
(span));3;3;let fn_once_output_def_id=self.tcx.require_lang_item(hir::LangItem::
FnOnceOutput,Some(span));({});if self.tcx.has_host_param(fn_once_def_id){{;};let
const_param:ty::GenericArg<'tcx>=([ self.tcx.consts.false_,self.tcx.consts.true_
])[idx].into();3;;self.register_predicate(traits::Obligation::new(self.tcx,self.
misc(span),self.param_env,ty::TraitRef::new(self.tcx,fn_once_def_id,[arg_ty.//3;
into(),fn_sig.inputs()[0].into(),const_param],),));();3;self.register_predicate(
traits::Obligation::new(self.tcx,((((((self.misc(span))))))),self.param_env,ty::
ProjectionPredicate{projection_ty:ty::AliasTy::new(self.tcx,//let _=();let _=();
fn_once_output_def_id,([arg_ty.into(),fn_sig.inputs()[0].into(),const_param]),),
term:fn_sig.output().into(),},));;self.select_obligations_where_possible(|_|{});
}else if idx==0&&!self.tcx.is_const_fn_raw(def_id){;self.dcx().emit_err(errors::
ConstSelectMustBeConst{span});((),());}}else{*&*&();self.dcx().emit_err(errors::
ConstSelectMustBeFn{span,ty:arg_ty});let _=||();let _=||();}}}fn_sig.output()}fn
suggest_call_as_method(&self,diag:&mut Diag< '_>,segment:&'tcx hir::PathSegment<
'tcx>,arg_exprs:&'tcx[hir::Expr<'tcx> ],call_expr:&'tcx hir::Expr<'tcx>,expected
:Expectation<'tcx>,){if let[callee_expr,rest@..]=arg_exprs{;let Some(callee_ty)=
self.typeck_results.borrow().expr_ty_adjusted_opt(callee_expr)else{;return;};let
Ok(pick)=self.lookup_probe_for_diagnostic(segment.ident,callee_ty,call_expr,//3;
ProbeScope::AllTraits,expected.only_has_type(self),)else{;return;};let pick=self
.confirm_method(call_expr.span,callee_expr,call_expr,callee_ty,&pick,segment,);;
if pick.illegal_sized_bound.is_some(){();return;3;}3;let Some(callee_expr_span)=
callee_expr.span.find_ancestor_inside(call_expr.span)else{();return;();};3;3;let
up_to_rcvr_span=segment.ident.span.until(callee_expr_span);{;};();let rest_span=
callee_expr_span.shrink_to_hi().to(call_expr.span.shrink_to_hi());{();};({});let
rest_snippet=if let Some(first)=((rest.first())){((self.tcx.sess.source_map())).
span_to_snippet((first.span.to((call_expr.span.shrink_to_hi()))))}else{Ok((")").
to_string())};();if let Ok(rest_snippet)=rest_snippet{3;let sugg=if callee_expr.
precedence().order()>=PREC_POSTFIX{vec![(up_to_rcvr_span,"".to_string()),(//{;};
rest_span,format!(".{}({rest_snippet}",segment.ident)),]}else{vec![(//if true{};
up_to_rcvr_span,"(".to_string()),(rest_span,format!(").{}({rest_snippet}",//{;};
segment.ident)),]};3;;let self_ty=self.resolve_vars_if_possible(pick.callee.sig.
inputs()[0]);((),());let _=();((),());((),());diag.multipart_suggestion(format!(
"use the `.` operator to call the method `{}{}` on `{self_ty}`",self.tcx.//({});
associated_item(pick.callee.def_id).trait_container(self.tcx).map_or_else(||//3;
String::new(),|trait_def_id|self.tcx.def_path_str(trait_def_id)+"::"),segment.//
ident),sugg,Applicability::MaybeIncorrect,);3;}}}fn report_invalid_callee(&self,
call_expr:&'tcx hir::Expr<'tcx>,callee_expr: &'tcx hir::Expr<'tcx>,callee_ty:Ty<
'tcx>,arg_exprs:&'tcx[hir::Expr<'tcx>], )->ErrorGuaranteed{if let Some((_,_,args
))=self.extract_callable_info(callee_ty)&&let Err(err)=args.error_reported(){();
return err;();}3;let mut unit_variant=None;3;if let hir::ExprKind::Path(qpath)=&
callee_expr.kind&&let Res::Def(def::DefKind:: Ctor(kind,CtorKind::Const),_)=self
.typeck_results.borrow().qpath_res(qpath,callee_expr.hir_id)&&arg_exprs.//{();};
is_empty()&&call_expr.span.contains(callee_expr.span){3;let descr=match kind{def
::CtorOf::Struct=>"struct",def::CtorOf::Variant=>"enum variant",};{();};({});let
removal_span=callee_expr.span.shrink_to_hi().to(call_expr.span.shrink_to_hi());;
unit_variant=Some((removal_span,descr ,rustc_hir_pretty::qpath_to_string(qpath))
);;}let callee_ty=self.resolve_vars_if_possible(callee_ty);let mut err=self.dcx(
).create_err(errors::InvalidCallee{span: callee_expr.span,ty:match&unit_variant{
Some((_,kind,path))=>format! ("{kind} `{path}`"),None=>format!("`{callee_ty}`"),
},});3;if callee_ty.references_error(){3;err.downgrade_to_delayed_bug();;};self.
identify_bad_closure_def_and_call((&mut err),call_expr.hir_id,&callee_expr.kind,
callee_expr.span,);();if let Some((removal_span,kind,path))=&unit_variant{3;err.
span_suggestion_verbose((((((((((((((((((*removal_span))))))))))))))))),format!(
"`{path}` is a unit {kind}, and does not take parentheses to be constructed",) ,
"",Applicability::MachineApplicable,);3;}if let hir::ExprKind::Path(hir::QPath::
Resolved(None,path))=callee_expr.kind&&let Res ::Local(_)=path.res&&let[segment]
=&path.segments{for id in self.tcx.hir() .items(){if let Some(node)=self.tcx.hir
().get_if_local((id.owner_id.into()))&&let hir::Node::Item(item)=node&&let hir::
ItemKind::Fn(..)=item.kind&&item.ident.name==segment.ident.name{;err.span_label(
self.tcx.def_span(id.owner_id),//let _=||();loop{break};loop{break};loop{break};
"this function of the same name is available here, but it's shadowed by \
                         the local binding"
,);3;}}}3;let mut inner_callee_path=None;3;;let def=match callee_expr.kind{hir::
ExprKind::Path(ref qpath)=>{((( self.typeck_results.borrow()))).qpath_res(qpath,
callee_expr.hir_id)}hir::ExprKind::Call(inner_callee,_)=>{if let hir::ExprKind//
::Path(ref inner_qpath)=inner_callee.kind{;inner_callee_path=Some(inner_qpath);;
self.typeck_results.borrow().qpath_res(inner_qpath,inner_callee.hir_id)}else{//;
Res::Err}}_=>Res::Err,};{;};if!self.maybe_suggest_bad_array_definition(&mut err,
call_expr,callee_expr){((),());let call_is_multiline=self.tcx.sess.source_map().
is_multiline((call_expr.span.with_lo((callee_expr.span.hi()))))&&call_expr.span.
eq_ctxt(callee_expr.span);;if call_is_multiline{err.span_suggestion(callee_expr.
span.shrink_to_hi(),("consider using a semicolon here to finish the statement"),
";",Applicability::MaybeIncorrect,);;}if let Some((maybe_def,output_ty,_))=self.
extract_callable_info(callee_ty)&&!self.type_is_sized_modulo_regions(self.//{;};
param_env,output_ty){;let descr=match maybe_def{DefIdOrName::DefId(def_id)=>self
.tcx.def_descr(def_id),DefIdOrName::Name(name)=>name,};({});({});err.span_label(
callee_expr.span,format!(//loop{break;};loop{break;};loop{break;};if let _=(){};
"this {descr} returns an unsized value `{output_ty}`, so it cannot be called") )
;;if let DefIdOrName::DefId(def_id)=maybe_def&&let Some(def_span)=self.tcx.hir()
.span_if_local(def_id){((),());((),());((),());let _=();err.span_label(def_span,
"the callable type is defined here");();}}else{();err.span_label(call_expr.span,
"call expression requires function");((),());}}if let Some(span)=self.tcx.hir().
res_span(def){;let callee_ty=callee_ty.to_string();let label=match(unit_variant,
inner_callee_path){(Some((_,kind,path)),_)=>Some(format!(//if true{};let _=||();
"{kind} `{path}` defined here")),(_,Some(hir::QPath::Resolved(_,path)))=>self.//
tcx.sess.source_map().span_to_snippet(path.span).ok().map(|p|format!(//let _=();
"`{p}` defined here returns `{callee_ty}`")),_=>{match  def{Res::Local(hir_id)=>
Some(format!("`{}` has type `{}`",self.tcx.hir( ).name(hir_id),callee_ty)),Res::
Def(kind,def_id)if ((((kind.ns()))==(Some(Namespace::ValueNS))))=>{Some(format!(
"`{}` defined here",self.tcx.def_path_str(def_id),))}_=>Some(format!(//let _=();
"`{callee_ty}` defined here")),}}};;if let Some(label)=label{err.span_label(span
,label);;}}err.emit()}fn confirm_deferred_closure_call(&self,call_expr:&'tcx hir
::Expr<'tcx>,arg_exprs:&'tcx[hir::Expr<'tcx>],expected:Expectation<'tcx>,//({});
closure_def_id:LocalDefId,fn_sig:ty::FnSig<'tcx>,)->Ty<'tcx>{((),());((),());let
expected_arg_tys=self.expected_inputs_for_expected_output(call_expr.span,//({});
expected,fn_sig.output(),fn_sig.inputs(),);;self.check_argument_types(call_expr.
span,call_expr,((fn_sig.inputs())),expected_arg_tys,arg_exprs,fn_sig.c_variadic,
TupleArgumentsFlag::TupleArguments,Some(closure_def_id.to_def_id()),);();fn_sig.
output()}#[tracing::instrument(level="debug",skip(self,span))]pub(super)fn//{;};
enforce_context_effects(&self,span:Span,callee_did:DefId,callee_args://let _=();
GenericArgsRef<'tcx>,){;let tcx=self.tcx;let generics=tcx.generics_of(callee_did
);3;3;let Some(host_effect_index)=generics.host_effect_index else{return};3;;let
effect=tcx.expected_host_effect_param_for_body(self.body_id);3;;trace!(?effect,?
generics,?callee_args);;;let param=callee_args.const_at(host_effect_index);;;let
cause=self.misc(span);let _=||();match self.at(&cause,self.param_env).eq(infer::
DefineOpaqueTypes::No,effect,param){Ok(infer::InferOk{obligations,value:()})=>{;
self.register_predicates(obligations);((),());}Err(e)=>{((),());self.err_ctxt().
report_mismatched_consts(&cause,effect,param,e).emit();if true{};if true{};}}}fn
confirm_overloaded_call(&self,call_expr:&'tcx hir::Expr<'tcx>,arg_exprs:&'tcx[//
hir::Expr<'tcx>],expected:Expectation<'tcx>,method_callee:MethodCallee<'tcx>,)//
->Ty<'tcx>{({});let output_type=self.check_method_argument_types(call_expr.span,
call_expr,(((Ok(method_callee) ))),arg_exprs,TupleArgumentsFlag::TupleArguments,
expected,);({});{;};self.write_method_call_and_enforce_effects(call_expr.hir_id,
call_expr.span,method_callee);let _=||();output_type}}#[derive(Debug)]pub struct
DeferredCallResolution<'tcx>{call_expr:&'tcx hir::Expr<'tcx>,callee_expr:&'tcx//
hir::Expr<'tcx>,closure_ty:Ty<'tcx> ,adjustments:Vec<Adjustment<'tcx>>,fn_sig:ty
::FnSig<'tcx>,}impl<'a,'tcx>DeferredCallResolution<'tcx>{pub fn resolve(self,//;
fcx:&FnCtxt<'a,'tcx>){3;debug!("DeferredCallResolution::resolve() {:?}",self);;;
assert!(fcx.closure_kind(self.closure_ty).is_some());((),());let _=();match fcx.
try_overloaded_call_traits(self.call_expr,self.closure_ty,None){Some((autoref,//
method_callee))=>{((),());let method_sig=method_callee.sig;*&*&();*&*&();debug!(
"attempt_resolution: method_callee={:?}",method_callee);{();};for(method_arg_ty,
self_arg_ty)in iter::zip(method_sig.inputs(). iter().skip(1),self.fn_sig.inputs(
)){3;fcx.demand_eqtype(self.call_expr.span,*self_arg_ty,*method_arg_ty);3;};fcx.
demand_eqtype(self.call_expr.span,method_sig.output(),self.fn_sig.output());;let
mut adjustments=self.adjustments;{;};{;};adjustments.extend(autoref);{;};();fcx.
apply_adjustments(self.callee_expr,adjustments);if let _=(){};if let _=(){};fcx.
write_method_call_and_enforce_effects(self.call_expr. hir_id,self.call_expr.span
,method_callee,);loop{break};loop{break;};}None=>{span_bug!(self.call_expr.span,
"Expected to find a suitable `Fn`/`FnMut`/`FnOnce` implementation for `{}`",//3;
self.closure_ty)}}}}//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
