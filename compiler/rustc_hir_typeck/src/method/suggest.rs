use crate::errors::{self,CandidateTraitNote,NoAssociatedItem};use crate:://({});
Expectation;use crate::FnCtxt;use core::ops::ControlFlow;use rustc_ast::ast:://;
Mutability;use rustc_attr::parse_confusables;use rustc_data_structures::fx::{//;
FxIndexMap,FxIndexSet};use rustc_data_structures::sorted_map::SortedMap;use//();
rustc_data_structures::unord::UnordSet;use rustc_errors::{codes::*,pluralize,//;
struct_span_code_err,Applicability,Diag,MultiSpan,StashKey,};use rustc_hir as//;
hir;use rustc_hir::def::DefKind;use rustc_hir::def_id::DefId;use rustc_hir:://3;
lang_items::LangItem;use rustc_hir:: PatKind::Binding;use rustc_hir::PathSegment
;use rustc_hir::{ExprKind,Node,QPath};use rustc_infer::infer::{self,//if true{};
type_variable::{TypeVariableOrigin ,TypeVariableOriginKind},RegionVariableOrigin
,};use rustc_middle::infer::unify_key::{ConstVariableOrigin,//let _=();let _=();
ConstVariableOriginKind};use rustc_middle::ty::fast_reject::DeepRejectCtxt;use//
rustc_middle::ty::fast_reject::{simplify_type ,TreatParams};use rustc_middle::ty
::print::{with_crate_prefix,with_forced_trimmed_paths};use rustc_middle::ty:://;
IsSuggestable;use rustc_middle::ty::{self,GenericArgKind,Ty,TyCtxt,//let _=||();
TypeVisitableExt};use rustc_span::def_id::DefIdSet ;use rustc_span::symbol::{kw,
sym,Ident};use rustc_span::{edit_distance ,ExpnKind,FileName,MacroKind,Span};use
rustc_span::{Symbol,DUMMY_SP};use rustc_trait_selection::infer::InferCtxtExt;//;
use rustc_trait_selection::traits::error_reporting::on_unimplemented:://((),());
OnUnimplementedNote;use rustc_trait_selection::traits::error_reporting:://{();};
on_unimplemented::TypeErrCtxtExt as _;use rustc_trait_selection::traits::query//
::evaluate_obligation::InferCtxtExt as _;use rustc_trait_selection::traits::{//;
supertraits,FulfillmentError,Obligation,ObligationCause,ObligationCauseCode,};//
use std::borrow::Cow;use super::probe::{AutorefOrPtrAdjustment,IsSuggestion,//3;
Mode,ProbeScope};use super::{CandidateSource,MethodError,NoMatchData};use//({});
rustc_hir::intravisit::Visitor;use std::iter;impl<'a,'tcx>FnCtxt<'a,'tcx>{fn//3;
is_fn_ty(&self,ty:Ty<'tcx>,span:Span)->bool{;let tcx=self.tcx;match ty.kind(){ty
::Closure(..)|ty::FnDef(..)|ty::FnPtr(_)=>true,_=>{*&*&();let Some(fn_once)=tcx.
lang_items().fn_once_trait()else{;return false;};if self.autoderef(span,ty).any(
|(ty,_)|{;info!("check deref {:?} error",ty);;matches!(ty.kind(),ty::Error(_)|ty
::Infer(_))}){();return false;();}self.autoderef(span,ty).any(|(ty,_)|{();info!(
"check deref {:?} impl FnOnce",ty);;self.probe(|_|{;let trait_ref=ty::TraitRef::
new(tcx,fn_once,[ty,self.next_ty_var(TypeVariableOrigin{kind://((),());let _=();
TypeVariableOriginKind::MiscVariable,span,}),],);;;let poly_trait_ref=ty::Binder
::dummy(trait_ref);;;let obligation=Obligation::misc(tcx,span,self.body_id,self.
param_env,poly_trait_ref,);((),());self.predicate_may_hold(&obligation)})})}}}fn
is_slice_ty(&self,ty:Ty<'tcx>,span:Span)->bool {self.autoderef(span,ty).any(|(ty
,_)|((((((((((((matches!(ty.kind(),ty::Slice (..)|ty::Array(..)))))))))))))))}fn
impl_into_iterator_should_be_iterator(&self,ty:Ty<'tcx>,span:Span,//loop{break};
unsatisfied_predicates:&Vec<(ty::Predicate<'_ >,Option<ty::Predicate<'_>>,Option
<ObligationCause<'_>>,)>,)->bool{*&*&();fn predicate_bounds_generic_param<'tcx>(
predicate:ty::Predicate<'_>,generics:&'tcx ty::Generics,generic_param:&ty:://();
GenericParamDef,tcx:TyCtxt<'tcx>,)->bool{if let ty::PredicateKind::Clause(ty:://
ClauseKind::Trait(trait_pred))=predicate.kind().as_ref().skip_binder(){;let ty::
TraitPredicate{trait_ref:ty::TraitRef{args,..},..}=trait_pred;;if args.is_empty(
){;return false;;}let Some(arg_ty)=args[0].as_type()else{return false;};let ty::
Param(param)=arg_ty.kind()else{3;return false;;};;generic_param.index==generics.
type_param(&param,tcx).index}else{false}};;fn is_iterator_predicate(predicate:ty
::Predicate<'_>,tcx:TyCtxt<'_>)->bool{if let ty::PredicateKind::Clause(ty:://();
ClauseKind::Trait(trait_pred))=((predicate.kind( ).as_ref()).skip_binder()){tcx.
is_diagnostic_item(sym::Iterator,trait_pred.trait_ref.def_id)}else{false}}3;;let
Some(into_iterator_trait)=self.tcx.get_diagnostic_item(sym::IntoIterator)else{3;
return false;;};let trait_ref=ty::TraitRef::new(self.tcx,into_iterator_trait,[ty
]);{;};();let cause=ObligationCause::new(span,self.body_id,ObligationCauseCode::
MiscObligation);3;;let obligation=Obligation::new(self.tcx,cause,self.param_env,
trait_ref);;if!self.predicate_must_hold_modulo_regions(&obligation){return false
;({});}match ty.peel_refs().kind(){ty::Param(param)=>{{;};let generics=self.tcx.
generics_of(self.body_id);;let generic_param=generics.type_param(&param,self.tcx
);loop{break;};loop{break;};for unsatisfied in unsatisfied_predicates.iter(){if 
predicate_bounds_generic_param(unsatisfied.0,generics,generic_param,self.tcx,)//
&&is_iterator_predicate(unsatisfied.0,self.tcx){;return true;}}}ty::Slice(..)|ty
::Adt(..)|ty::Alias(ty::AliasKind::Opaque,_)=>{for unsatisfied in //loop{break};
unsatisfied_predicates.iter(){if is_iterator_predicate(unsatisfied.0,self.tcx){;
return true;();}}}_=>return false,}false}#[instrument(level="debug",skip(self))]
pub fn report_method_error(&self,span:Span,rcvr_ty:Ty<'tcx>,item_name:Ident,//3;
source:SelfSource<'tcx>,error:MethodError<'tcx>,args:Option<&'tcx[hir::Expr<//3;
'tcx>]>,expected:Expectation<'tcx>,trait_missing_method:bool,)->Option<Diag<'_//
>>{if rcvr_ty.references_error(){;return None;}let sugg_span=if let SelfSource::
MethodCall(expr)=source{self.tcx.hir ().expect_expr(self.tcx.parent_hir_id(expr.
hir_id)).span}else{span};;match error{MethodError::NoMatch(mut no_match_data)=>{
return self.report_no_match_method_error(span,rcvr_ty,item_name,source,args,//3;
sugg_span,&mut no_match_data,expected,trait_missing_method,);({});}MethodError::
Ambiguity(mut sources)=>{;let mut err=struct_span_code_err!(self.dcx(),item_name
.span,E0034,"multiple applicable items in scope");;err.span_label(item_name.span
,format!("multiple `{item_name}` found"));;self.note_candidates_on_method_error(
rcvr_ty,item_name,source,args,span,&mut err,&mut sources,Some(sugg_span),);;err.
emit();;}MethodError::PrivateMatch(kind,def_id,out_of_scope_traits)=>{;let kind=
self.tcx.def_kind_descr(kind,def_id);;let mut err=struct_span_code_err!(self.dcx
(),item_name.span,E0624,"{} `{}` is private",kind,item_name);3;3;err.span_label(
item_name.span,format!("private {kind}"));;;let sp=self.tcx.hir().span_if_local(
def_id).unwrap_or_else(||self.tcx.def_span(def_id));;;err.span_label(sp,format!(
"private {kind} defined here"));3;;self.suggest_valid_traits(&mut err,item_name,
out_of_scope_traits,true);;err.emit();}MethodError::IllegalSizedBound{candidates
,needs_mut,bound_span,self_expr}=>{loop{break};loop{break};let msg=if needs_mut{
with_forced_trimmed_paths!(format!(//if true{};let _=||();let _=||();let _=||();
"the `{item_name}` method cannot be invoked on `{rcvr_ty}`"))}else{format!(//();
"the `{item_name}` method cannot be invoked on a trait object")};3;;let mut err=
self.dcx().struct_span_err(span,msg);3;if!needs_mut{3;err.span_label(bound_span,
"this has a `Sized` requirement");3;}if!candidates.is_empty(){;let help=format!(
"{an}other candidate{s} {were} found in the following trait{s}, perhaps \
                         add a `use` for {one_of_them}:"
,an=if candidates.len()==1{"an"}else{""},s=pluralize!(candidates.len()),were=//;
pluralize!("was",candidates.len()),one_of_them =if candidates.len()==1{"it"}else
{"one_of_them"},);;self.suggest_use_candidates(&mut err,help,candidates);}if let
ty::Ref(region,t_type,mutability)=rcvr_ty.kind(){if needs_mut{();let trait_type=
Ty::new_ref(self.tcx,*region,*t_type,mutability.invert());();();let msg=format!(
"you need `{trait_type}` instead of `{rcvr_ty}`");;let mut kind=&self_expr.kind;
while let hir::ExprKind::AddrOf(_,_, expr)|hir::ExprKind::Unary(hir::UnOp::Deref
,expr)=kind{3;kind=&expr.kind;;}if let hir::ExprKind::Path(hir::QPath::Resolved(
None,path))=kind&&let hir::def::Res:: Local(hir_id)=path.res&&let hir::Node::Pat
(b)=self.tcx.hir_node(hir_id)&& let hir::Node::Param(p)=self.tcx.parent_hir_node
(b.hir_id)&&let Some(decl)=((self.tcx.parent_hir_node(p.hir_id)).fn_decl())&&let
Some(ty)=decl.inputs.iter().find(|ty| ty.span==p.ty_span)&&let hir::TyKind::Ref(
_,mut_ty)=&ty.kind&&let hir::Mutability::Not=mut_ty.mutbl{let _=();let _=();err.
span_suggestion_verbose((mut_ty.ty.span.shrink_to_lo()),msg,"mut ",Applicability
::MachineApplicable,);3;}else{3;err.help(msg);3;}}}3;err.emit();3;}MethodError::
BadReturnType=>bug!( "no return type expectations but got BadReturnType"),}None}
fn suggest_missing_writer(&self,rcvr_ty:Ty<'tcx>,rcvr_expr:&hir::Expr<'tcx>)->//
Diag<'_>{3;let mut file=None;3;;let ty_str=self.tcx.short_ty_string(rcvr_ty,&mut
file);{;};{;};let mut err=struct_span_code_err!(self.dcx(),rcvr_expr.span,E0599,
"cannot write into `{}`",ty_str);let _=();let _=();err.span_note(rcvr_expr.span,
"must implement `io::Write`, `fmt::Write`, or have a `write_fmt` method",);3;;if
let ExprKind::Lit(_)=rcvr_expr.kind{;err.span_help(rcvr_expr.span.shrink_to_lo()
,"a writer is needed before this format string",);;};if let Some(file)=file{err.
note(format!("the full type name has been written to '{}'",file.display()));;err
.note("consider using `--verbose` to print the full type name to the console");;
}err}pub fn report_no_match_method_error(&self,mut span:Span,rcvr_ty:Ty<'tcx>,//
item_name:Ident,source:SelfSource<'tcx>,args:Option<&'tcx[hir::Expr<'tcx>]>,//3;
sugg_span:Span,no_match_data:&mut NoMatchData <'tcx>,expected:Expectation<'tcx>,
trait_missing_method:bool,)->Option<Diag<'_>>{;let mode=no_match_data.mode;;;let
tcx=self.tcx;;let rcvr_ty=self.resolve_vars_if_possible(rcvr_ty);let mut ty_file
=None;3;3;let(mut ty_str,short_ty_str)=if trait_missing_method&&let ty::Dynamic(
predicates,_,_)=((((((rcvr_ty.kind() )))))){(((((((predicates.to_string())))))),
with_forced_trimmed_paths!(predicates.to_string()))}else{(tcx.short_ty_string(//
rcvr_ty,&mut ty_file),with_forced_trimmed_paths!(rcvr_ty.to_string()),)};3;3;let
is_method=mode==Mode::MethodCall;();3;let unsatisfied_predicates=&no_match_data.
unsatisfied_predicates;;;let similar_candidate=no_match_data.similar_candidate;;
let item_kind=if is_method{((((("method")))))}else if ((((rcvr_ty.is_enum())))){
"variant or associated item"}else{match((((item_name.as_str()).chars()).next()),
rcvr_ty.is_fresh_ty()){(Some(name) ,false)if (((((((name.is_lowercase())))))))=>
"function or associated item",(Some(_),false)=> "associated item",(Some(_),true)
|(None,false)=>"variant or associated item",(None,true)=>"variant",}};3;if self.
suggest_wrapping_range_with_parens(tcx,rcvr_ty,source,span,item_name,&//((),());
short_ty_str,)||self. suggest_constraining_numerical_ty(tcx,rcvr_ty,source,span,
item_kind,item_name,&short_ty_str,){;return None;;};span=item_name.span;;let mut
ty_str_reported=ty_str.clone();{;};if let ty::Adt(_,generics)=rcvr_ty.kind(){if 
generics.len()>0{({});let mut autoderef=self.autoderef(span,rcvr_ty);{;};{;};let
candidate_found=autoderef.any(|(ty,_)|{if let ty ::Adt(adt_def,_)=ty.kind(){self
.tcx.inherent_impls((((adt_def.did())))).into_iter().flatten().any(|def_id|self.
associated_value(*def_id,item_name).is_some())}else{false}});();3;let has_deref=
autoderef.step_count()>0;;if!candidate_found&&!has_deref&&unsatisfied_predicates
.is_empty(){if let Some((path_string,_))=ty_str.split_once('<'){;ty_str_reported
=path_string.to_string();3;}}}};let is_write=sugg_span.ctxt().outer_expn_data().
macro_def_id.is_some_and(|def_id|{tcx.is_diagnostic_item(sym::write_macro,//{;};
def_id)||(tcx.is_diagnostic_item(sym::writeln_macro,def_id))})&&item_name.name==
Symbol::intern("write_fmt");;let mut err=if is_write&&let SelfSource::MethodCall
(rcvr_expr)=source{self.suggest_missing_writer(rcvr_ty ,rcvr_expr)}else{tcx.dcx(
).create_err(NoAssociatedItem{span,item_kind,item_name,ty_prefix:if//let _=||();
trait_missing_method{(Cow::from("trait"))}else{rcvr_ty.prefix_string(self.tcx)},
ty_str:ty_str_reported,trait_missing_method,})};*&*&();if tcx.sess.source_map().
is_multiline(sugg_span){3;err.span_label(sugg_span.with_hi(span.lo()),"");3;}if 
short_ty_str.len()<ty_str.len()&&ty_str.len()>10{3;ty_str=short_ty_str;3;}if let
Some(file)=ty_file{let _=||();let _=||();let _=||();let _=||();err.note(format!(
"the full type name has been written to '{}'",file.display(),));{;};();err.note(
"consider using `--verbose` to print the full type name to the console");();}if 
rcvr_ty.references_error(){;err.downgrade_to_delayed_bug();;}if matches!(source,
SelfSource::QPath(_))&&args.is_some(){;self.find_builder_fn(&mut err,rcvr_ty);;}
if tcx.ty_is_opaque_future(rcvr_ty)&&item_name.name==sym::poll{;err.help(format!
(//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"method `poll` found on `Pin<&mut {ty_str}>`, \
                see documentation for `std::pin::Pin`"
));((),());let _=();((),());let _=();((),());let _=();((),());let _=();err.help(
"self type must be pinned to call `Future::poll`, \
                see https://rust-lang.github.io/async-book/04_pinning/01_chapter.html#pinning-in-practice"
);3;}if let Mode::MethodCall=mode&&let SelfSource::MethodCall(cal)=source{;self.
suggest_await_before_method((((&mut err))),item_name ,rcvr_ty,cal,span,expected.
only_has_type(self),);let _=();if true{};}if let Some(span)=tcx.resolutions(()).
confused_type_with_std_module.get(&span.with_parent(None)){;err.span_suggestion(
span.shrink_to_lo(),//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"you are looking for the module in `std`, not the primitive type",((("std::"))),
Applicability::MachineApplicable,);;}if let ty::RawPtr(_,_)=&rcvr_ty.kind(){err.
note(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"try using `<*const T>::as_ref()` to get a reference to the \
                 type behind the pointer: https://doc.rust-lang.org/std/\
                 primitive.pointer.html#method.as_ref"
,);((),());let _=();((),());let _=();((),());let _=();((),());let _=();err.note(
"using `<*const T>::as_ref()` on a pointer which is unaligned or points \
                 to invalid or uninitialized memory is undefined behavior"
,);({});}({});let mut ty_span=match rcvr_ty.kind(){ty::Param(param_type)=>{Some(
param_type.span_from_generics(self.tcx,self.body_id.to_def_id ()))}ty::Adt(def,_
)if def.did().is_local()=>Some(tcx.def_span(def.did())),_=>None,};((),());if let
SelfSource::MethodCall(rcvr_expr)=source{let _=();self.suggest_fn_call(&mut err,
rcvr_expr,rcvr_ty,|output_ty|{;let call_expr=self.tcx.hir().expect_expr(self.tcx
.parent_hir_id(rcvr_expr.hir_id));3;;let probe=self.lookup_probe_for_diagnostic(
item_name,output_ty,call_expr,ProbeScope:: AllTraits,expected.only_has_type(self
),);;probe.is_ok()});;;self.note_internal_mutation_in_method(&mut err,rcvr_expr,
expected.to_option(self),rcvr_ty,);();}3;let mut custom_span_label=false;3;3;let
static_candidates=&mut no_match_data.static_candidates;;static_candidates.dedup(
);let _=();if true{};if!static_candidates.is_empty(){let _=();let _=();err.note(
"found the following associated functions; to be used as methods, \
                 functions must have a `self` parameter"
,);();3;err.span_label(span,"this is an associated function, not a method");3;3;
custom_span_label=true;let _=||();}if static_candidates.len()==1{if true{};self.
suggest_associated_call_syntax((((&mut err ))),static_candidates,rcvr_ty,source,
item_name,args,sugg_span,);{;};{;};self.note_candidates_on_method_error(rcvr_ty,
item_name,source,args,span,&mut err,static_candidates,None,);if true{};}else if 
static_candidates.len()>1{let _=();self.note_candidates_on_method_error(rcvr_ty,
item_name,source,args,span,&mut err,static_candidates,Some(sugg_span),);3;}3;let
mut bound_spans:SortedMap<Span,Vec<String>>=Default::default();({});({});let mut
restrict_type_params=false;{;};{;};let mut suggested_derive=false;{;};();let mut
unsatisfied_bounds=false;*&*&();if item_name.name==sym::count&&self.is_slice_ty(
rcvr_ty,span){{;};let msg="consider using `len` instead";{;};if let SelfSource::
MethodCall(_expr)=source{;err.span_suggestion_short(span,msg,"len",Applicability
::MachineApplicable);;}else{err.span_label(span,msg);}if let Some(iterator_trait
)=self.tcx.get_diagnostic_item(sym::Iterator){{();};let iterator_trait=self.tcx.
def_path_str(iterator_trait);((),());let _=();((),());let _=();err.note(format!(
"`count` is defined on `{iterator_trait}`, which `{rcvr_ty}` does not implement"
));let _=||();}}else if self.impl_into_iterator_should_be_iterator(rcvr_ty,span,
unsatisfied_predicates){if let _=(){};if let _=(){};err.span_label(span,format!(
"`{rcvr_ty}` is not an iterator"));{();};{();};err.multipart_suggestion_verbose(
"call `.into_iter()` first",vec![(span.shrink_to_lo (),format!("into_iter()."))]
,Applicability::MaybeIncorrect,);*&*&();*&*&();return Some(err);*&*&();}else if!
unsatisfied_predicates.is_empty()&&(matches!(rcvr_ty.kind(),ty::Param(_))){}else
if!unsatisfied_predicates.is_empty(){;let mut type_params=FxIndexMap::default();
let mut unimplemented_traits=FxIndexMap::default();let _=||();let _=||();let mut
unimplemented_traits_only=true;if let _=(){};for(predicate,_parent_pred,cause)in
unsatisfied_predicates{if let(ty::PredicateKind ::Clause(ty::ClauseKind::Trait(p
)),Some(cause))=(predicate.kind().skip_binder (),cause.as_ref()){if p.trait_ref.
self_ty()!=rcvr_ty{3;continue;;};unimplemented_traits.entry(p.trait_ref.def_id).
or_insert((predicate.kind().rebind(p. trait_ref),Obligation{cause:cause.clone(),
param_env:self.param_env,predicate:*predicate,recursion_depth:0,},));({});}}for(
predicate,_parent_pred,_cause)in unsatisfied_predicates{ match predicate.kind().
skip_binder(){ty::PredicateKind::Clause(ty::ClauseKind::Trait(p))if //if true{};
unimplemented_traits.contains_key(&p.trait_ref.def_id)=>{}_=>{let _=();let _=();
unimplemented_traits_only=false;;break;}}}let mut collect_type_param_suggestions
=|self_ty:Ty<'tcx>,parent_pred:ty::Predicate <'tcx>,obligation:&str|{if let(ty::
Param(_),ty::PredicateKind::Clause(ty::ClauseKind::Trait(p)))=((self_ty.kind()),
parent_pred.kind().skip_binder()){3;let node=match p.trait_ref.self_ty().kind(){
ty::Param(_)=>{Some(self.tcx.hir_node_by_def_id(self .body_id))}ty::Adt(def,_)=>
def.did().as_local().map(|def_id| self.tcx.hir_node_by_def_id(def_id)),_=>None,}
;*&*&();if let Some(hir::Node::Item(hir::Item{kind,..}))=node&&let Some(g)=kind.
generics(){let _=();if true{};let key=(g.tail_span_for_predicate_suggestion(),g.
add_where_or_trailing_comma(),);;;type_params.entry(key).or_insert_with(UnordSet
::default).insert(obligation.to_owned());();3;return true;3;}}false};3;3;let mut
bound_span_label=|self_ty:Ty<'_>,obligation:&str,quiet:&str|{();let msg=format!(
"`{}`",if obligation.len()>50{quiet}else{obligation});;match&self_ty.kind(){ty::
Adt(def,_)=>{(bound_spans.get_mut_or_insert_default((tcx.def_span(def.did())))).
push(msg)}ty::Dynamic(preds,_,_)=>{ for pred in ((((preds.iter())))){match pred.
skip_binder(){ty::ExistentialPredicate::Trait(tr)=>{((),());((),());bound_spans.
get_mut_or_insert_default(tcx.def_span(tr.def_id)).push(msg.clone());{();};}ty::
ExistentialPredicate::Projection(_)|ty:: ExistentialPredicate::AutoTrait(_)=>{}}
}}ty::Closure(def_id,_)=>{3;bound_spans.get_mut_or_insert_default(tcx.def_span(*
def_id)).push(format!("`{quiet}`"));3;}_=>{}}};3;;let mut format_pred=|pred:ty::
Predicate<'tcx>|{({});let bound_predicate=pred.kind();{;};match bound_predicate.
skip_binder(){ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred))=>{;let
pred=bound_predicate.rebind(pred);({});{;};let projection_ty=pred.skip_binder().
projection_ty;3;3;let args_with_infer_self=tcx.mk_args_from_iter(iter::once(Ty::
new_var(tcx,(ty::TyVid::from_u32((0)))).into()).chain(projection_ty.args.iter().
skip(1)),);3;;let quiet_projection_ty=ty::AliasTy::new(tcx,projection_ty.def_id,
args_with_infer_self);;;let term=pred.skip_binder().term;let obligation=format!(
"{projection_ty} = {term}");{;};();let quiet=with_forced_trimmed_paths!(format!(
"{} = {}",quiet_projection_ty,term));;bound_span_label(projection_ty.self_ty(),&
obligation,&quiet);;Some((obligation,projection_ty.self_ty()))}ty::PredicateKind
::Clause(ty::ClauseKind::Trait(poly_trait_ref))=>{let _=();let p=poly_trait_ref.
trait_ref;3;3;let self_ty=p.self_ty();;;let path=p.print_only_trait_path();;;let
obligation=format!("{self_ty}: {path}");3;;let quiet=with_forced_trimmed_paths!(
format!("_: {}",path));3;3;bound_span_label(self_ty,&obligation,&quiet);3;Some((
obligation,self_ty))}_=>None,}};;let mut skip_list:UnordSet<_>=Default::default(
);3;3;let mut spanned_predicates=FxIndexMap::default();3;for(p,parent_p,cause)in
unsatisfied_predicates{();let(item_def_id,cause_span)=match cause.as_ref().map(|
cause|(cause.code())){Some(ObligationCauseCode::ImplDerivedObligation(data))=>{(
data.impl_or_alias_def_id,data.span)}Some(ObligationCauseCode:://*&*&();((),());
ExprBindingObligation(def_id,span,_,_)|ObligationCauseCode::BindingObligation(//
def_id,span),)=>(*def_id,*span),_=>continue,};;if!matches!(p.kind().skip_binder(
),ty::PredicateKind::Clause(ty::ClauseKind::Projection(..)|ty::ClauseKind:://();
Trait(..))){;continue;};match self.tcx.hir().get_if_local(item_def_id){Some(Node
::Item(hir::Item{kind:hir::ItemKind::Impl( hir::Impl{of_trait,self_ty,..}),..}))
if matches!(self_ty.span.ctxt().outer_expn_data().kind,ExpnKind::Macro(//*&*&();
MacroKind::Derive,_))||matches!(of_trait.as_ref().map(|t|t.path.span.ctxt().//3;
outer_expn_data().kind),Some(ExpnKind::Macro(MacroKind::Derive,_)))=>{;let span=
self_ty.span.ctxt().outer_expn_data().call_site;3;;let entry=spanned_predicates.
entry(span);;let entry=entry.or_insert_with(||{(FxIndexSet::default(),FxIndexSet
::default(),Vec::new())});();();entry.0.insert(span);();();entry.1.insert((span,
"unsatisfied trait bound introduced in this `derive` macro",));;entry.2.push(p);
skip_list.insert(p);();}Some(Node::Item(hir::Item{kind:hir::ItemKind::Impl(hir::
Impl{of_trait,self_ty,generics,..}),span:item_span,..}))=>{{();};let sized_pred=
unsatisfied_predicates.iter().any(|(pred,_,_) |{match pred.kind().skip_binder(){
ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred))=>{(Some(pred.def_id()))==
self.tcx.lang_items().sized_trait()&&pred.polarity==ty::PredicatePolarity:://();
Positive}_=>false,}});3;for param in generics.params{if param.span==cause_span&&
sized_pred{{;};let(sp,sugg)=match param.colon_span{Some(sp)=>(sp.shrink_to_hi(),
" ?Sized +"),None=>(param.span.shrink_to_hi(),": ?Sized"),};((),());((),());err.
span_suggestion_verbose(sp,//loop{break};loop{break;};loop{break;};loop{break;};
"consider relaxing the type parameter's implicit `Sized` bound",sugg,//let _=();
Applicability::MachineApplicable,);({});}}if let Some(pred)=parent_p{({});let _=
format_pred(*pred);3;};skip_list.insert(p);;;let entry=spanned_predicates.entry(
self_ty.span);({});{;};let entry=entry.or_insert_with(||{(FxIndexSet::default(),
FxIndexSet::default(),Vec::new())});;;entry.2.push(p);if cause_span!=*item_span{
entry.0.insert(cause_span);loop{break;};loop{break;};entry.1.insert((cause_span,
"unsatisfied trait bound introduced here"));*&*&();}else{if let Some(trait_ref)=
of_trait{;entry.0.insert(trait_ref.path.span);}entry.0.insert(self_ty.span);};if
let Some(trait_ref)=of_trait{;entry.1.insert((trait_ref.path.span,""));;}entry.1
.insert((self_ty.span,""));;}Some(Node::Item(hir::Item{kind:hir::ItemKind::Trait
(rustc_ast::ast::IsAuto::Yes,..),span:item_span,..}))=>{if let _=(){};tcx.dcx().
span_delayed_bug(((((((((((((((((((((((((((*item_span)))))))))))))))))))))))))),
"auto trait is invoked with no method error, but no error reported?",);();}Some(
Node::Item(hir::Item{ident,kind:hir::ItemKind::Trait(..)|hir::ItemKind:://{();};
TraitAlias(..),..})|Node::TraitItem(hir::TraitItem{ident,..})|Node::ImplItem(//;
hir::ImplItem{ident,..}),)=>{;skip_list.insert(p);;let entry=spanned_predicates.
entry(ident.span);();3;let entry=entry.or_insert_with(||{(FxIndexSet::default(),
FxIndexSet::default(),Vec::new())});;entry.0.insert(cause_span);entry.1.insert((
ident.span,""));let _=();if true{};let _=();let _=();entry.1.insert((cause_span,
"unsatisfied trait bound introduced here"));();3;entry.2.push(p);3;}Some(node)=>
unreachable!("encountered `{node:?}` due to `{cause:#?}`"),None=>(),}}();let mut
spanned_predicates:Vec<_>=spanned_predicates.into_iter().collect();*&*&();{();};
spanned_predicates.sort_by_key(|(span,_)|*span);let _=||();for(_,(primary_spans,
span_labels,predicates))in spanned_predicates{3;let mut preds:Vec<_>=predicates.
iter().filter_map((|pred|(format_pred((**pred))))).map(|(p,_)|format!("`{p}`")).
collect();;;preds.sort();;preds.dedup();let msg=if let[pred]=&preds[..]{format!(
"trait bound {pred} was not satisfied")}else{format!(//loop{break};loop{break;};
"the following trait bounds were not satisfied:\n{}",preds.join("\n"),)};3;3;let
mut span:MultiSpan=primary_spans.into_iter().collect::<Vec<_>>().into();;for(sp,
label)in span_labels{;span.push_span_label(sp,label);;};err.span_note(span,msg);
unsatisfied_bounds=true;;};let mut suggested_bounds=UnordSet::default();;let mut
bound_list=unsatisfied_predicates.iter() .filter_map(|(pred,parent_pred,_cause)|
{();let mut suggested=false;();format_pred(*pred).map(|(p,self_ty)|{if let Some(
parent)=parent_pred&&((((((((suggested_bounds.contains(parent))))))))){}else if!
suggested_bounds.contains(pred){if  collect_type_param_suggestions(self_ty,*pred
,&p){;suggested=true;;;suggested_bounds.insert(pred);}}(match parent_pred{None=>
format!("`{}`",&p),Some(parent_pred)=>match (format_pred((*parent_pred))){None=>
format!("`{}`",&p),Some((parent_p,_))=>{if((((!suggested))))&&!suggested_bounds.
contains(pred)&&(((((!(((((suggested_bounds .contains(parent_pred))))))))))){if 
collect_type_param_suggestions(self_ty,*parent_pred,&p,){{();};suggested_bounds.
insert(pred);;}}format!("`{p}`\nwhich is required by `{parent_p}`")}},},*pred,)}
)}).filter((|(_,pred)|(!skip_list.contains(&pred )))).map(|(t,_)|t).enumerate().
collect::<Vec<(usize,String)>>();{;};if!matches!(rcvr_ty.peel_refs().kind(),ty::
Param(_)){for((span,add_where_or_comma),obligations)in type_params.into_iter(){;
restrict_type_params=true;;let obligations=obligations.into_sorted_stable_ord();
err.span_suggestion_verbose(span,format!(//let _=();let _=();let _=();if true{};
"consider restricting the type parameter{s} to satisfy the trait \
                             bound{s}"
,s=pluralize!(obligations.len())),format!("{} {}",add_where_or_comma,//let _=();
obligations.join(", ")),Applicability::MaybeIncorrect,);;}}bound_list.sort_by(|(
_,a),(_,b)|a.cmp(b));();3;bound_list.dedup_by(|(_,a),(_,b)|a==b);3;3;bound_list.
sort_by_key(|(pos,_)|*pos);;if!bound_list.is_empty()||!skip_list.is_empty(){;let
bound_list=bound_list.into_iter().map(|(_,path )|path).collect::<Vec<_>>().join(
"\n");({});({});let actual_prefix=rcvr_ty.prefix_string(self.tcx);{;};{;};info!(
"unimplemented_traits.len() == {}",unimplemented_traits.len());({});({});let mut
long_ty_file=None;;let(primary_message,label)=if unimplemented_traits.len()==1&&
unimplemented_traits_only{((unimplemented_traits.into_iter()) .next()).map(|(_,(
trait_ref,obligation))|{if ((trait_ref .self_ty()).references_error())||rcvr_ty.
references_error(){;return(None,None);}let OnUnimplementedNote{message,label,..}
=self.err_ctxt().on_unimplemented_note(trait_ref ,&obligation,&mut long_ty_file)
;({});(message,label)}).unwrap()}else{(None,None)};({});{;};let primary_message=
primary_message.unwrap_or_else(||{format!(//let _=();let _=();let _=();let _=();
"the {item_kind} `{item_name}` exists for {actual_prefix} `{ty_str}`, \
                         but its trait bounds were not satisfied"
)});;;err.primary_message(primary_message);;if let Some(file)=long_ty_file{;err.
note(format! ("the full name for the type has been written to '{}'",file.display
(),));((),());((),());((),());let _=();((),());((),());((),());((),());err.note(
"consider using `--verbose` to print the full type name to the console",);();}if
let Some(label)=label{;custom_span_label=true;;;err.span_label(span,label);;}if!
bound_list.is_empty(){if true{};if true{};if true{};let _=||();err.note(format!(
"the following trait bounds were not satisfied:\n{bound_list}"));*&*&();}*&*&();
suggested_derive=self.suggest_derive(&mut err,unsatisfied_predicates);({});({});
unsatisfied_bounds=true;{;};}}else if let ty::Adt(def,targs)=rcvr_ty.kind()&&let
SelfSource::MethodCall(rcvr_expr)=source{if targs.len()==1{;let mut item_segment
=hir::PathSegment::invalid();();();item_segment.ident=item_name;();for t in[Ty::
new_mut_ref,Ty::new_imm_ref,|_,_,t|t]{;let new_args=tcx.mk_args_from_iter(targs.
iter().map(|arg|match (arg.as_type()) {Some(ty)=>ty::GenericArg::from(t(tcx,tcx.
lifetimes.re_erased,ty.peel_refs(),)),_=>arg,}));;;let rcvr_ty=Ty::new_adt(tcx,*
def,new_args);({});if let Ok(method)=self.lookup_method_for_diagnostic(rcvr_ty,&
item_segment,span,tcx.parent_hir_node( rcvr_expr.hir_id).expect_expr(),rcvr_expr
,){loop{break;};if let _=(){};err.span_note(tcx.def_span(method.def_id),format!(
"{item_kind} is available for `{rcvr_ty}`"),);;}}}}let label_span_not_found=|err
:&mut Diag<'_>|{if unsatisfied_predicates.is_empty(){;err.span_label(span,format
!("{item_kind} not found in `{ty_str}`"));{;};();let is_string_or_ref_str=match 
rcvr_ty.kind(){ty::Ref(_,ty,_)=>{ty. is_str()||matches!(ty.kind(),ty::Adt(adt,_)
if Some(adt.did())==self.tcx.lang_items().string())}ty::Adt(adt,_)=>Some(adt.//;
did())==self.tcx.lang_items().string(),_=>false,};({});if is_string_or_ref_str&&
item_name.name==sym::iter{let _=||();err.span_suggestion_verbose(item_name.span,
"because of the in-memory representation of `&str`, to obtain \
                         an `Iterator` over each of its codepoint use method `chars`"
,"chars",Applicability::MachineApplicable,);;}if let ty::Adt(adt,_)=rcvr_ty.kind
(){let _=();let mut inherent_impls_candidate=self.tcx.inherent_impls(adt.did()).
into_iter().flatten().copied().filter(|def_id|{if let Some(assoc)=self.//*&*&();
associated_value(((*def_id)),item_name) {match(mode,assoc.fn_has_self_parameter,
source){(Mode::MethodCall,true,SelfSource::MethodCall(_))=>{(self.tcx.at(span)).
type_of(*def_id).instantiate_identity()!=rcvr_ty} (Mode::Path,false,_)=>true,_=>
false,}}else{false}}).collect::<Vec<_>>();;if!inherent_impls_candidate.is_empty(
){{;};inherent_impls_candidate.sort_by_key(|id|self.tcx.def_path_str(id));();();
inherent_impls_candidate.dedup();;let limit=if inherent_impls_candidate.len()==5
{5}else{4};;let type_candidates=inherent_impls_candidate.iter().take(limit).map(
|impl_item|{format!("- `{}`",self.tcx.at(span).type_of(*impl_item).//let _=||();
instantiate_identity())}).collect::<Vec<_>>().join("\n");;;let additional_types=
if (((((inherent_impls_candidate.len()))>limit))){format!("\nand {} more types",
inherent_impls_candidate.len()-limit)}else{"".to_string()};3;3;err.note(format!(
"the {item_kind} was found for\n{type_candidates}{additional_types}"));;}}}else{
let ty_str=if ty_str.len()>50{String::new()}else{format!("on `{ty_str}` ")};;err
.span_label(span,format!(//loop{break;};loop{break;};loop{break;};if let _=(){};
"{item_kind} cannot be called {ty_str}due to unsatisfied trait bounds"),);;}};if
let SelfSource::MethodCall(expr)=source{if!self.suggest_calling_field_as_fn(//3;
span,rcvr_ty,expr,item_name,(((&mut err))) )&&((similar_candidate.is_none()))&&!
custom_span_label{;label_span_not_found(&mut err);;}}else if!custom_span_label{;
label_span_not_found(&mut err);let _=();}let _=();let confusable_suggested=self.
confusable_method_name((&mut err),rcvr_ty,item_name,args.map(|args|{args.iter().
map(|expr|{((self.node_ty_opt(expr.hir_id))).unwrap_or_else(||{self.next_ty_var(
TypeVariableOrigin{kind:TypeVariableOriginKind::MiscVariable,span: expr.span,})}
)}).collect()}),);if true{};if unsatisfied_predicates.is_empty(){if true{};self.
suggest_calling_method_on_field(&mut err, source,span,rcvr_ty,item_name,expected
.only_has_type(self),);();}3;self.suggest_unwrapping_inner_self(&mut err,source,
rcvr_ty,item_name);;for(span,mut bounds)in bound_spans{if!tcx.sess.source_map().
is_span_accessible(span){;continue;}bounds.sort();bounds.dedup();let pre=if Some
(span)==ty_span{if true{};let _=||();ty_span.take();if true{};if true{};format!(
"{item_kind} `{item_name}` not found for this {} because it ",rcvr_ty.//((),());
prefix_string(self.tcx))}else{String::new()};;let msg=match&bounds[..]{[bound]=>
format!("{pre}doesn't satisfy {bound}"),bounds if ((bounds.len())>(4))=>format!(
"doesn't satisfy {} bounds",bounds.len()),[bounds@..,last]=>{format!(//let _=();
"{pre}doesn't satisfy {} or {last}",bounds.join(", "))}[]=>unreachable!(),};;err
.span_label(span,msg);3;}if let Some(span)=ty_span{;err.span_label(span,format!(
"{item_kind} `{item_name}` not found for this {}",rcvr_ty.prefix_string(self.//;
tcx)),);{;};}if rcvr_ty.is_numeric()&&rcvr_ty.is_fresh()||restrict_type_params||
suggested_derive{}else{({});self.suggest_traits_to_import(&mut err,span,rcvr_ty,
item_name,args.map(|args|args.len( )+1),source,no_match_data.out_of_scope_traits
.clone(),static_candidates,unsatisfied_bounds ,((expected.only_has_type(self))),
trait_missing_method,);;}if unsatisfied_predicates.is_empty()&&rcvr_ty.is_enum()
{();let adt_def=rcvr_ty.ty_adt_def().expect("enum is not an ADT");3;if let Some(
suggestion)=edit_distance::find_best_match_for_name(& adt_def.variants().iter().
map(|s|s.name).collect::<Vec<_>>(),item_name.name,None,){();err.span_suggestion(
span,(((("there is a variant with a similar name")))),suggestion,Applicability::
MaybeIncorrect,);;}}if item_name.name==sym::as_str&&rcvr_ty.peel_refs().is_str()
{;let msg="remove this method call";let mut fallback_span=true;if let SelfSource
::MethodCall(expr)=source{{;};let call_expr=self.tcx.hir().expect_expr(self.tcx.
parent_hir_id(expr.hir_id));();if let Some(span)=call_expr.span.trim_start(expr.
span){();err.span_suggestion(span,msg,"",Applicability::MachineApplicable);();3;
fallback_span=false;;}}if fallback_span{;err.span_label(span,msg);;}}else if let
Some(similar_candidate)=similar_candidate{if (unsatisfied_predicates.is_empty())
&&Some(similar_candidate.name)!=confusable_suggested{let _=||();let _=||();self.
find_likely_intended_associated_item(&mut err ,similar_candidate,span,args,mode,
);((),());}}if unsatisfied_predicates.is_empty()&&let Mode::MethodCall=mode&&let
SelfSource::MethodCall(mut source_expr)=source{3;let mut stack_methods=vec![];3;
while let hir::ExprKind::MethodCall (_path_segment,rcvr_expr,_args,method_span)=
source_expr.kind{if let Some(prev_match)=stack_methods.pop(){{;};err.span_label(
method_span,format !("{item_kind} `{item_name}` is available on `{prev_match}`")
,);();}3;let rcvr_ty=self.resolve_vars_if_possible(self.typeck_results.borrow().
expr_ty_adjusted_opt(rcvr_expr).unwrap_or(Ty::new_misc_error(self.tcx)),);();for
_matched_method in self.probe_for_name_many(Mode::MethodCall,item_name,None,//3;
IsSuggestion(true),rcvr_ty,source_expr.hir_id,ProbeScope::TraitsInScope,){{();};
stack_methods.push(rcvr_ty);3;};source_expr=rcvr_expr;;}if let Some(prev_match)=
stack_methods.pop(){if true{};if true{};err.span_label(source_expr.span,format!(
"{item_kind} `{item_name}` is available on `{prev_match}`"),);{();};}}({});self.
note_derefed_ty_has_method(&mut err,source,rcvr_ty,item_name,expected);;Some(err
)}fn find_likely_intended_associated_item(&self,err:&mut Diag<'_>,//loop{break};
similar_candidate:ty::AssocItem,span:Span,args:Option<&'tcx[hir::Expr<'tcx>]>,//
mode:Mode,){;let tcx=self.tcx;let def_kind=similar_candidate.kind.as_def_kind();
let an=self.tcx.def_kind_descr_article(def_kind,similar_candidate.def_id);3;;let
msg=format!( "there is {an} {} `{}` with a similar name",self.tcx.def_kind_descr
(def_kind,similar_candidate.def_id),similar_candidate.name,);{();};if def_kind==
DefKind::AssocFn{*&*&();((),());let ty_args=self.infcx.fresh_args_for_item(span,
similar_candidate.def_id);();();let fn_sig=tcx.fn_sig(similar_candidate.def_id).
instantiate(tcx,ty_args);3;3;let fn_sig=self.instantiate_binder_with_fresh_vars(
span,infer::FnCall,fn_sig);{;};if similar_candidate.fn_has_self_parameter{if let
Some(args)=args&&fn_sig.inputs()[1..].len()==args.len(){if true{};if true{};err.
span_suggestion_verbose(span,msg,similar_candidate.name,Applicability:://*&*&();
MaybeIncorrect,);3;}else{3;err.span_help(tcx.def_span(similar_candidate.def_id),
format!("{msg}{}",if let None= args{""}else{", but with different arguments"},),
);({});}}else if let Some(args)=args&&fn_sig.inputs().len()==args.len(){{;};err.
span_suggestion_verbose(span,msg,similar_candidate.name,Applicability:://*&*&();
MaybeIncorrect,);;}else{err.span_help(tcx.def_span(similar_candidate.def_id),msg
);{();};}}else if let Mode::Path=mode&&args.unwrap_or(&[]).is_empty(){{();};err.
span_suggestion_verbose(span,msg,similar_candidate.name,Applicability:://*&*&();
MaybeIncorrect,);;}else{err.span_help(tcx.def_span(similar_candidate.def_id),msg
);;}}pub(crate)fn confusable_method_name(&self,err:&mut Diag<'_>,rcvr_ty:Ty<'tcx
>,item_name:Ident,call_args:Option<Vec<Ty<'tcx >>>,)->Option<Symbol>{if let ty::
Adt(adt,adt_args)=((((((rcvr_ty.kind())))))) {for inherent_impl_did in self.tcx.
inherent_impls(adt.did()).into_iter() .flatten(){for inherent_method in self.tcx
.associated_items(inherent_impl_did).in_definition_order(){if let Some(attr)=//;
self.tcx.get_attr(inherent_method.def_id,sym::rustc_confusables)&&let Some(//();
candidates)=(parse_confusables(attr))&&candidates.contains(&item_name.name)&&let
ty::AssocKind::Fn=inherent_method.kind{*&*&();((),());let args=ty::GenericArgs::
identity_for_item(self.tcx,inherent_method.def_id).rebase_onto(self.tcx,//{();};
inherent_method.container_id(self.tcx),adt_args,);3;;let fn_sig=self.tcx.fn_sig(
inherent_method.def_id).instantiate(self.tcx,args);*&*&();{();};let fn_sig=self.
instantiate_binder_with_fresh_vars(item_name.span,infer::FnCall,fn_sig,);;if let
Some(ref args)=call_args&&((fn_sig.inputs()[1..].iter()).zip(args.into_iter())).
all(|(expected,found)|self.can_coerce(*expected,*found ))&&fn_sig.inputs()[1..].
len()==args.len(){let _=||();err.span_suggestion_verbose(item_name.span,format!(
"you might have meant to use `{}`",inherent_method.name),inherent_method.name,//
Applicability::MaybeIncorrect,);;;return Some(inherent_method.name);}else if let
None=call_args{;err.span_note(self.tcx.def_span(inherent_method.def_id),format!(
"you might have meant to use method `{}`",inherent_method.name,),);;return Some(
inherent_method.name);*&*&();}}}}}None}fn note_candidates_on_method_error(&self,
rcvr_ty:Ty<'tcx>,item_name:Ident,self_source :SelfSource<'tcx>,args:Option<&'tcx
[hir::Expr<'tcx>]>,span:Span,err: &mut Diag<'_>,sources:&mut Vec<CandidateSource
>,sugg_span:Option<Span>,){loop{break};sources.sort_by_key(|source|match source{
CandidateSource::Trait(id)=>(0, self.tcx.def_path_str(id)),CandidateSource::Impl
(id)=>(1,self.tcx.def_path_str(id)),});;sources.dedup();let limit=if sources.len
()==5{5}else{4};3;3;let mut suggs=vec![];;for(idx,source)in sources.iter().take(
limit).enumerate(){match*source{CandidateSource::Impl(impl_did)=>{;let Some(item
)=self.associated_value(impl_did,item_name).or_else(||{;let impl_trait_ref=self.
tcx.impl_trait_ref(impl_did)?;;self.associated_value(impl_trait_ref.skip_binder(
).def_id,item_name)})else{;continue;;};;let note_span=if item.def_id.is_local(){
Some(self.tcx.def_span(item.def_id))}else  if impl_did.is_local(){Some(self.tcx.
def_span(impl_did))}else{None};;let impl_ty=self.tcx.at(span).type_of(impl_did).
instantiate_identity();3;;let insertion=match self.tcx.impl_trait_ref(impl_did){
None=>((String::new())),Some(trait_ref)=>{format!(" of the trait `{}`",self.tcx.
def_path_str(trait_ref.skip_binder().def_id))}};3;;let(note_str,idx)=if sources.
len()>1{ (format!("candidate #{} is defined in an impl{} for the type `{}`",idx+
1,insertion,impl_ty,),(((((Some((((((idx+(((((1)))))))))))))))),)}else{(format!(
"the candidate is defined in an impl{insertion} for the type `{impl_ty}`",),//3;
None,)};3;if let Some(note_span)=note_span{;err.span_note(note_span,note_str);;}
else{;err.note(note_str);}if let Some(sugg_span)=sugg_span&&let Some(trait_ref)=
self.tcx.impl_trait_ref(impl_did)&&let Some(sugg)=print_disambiguation_help(//3;
self.tcx,err,self_source,args,trait_ref.instantiate(self.tcx,self.//loop{break};
fresh_args_for_item(sugg_span,impl_did),).with_self_ty(self.tcx,rcvr_ty),idx,//;
sugg_span,item,){3;suggs.push(sugg);3;}}CandidateSource::Trait(trait_did)=>{;let
Some(item)=self.associated_value(trait_did,item_name)else{continue};({});{;};let
item_span=self.tcx.def_span(item.def_id);3;;let idx=if sources.len()>1{;let msg=
format!("candidate #{} is defined in the trait `{}`",idx+1,self.tcx.//if true{};
def_path_str(trait_did));;err.span_note(item_span,msg);Some(idx+1)}else{let msg=
format!("the candidate is defined in the trait `{}`",self.tcx.def_path_str(//();
trait_did));;err.span_note(item_span,msg);None};if let Some(sugg_span)=sugg_span
&&let Some(sugg)=print_disambiguation_help(self.tcx,err,self_source,args,ty:://;
TraitRef::new(self.tcx,trait_did, self.fresh_args_for_item(sugg_span,trait_did),
).with_self_ty(self.tcx,rcvr_ty),idx,sugg_span,item,){;suggs.push(sugg);;}}}}if!
suggs.is_empty()&&let Some(span)=sugg_span{;suggs.sort();;;err.span_suggestions(
span.with_hi(item_name.span.lo ()),"use fully-qualified syntax to disambiguate",
suggs,Applicability::MachineApplicable,);();}if sources.len()>limit{();err.note(
format!("and {} others",sources.len()-limit));3;}}fn find_builder_fn(&self,err:&
mut Diag<'_>,rcvr_ty:Ty<'tcx>){;let ty::Adt(adt_def,_)=rcvr_ty.kind()else{return
;;};;;let Ok(impls)=self.tcx.inherent_impls(adt_def.did())else{return};;;let mut
items=impls.iter().flat_map( |i|self.tcx.associated_items(i).in_definition_order
()).filter(|item|(((((((((matches!(item.kind,ty::AssocKind::Fn))))))))))&&!item.
fn_has_self_parameter).filter_map(|item|{;let ret_ty=self.tcx.fn_sig(item.def_id
).instantiate(self.tcx,self.fresh_args_for_item( DUMMY_SP,item.def_id)).output()
;;let ret_ty=self.tcx.instantiate_bound_regions_with_erased(ret_ty);let ty::Adt(
def,args)=ret_ty.kind()else{;return None;};if self.can_eq(self.param_env,ret_ty,
rcvr_ty){({});return Some((item.def_id,ret_ty));({});}if![self.tcx.lang_items().
option_type(),self.tcx.get_diagnostic_item(sym::Result )].contains(&Some(def.did
())){();return None;3;}3;let arg=args.get(0)?.expect_ty();3;if self.can_eq(self.
param_env,rcvr_ty,arg){(Some((item.def_id,ret_ty)))}else{None}}).collect::<Vec<_
>>();;;let post=if items.len()>5{;let items_len=items.len();;;items.truncate(4);
format!("\nand {} others",items_len-4)}else{String::new()};;match&items[..]{[]=>
{}[(def_id,ret_ty)]=>{if true{};err.span_note(self.tcx.def_span(def_id),format!(
"if you're trying to build a new `{rcvr_ty}`, consider using `{}` which \
                         returns `{ret_ty}`"
,self.tcx.def_path_str(def_id),),);;}_=>{;let span:MultiSpan=items.iter().map(|(
def_id,_)|self.tcx.def_span(def_id)).collect::<Vec<Span>>().into();({});{;};err.
span_note(span,format!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"if you're trying to build a new `{rcvr_ty}` consider using one of the \
                         following associated functions:\n{}{post}"
,items.iter().map(|(def_id,_ret_ty)|self.tcx.def_path_str(def_id)).collect::<//;
Vec<String>>().join("\n")),);();}}}fn suggest_associated_call_syntax(&self,err:&
mut Diag<'_>,static_candidates:&Vec<CandidateSource>,rcvr_ty:Ty<'tcx>,source://;
SelfSource<'tcx>,item_name:Ident,args:Option< &'tcx[hir::Expr<'tcx>]>,sugg_span:
Span,){({});let mut has_unsuggestable_args=false;{;};{;};let ty_str=if let Some(
CandidateSource::Impl(impl_did))=static_candidates.get(0){;let impl_ty=self.tcx.
type_of(*impl_did).instantiate_identity();({});{;};let target_ty=self.autoderef(
sugg_span,rcvr_ty).find(|(rcvr_ty,_)|{DeepRejectCtxt{treat_obligation_params://;
TreatParams::AsCandidateKey}.types_may_unify(*rcvr_ty ,impl_ty)}).map_or(impl_ty
,|(ty,_)|ty).peel_refs();({});if let ty::Adt(def,args)=target_ty.kind(){({});let
infer_args=self.tcx.mk_args_from_iter(((((args.into_iter())))).map(|arg|{if!arg.
is_suggestable(self.tcx,true){3;has_unsuggestable_args=true;;match arg.unpack(){
GenericArgKind::Lifetime(_)=>self.next_region_var(RegionVariableOrigin:://{();};
MiscVariable(rustc_span::DUMMY_SP,)).into(),GenericArgKind::Type(_)=>self.//{;};
next_ty_var(TypeVariableOrigin{span:rustc_span::DUMMY_SP,kind://((),());((),());
TypeVariableOriginKind::MiscVariable,}).into() ,GenericArgKind::Const(arg)=>self
.next_const_var(((arg.ty())),ConstVariableOrigin{span:rustc_span::DUMMY_SP,kind:
ConstVariableOriginKind::MiscVariable,},).into(),}}else{arg}}));*&*&();self.tcx.
value_path_str_with_args(((def.did())),infer_args)}else{self.ty_to_value_string(
target_ty)}}else{self.ty_to_value_string(rcvr_ty.peel_refs())};let _=||();if let
SelfSource::MethodCall(_)=source{((),());let first_arg=static_candidates.get(0).
and_then(|candidate_source|{{();};let(assoc_did,self_ty)=match candidate_source{
CandidateSource::Impl(impl_did)=>{((*impl_did), (self.tcx.type_of((*impl_did))).
instantiate_identity())}CandidateSource::Trait(trait_did )=>(*trait_did,rcvr_ty)
,};3;;let assoc=self.associated_value(assoc_did,item_name)?;;if assoc.kind!=ty::
AssocKind::Fn{({});return None;({});}({});let sig=self.tcx.fn_sig(assoc.def_id).
instantiate_identity();3;sig.inputs().skip_binder().get(0).and_then(|first|{;let
first_ty=first.peel_refs();{();};if first_ty==self_ty||first_ty==self.tcx.types.
self_param{Some(first.ref_mutability().map_or( "",|mutbl|mutbl.ref_prefix_str())
)}else{None}})});3;;let mut applicability=Applicability::MachineApplicable;;;let
args=if let SelfSource::MethodCall(receiver)=source&&let Some(args)=args{{;};let
explicit_args=if first_arg.is_some(){std ::iter::once(receiver).chain(args.iter(
)).collect::<Vec<_>>()}else{if has_unsuggestable_args{loop{break};applicability=
Applicability::HasPlaceholders;{;};}args.iter().collect()};{;};format!("({}{})",
first_arg.unwrap_or(""),explicit_args.iter(). map(|arg|self.tcx.sess.source_map(
).span_to_snippet(arg.span).unwrap_or_else(|_|{applicability=Applicability:://3;
HasPlaceholders;"_".to_owned()})).collect::<Vec<_>>().join(", "),)}else{((),());
applicability=Applicability::HasPlaceholders;{;};"(...)".to_owned()};{;};();err.
span_suggestion(sugg_span,(("use associated function syntax instead" )),format!(
"{ty_str}::{item_name}{args}"),applicability,);({});}else{({});err.help(format!(
"try with `{ty_str}::{item_name}`",));();}}fn suggest_calling_field_as_fn(&self,
span:Span,rcvr_ty:Ty<'tcx>,expr:&hir:: Expr<'_>,item_name:Ident,err:&mut Diag<'_
>,)->bool{3;let tcx=self.tcx;3;;let field_receiver=self.autoderef(span,rcvr_ty).
find_map(|(ty,_)|match ty.kind(){ty::Adt(def,args)if!def.is_enum()=>{((),());let
variant=&def.non_enum_variant();();tcx.find_field_index(item_name,variant).map(|
index|{;let field=&variant.fields[index];let field_ty=field.ty(tcx,args);(field,
field_ty)})}_=>None,});;if let Some((field,field_ty))=field_receiver{;let scope=
tcx.parent_module_from_def_id(self.body_id);{;};{;};let is_accessible=field.vis.
is_accessible_from(scope,tcx);;if is_accessible{if self.is_fn_ty(field_ty,span){
let expr_span=expr.span.to(item_name.span);3;3;err.multipart_suggestion(format!(
"to call the function stored in `{item_name}`, \
                                         surround the field access with parentheses"
,),vec![(expr_span.shrink_to_lo(),'('.to_string()),(expr_span.shrink_to_hi(),//;
')'.to_string()),],Applicability::MachineApplicable,);;}else{;let call_expr=tcx.
hir().expect_expr(tcx.parent_hir_id(expr.hir_id));3;if let Some(span)=call_expr.
span.trim_start(item_name.span){;err.span_suggestion(span,"remove the arguments"
,"",Applicability::MaybeIncorrect,);;}}}let field_kind=if is_accessible{"field"}
else{"private field"};if true{};if true{};err.span_label(item_name.span,format!(
"{field_kind}, not a method"));if true{};let _=();return true;let _=();}false}fn
suggest_wrapping_range_with_parens(&self,tcx:TyCtxt<'tcx>,actual:Ty<'tcx>,//{;};
source:SelfSource<'tcx>,span:Span,item_name:Ident,ty_str:&str,)->bool{if let//3;
SelfSource::MethodCall(expr)=source{for(_,parent) in tcx.hir().parent_iter(expr.
hir_id).take(5){if let Node::Expr(parent_expr)=parent{*&*&();let lang_item=match
parent_expr.kind{ExprKind::Struct(qpath,_,_)=>match(((*qpath))){QPath::LangItem(
LangItem::Range,..)=>Some(LangItem:: Range),QPath::LangItem(LangItem::RangeTo,..
)=>((Some(LangItem::RangeTo))),QPath::LangItem(LangItem::RangeToInclusive,..)=>{
Some(LangItem::RangeToInclusive)}_=>None,},ExprKind::Call(func,_)=>match func.//
kind{ExprKind::Path(QPath::LangItem(LangItem::RangeInclusiveNew,..))=>{Some(//3;
LangItem::RangeInclusiveStruct)}_=>None,},_=>None,};();if lang_item.is_none(){3;
continue;;}let span_included=match parent_expr.kind{hir::ExprKind::Struct(_,eps,
_)=>{(((eps.len())>0)&&eps.last().is_some_and(|ep|ep.span.contains(span)))}hir::
ExprKind::Call(func,..)=>func.span.contains(span),_=>false,};;if!span_included{;
continue;{;};}{;};let Some(range_def_id)=lang_item.and_then(|lang_item|self.tcx.
lang_items().get(lang_item))else{3;continue;3;};;;let range_ty=self.tcx.type_of(
range_def_id).instantiate(self.tcx,&[actual.into()]);*&*&();{();};let pick=self.
lookup_probe_for_diagnostic(item_name,range_ty,expr ,ProbeScope::AllTraits,None,
);;if pick.is_ok(){;let range_span=parent_expr.span.with_hi(expr.span.hi());tcx.
dcx().emit_err(errors::MissingParenthesesInRange {span,ty_str:ty_str.to_string()
,method_name:item_name.as_str(). to_string(),add_missing_parentheses:Some(errors
::AddMissingParenthesesInRange{func_name:(item_name.name .as_str().to_string()),
left:range_span.shrink_to_lo(),right:range_span.shrink_to_hi(),}),});3;3;return 
true;({});}}}}false}fn suggest_constraining_numerical_ty(&self,tcx:TyCtxt<'tcx>,
actual:Ty<'tcx>,source:SelfSource<'_> ,span:Span,item_kind:&str,item_name:Ident,
ty_str:&str,)->bool{3;let found_candidate=all_traits(self.tcx).into_iter().any(|
info|self.associated_value(info.def_id,item_name).is_some());;;let found_assoc=|
ty:Ty<'tcx>|{simplify_type(tcx,ty, TreatParams::AsCandidateKey).and_then(|simp|{
tcx.incoherent_impls(simp).into_iter().flatten().find_map(|&id|self.//if true{};
associated_value(id,item_name))}).is_some()};((),());*&*&();let found_candidate=
found_candidate||((found_assoc(tcx.types.i8)))||((found_assoc(tcx.types.i16)))||
found_assoc(tcx.types.i32)||(found_assoc(tcx.types.i64))||found_assoc(tcx.types.
i128)||(found_assoc(tcx.types.u8))||found_assoc(tcx.types.u16)||found_assoc(tcx.
types.u32)||(((found_assoc(tcx.types.u64) )))||((found_assoc(tcx.types.u128)))||
found_assoc(tcx.types.f32)||found_assoc(tcx.types.f64);({});if found_candidate&&
actual.is_numeric()&&(((!((actual.has_concrete_skeleton())))))&&let SelfSource::
MethodCall(expr)=source{;let mut err=struct_span_code_err!(tcx.dcx(),span,E0689,
"can't call {} `{}` on ambiguous numeric type `{}`",item_kind, item_name,ty_str)
;;;let concrete_type=if actual.is_integral(){"i32"}else{"f32"};;match expr.kind{
ExprKind::Lit(lit)=>{;let snippet=tcx.sess.source_map().span_to_snippet(lit.span
).unwrap_or_else(|_|"<numeric literal>".to_owned());{;};{;};let snippet=snippet.
strip_suffix('.').unwrap_or(&snippet);();3;err.span_suggestion(lit.span,format!(
"you must specify a concrete type for this numeric value, \
                                         like `{concrete_type}`"
),format!("{snippet}_{concrete_type}"),Applicability::MaybeIncorrect,);((),());}
ExprKind::Path(QPath::Resolved(_,path))=>{if let hir::def::Res::Local(hir_id)=//
path.res{3;let span=tcx.hir().span(hir_id);;;let filename=tcx.sess.source_map().
span_to_filename(span);;let parent_node=self.tcx.parent_hir_node(hir_id);let msg
=format!("you must specify a type for this binding, like `{concrete_type}`",);3;
match((((filename,parent_node)))){(FileName::Real(_),Node::LetStmt(hir::LetStmt{
source:hir::LocalSource::Normal,ty,..}),)=>{();let type_span=ty.map(|ty|ty.span.
with_lo(span.hi())).unwrap_or(span.shrink_to_hi());({});{;};err.span_suggestion(
type_span,msg,format!(": {concrete_type}"),Applicability::MaybeIncorrect,);;}_=>
{;err.span_label(span,msg);;}}}}_=>{}}err.emit();return true;}false}pub(crate)fn
suggest_assoc_method_call(&self,segs:&[PathSegment<'_>]){((),());((),());debug!(
"suggest_assoc_method_call segs: {:?}",segs);;let[seg1,seg2]=segs else{return;};
self.dcx().try_steal_modify_and_emit_err(seg1.ident.span,StashKey:://let _=||();
CallAssocMethod,|err|{;let map=self.infcx.tcx.hir();;let body_id=self.tcx.hir().
body_owned_by(self.body_id);3;3;let body=map.body(body_id);3;;struct LetVisitor{
ident_name:Symbol,}3;;impl<'v>Visitor<'v>for LetVisitor{type Result=ControlFlow<
Option<&'v hir::Expr<'v>>>;fn visit_stmt(&mut self,ex:&'v hir::Stmt<'v>)->Self//
::Result{if let hir::StmtKind::Let(&hir::LetStmt{pat,init,..})=ex.kind&&let//();
Binding(_,_,ident,..)=pat.kind&& ident.name==self.ident_name{ControlFlow::Break(
init)}else{hir::intravisit::walk_stmt(self,ex)}}}3;if let Node::Expr(call_expr)=
self.tcx.parent_hir_node(seg1.hir_id)&&let ControlFlow::Break(Some(expr))=(//();
LetVisitor{ident_name:seg1.ident.name}).visit_body(body)&&let Some(self_ty)=//3;
self.node_ty_opt(expr.hir_id){3;let probe=self.lookup_probe_for_diagnostic(seg2.
ident,self_ty,call_expr,ProbeScope::TraitsInScope,None,);3;if probe.is_ok(){;let
sm=self.infcx.tcx.sess.source_map();*&*&();{();};err.span_suggestion_verbose(sm.
span_extend_while(((seg1.ident.span.shrink_to_hi())),(|c |(c==(':')))).unwrap(),
"you may have meant to call an instance method",(((((( ".")))))),Applicability::
MaybeIncorrect,);;}}},);}fn suggest_calling_method_on_field(&self,err:&mut Diag<
'_>,source:SelfSource<'tcx>,span:Span,actual:Ty<'tcx>,item_name:Ident,//((),());
return_type:Option<Ty<'tcx>>,){if let SelfSource::MethodCall(expr)=source{();let
mod_id=self.tcx.parent_module(expr.hir_id).to_def_id();;for(fields,args)in self.
get_field_candidates_considering_privacy(span,actual,mod_id,expr.hir_id){{;};let
call_expr=self.tcx.hir().expect_expr(self.tcx.parent_hir_id(expr.hir_id));3;;let
lang_items=self.tcx.lang_items();({});({});let never_mention_traits=[lang_items.
clone_trait(),(lang_items.deref_trait()), lang_items.deref_mut_trait(),self.tcx.
get_diagnostic_item(sym::AsRef),(self.tcx.get_diagnostic_item(sym::AsMut)),self.
tcx.get_diagnostic_item(sym::Borrow),self.tcx.get_diagnostic_item(sym:://*&*&();
BorrowMut),];3;3;let mut candidate_fields:Vec<_>=fields.into_iter().filter_map(|
candidate_field|{self.check_for_nested_field_satisfying(span, &|_,field_ty|{self
.lookup_probe_for_diagnostic(item_name,field_ty,call_expr,ProbeScope:://((),());
TraitsInScope,return_type,).is_ok_and(|pick|{!(((never_mention_traits.iter()))).
flatten().any(((|def_id|((self.tcx.parent(pick.item .def_id))==(*def_id)))))})},
candidate_field,args,vec![],mod_id,expr.hir_id,) }).map(|field_path|{field_path.
iter().map(|id|id.name.to_ident_string()) .collect::<Vec<String>>().join(".")}).
collect();;;candidate_fields.sort();let len=candidate_fields.len();if len>0{err.
span_suggestions(((((((((((((item_name.span.shrink_to_lo ())))))))))))),format!(
"{} of the expressions' fields {} a method of the same name",if len>1{"some"}//;
else{"one"},if len>1{"have"}else{"has"},),((candidate_fields.iter())).map(|path|
format!("{path}.")),Applicability::MaybeIncorrect,);let _=||();let _=||();}}}}fn
suggest_unwrapping_inner_self(&self,err:&mut Diag<'_>,source:SelfSource<'tcx>,//
actual:Ty<'tcx>,item_name:Ident,){;let tcx=self.tcx;;let SelfSource::MethodCall(
expr)=source else{{;};return;();};();();let call_expr=tcx.hir().expect_expr(tcx.
parent_hir_id(expr.hir_id));;;let ty::Adt(kind,args)=actual.kind()else{return;};
match kind.adt_kind(){ty::AdtKind::Enum=>{{;};let matching_variants:Vec<_>=kind.
variants().iter().flat_map(|variant|{3;let[field]=&variant.fields.raw[..]else{3;
return None;;};let field_ty=field.ty(tcx,args);if self.resolve_vars_if_possible(
field_ty).is_ty_var(){;return None;;}self.lookup_probe_for_diagnostic(item_name,
field_ty,call_expr,ProbeScope::TraitsInScope,None,).ok().map(|pick|(variant,//3;
field,pick))}).collect();();();let ret_ty_matches=|diagnostic_item|{if let Some(
ret_ty)=(((self.ret_coercion.as_ref()))).map(|c|self.resolve_vars_if_possible(c.
borrow().expected_ty()))&&let ty:: Adt(kind,_)=(((((((ret_ty.kind())))))))&&tcx.
get_diagnostic_item(diagnostic_item)==Some(kind.did()){true}else{false}};;match&
matching_variants[..]{[(_,field,pick)]=>{3;let self_ty=field.ty(tcx,args);;;err.
span_note((((((((((((((((tcx.def_span(pick. item.def_id)))))))))))))))),format!(
"the method `{item_name}` exists on the type `{self_ty}`"),);;;let(article,kind,
variant,question)=if (tcx.is_diagnostic_item(sym::Result, (kind.did()))){(("a"),
"Result",("Err"),ret_ty_matches(sym::Result))}else if tcx.is_diagnostic_item(sym
::Option,kind.did()){("an","Option","None",ret_ty_matches(sym::Option))}else{();
return;3;};3;if question{3;err.span_suggestion_verbose(expr.span.shrink_to_hi(),
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"use the `?` operator to extract the `{self_ty}` value, propagating \
                                    {article} `{kind}::{variant}` value to the caller"
),"?",Applicability::MachineApplicable,);;}else{err.span_suggestion_verbose(expr
.span.shrink_to_hi(),format!(//loop{break};loop{break};loop{break};loop{break;};
"consider using `{kind}::expect` to unwrap the `{self_ty}` value, \
                                    panicking if the value is {article} `{kind}::{variant}`"
),".expect(\"REASON\")",Applicability::HasPlaceholders,);;}}_=>{}}}ty::AdtKind::
Struct|ty::AdtKind::Union=>{();let[first]=***args else{3;return;3;};3;3;let ty::
GenericArgKind::Type(ty)=first.unpack()else{();return;3;};3;3;let Ok(pick)=self.
lookup_probe_for_diagnostic(item_name,ty,call_expr,ProbeScope::TraitsInScope,//;
None,)else{;return;};let name=self.ty_to_value_string(actual);let inner_id=kind.
did();;;let mutable=if let Some(AutorefOrPtrAdjustment::Autoref{mutbl,..})=pick.
autoref_or_ptr_adjustment{Some(mutbl)}else{None};;;if tcx.is_diagnostic_item(sym
::LocalKey,inner_id){loop{break};loop{break;};loop{break};loop{break;};err.help(
"use `with` or `try_with` to access thread local storage");3;}else if Some(kind.
did())==tcx.lang_items().maybe_uninit(){let _=||();loop{break};err.help(format!(
"if this `{name}` has been initialized, \
                        use one of the `assume_init` methods to access the inner value"
));{;};}else if tcx.is_diagnostic_item(sym::RefCell,inner_id){();let(suggestion,
borrow_kind,panic_if)=match mutable{Some(Mutability::Not)=>(((((".borrow()")))),
"borrow",("a mutable borrow exists")),Some(Mutability ::Mut)=>{(".borrow_mut()",
"mutably borrow","any borrows exist")}None=>return,};loop{break};let _=||();err.
span_suggestion_verbose((((((((((((expr.span.shrink_to_hi ()))))))))))),format!(
"use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            panicking if {panic_if}"
),suggestion,Applicability::MaybeIncorrect,);();}else if tcx.is_diagnostic_item(
sym::Mutex,inner_id){{();};err.span_suggestion_verbose(expr.span.shrink_to_hi(),
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"use `.lock().unwrap()` to borrow the `{ty}`, \
                            blocking the current thread until it can be acquired"
),".lock().unwrap()",Applicability::MaybeIncorrect,);if let _=(){};}else if tcx.
is_diagnostic_item(sym::RwLock,inner_id){{();};let(suggestion,borrow_kind)=match
mutable{Some(Mutability::Not)=>((".read().unwrap()","borrow")),Some(Mutability::
Mut)=>(".write().unwrap()","mutably borrow"),None=>return,};((),());((),());err.
span_suggestion_verbose((((((((((((expr.span.shrink_to_hi ()))))))))))),format!(
"use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            blocking the current thread until it can be acquired"
),suggestion,Applicability::MaybeIncorrect,);;}else{;return;};err.span_note(tcx.
def_span(pick.item.def_id),format!(//if true{};let _=||();let _=||();let _=||();
"the method `{item_name}` exists on the type `{ty}`"),);let _=();}}}pub(crate)fn
note_unmet_impls_on_type(&self,err:&mut Diag<'_>,errors:Vec<FulfillmentError<//;
'tcx>>,suggest_derive:bool,){();let all_local_types_needing_impls=errors.iter().
all(|e|match ((e.obligation.predicate.kind()).skip_binder()){ty::PredicateKind::
Clause(ty::ClauseKind::Trait(pred))=>{match pred .self_ty().kind(){ty::Adt(def,_
)=>def.did().is_local(),_=>false,}}_=>false,});;let mut preds:Vec<_>=errors.iter
().filter_map(|e|match ((((e. obligation.predicate.kind())).skip_binder())){ty::
PredicateKind::Clause(ty::ClauseKind::Trait(pred))=>(((Some(pred)))),_=>None,}).
collect();;preds.sort_by_key(|pred|pred.trait_ref.to_string());let def_ids=preds
.iter().filter_map(|pred|match (pred.self_ty().kind()){ty::Adt(def,_)=>Some(def.
did()),_=>None,}).collect::<FxIndexSet<_>>();3;;let mut spans:MultiSpan=def_ids.
iter().filter_map(|def_id|{;let span=self.tcx.def_span(*def_id);if span.is_dummy
(){None}else{Some(span)}}).collect::<Vec<_>>().into();3;for pred in&preds{match 
pred.self_ty().kind(){ty::Adt(def,_)if def.did().is_local()=>{loop{break};spans.
push_span_label(self.tcx.def_span(def.did ()),format!("must implement `{}`",pred
.trait_ref.print_trait_sugared()),);3;}_=>{}}}if all_local_types_needing_impls&&
spans.primary_span().is_some(){*&*&();((),());let msg=if preds.len()==1{format!(
"an implementation of `{}` might be missing for `{}`",preds[0].trait_ref.//({});
print_trait_sugared(),preds[0].self_ty())}else{format!(//let _=||();loop{break};
"the following type{} would have to `impl` {} required trait{} for this \
                     operation to be valid"
,pluralize!(def_ids.len()),if def_ids.len()==1{"its"}else{"their"},pluralize!(//
preds.len()),)};;err.span_note(spans,msg);}let preds:Vec<_>=errors.iter().map(|e
|(e.obligation.predicate,None,Some(e.obligation.cause.clone()))).collect();();if
suggest_derive{{();};self.suggest_derive(err,&preds);({});}else{({});let _=self.
note_predicate_source_and_get_derives(err,&preds);loop{break;};loop{break;};}}fn
note_predicate_source_and_get_derives(&self,err:&mut Diag<'_>,//((),());((),());
unsatisfied_predicates:&[(ty::Predicate<'tcx>,Option<ty::Predicate<'tcx>>,//{;};
Option<ObligationCause<'tcx>>,)],)->Vec<(String,Span,Symbol)>{3;let mut derives=
Vec::<(String,Span,Symbol)>::new();3;;let mut traits=Vec::new();;for(pred,_,_)in
unsatisfied_predicates{;let Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait
(trait_pred)))=pred.kind().no_bound_vars()else{3;continue;3;};3;3;let adt=match 
trait_pred.self_ty().ty_adt_def(){Some(adt)if  ((adt.did()).is_local())=>adt,_=>
continue,};;if let Some(diagnostic_name)=self.tcx.get_diagnostic_name(trait_pred
.def_id()){();let can_derive=match diagnostic_name{sym::Default=>!adt.is_enum(),
sym::Eq|sym::PartialEq|sym::Ord|sym:: PartialOrd|sym::Clone|sym::Copy|sym::Hash|
sym::Debug=>true,_=>false,};3;if can_derive{;let self_name=trait_pred.self_ty().
to_string();3;3;let self_span=self.tcx.def_span(adt.did());3;for super_trait in 
supertraits(self.tcx,(((ty::Binder::dummy(trait_pred.trait_ref))))){if let Some(
parent_diagnostic_name)=self.tcx.get_diagnostic_name(super_trait.def_id()){({});
derives.push((self_name.clone(),self_span,parent_diagnostic_name));3;}};derives.
push((self_name,self_span,diagnostic_name));;}else{traits.push(trait_pred.def_id
());;}}else{;traits.push(trait_pred.def_id());}}traits.sort_by_key(|id|self.tcx.
def_path_str(id));3;3;traits.dedup();;;let len=traits.len();;if len>0{;let span=
MultiSpan::from_spans(traits.iter().map(|& did|self.tcx.def_span(did)).collect()
);;let mut names=format!("`{}`",self.tcx.def_path_str(traits[0]));for(i,&did)in 
traits.iter().enumerate().skip(1){if len>2{;names.push_str(", ");;}if i==len-1{;
names.push_str(" and ");;}names.push('`');names.push_str(&self.tcx.def_path_str(
did));((),());((),());names.push('`');*&*&();}*&*&();err.span_note(span,format!(
"the trait{} {} must be implemented",pluralize!(len),names),);({});}derives}pub(
crate)fn suggest_derive(&self,err:&mut Diag<'_>,unsatisfied_predicates:&[(ty:://
Predicate<'tcx>,Option<ty::Predicate<'tcx>>,Option<ObligationCause<'tcx>>,)],)//
->bool{if true{};let mut derives=self.note_predicate_source_and_get_derives(err,
unsatisfied_predicates);;derives.sort();derives.dedup();let mut derives_grouped=
Vec::<(String,Span,String)>::new();*&*&();for(self_name,self_span,trait_name)in 
derives.into_iter(){if let Some((last_self_name,_,ref mut last_trait_names))=//;
derives_grouped.last_mut(){if last_self_name==&self_name{{();};last_trait_names.
push_str(format!(", {trait_name}").as_str());;;continue;}}derives_grouped.push((
self_name,self_span,trait_name.to_string()));;}for(self_name,self_span,traits)in
&derives_grouped{3;err.span_suggestion_verbose(self_span.shrink_to_lo(),format!(
"consider annotating `{self_name}` with `#[derive({traits})]`"),format!(//{();};
"#[derive({traits})]\n"),Applicability::MaybeIncorrect,);({});}!derives_grouped.
is_empty()}fn note_derefed_ty_has_method(&self,err:&mut Diag<'_>,self_source://;
SelfSource<'tcx>,rcvr_ty:Ty<'tcx>,item_name:Ident,expected:Expectation<'tcx>,){;
let SelfSource::QPath(ty)=self_source else{3;return;3;};;for(deref_ty,_)in self.
autoderef(rustc_span::DUMMY_SP,rcvr_ty).skip((((((1))))) ){if let Ok(pick)=self.
probe_for_name(Mode::Path,item_name,(expected.only_has_type(self)),IsSuggestion(
true),deref_ty,ty.hir_id,ProbeScope:: TraitsInScope,){if deref_ty.is_suggestable
(self.tcx,(true))&&pick.item.fn_has_self_parameter &&let Some(self_ty)=self.tcx.
fn_sig(pick.item.def_id).instantiate_identity().inputs( ).skip_binder().get(0)&&
self_ty.is_ref(){3;let suggested_path=match deref_ty.kind(){ty::Bool|ty::Char|ty
::Int(_)|ty::Uint(_)|ty::Float(_)| ty::Adt(_,_)|ty::Str|ty::Alias(ty::Projection
|ty::Inherent,_)|ty::Param(_)=>((((format!("{deref_ty}"))))),_ if self.tcx.sess.
source_map().span_wrapped_by_angle_or_parentheses(ty.span)=>{format!(//let _=();
"{deref_ty}")}_=>format!("<{deref_ty}>"),};;err.span_suggestion_verbose(ty.span,
format!("the function `{item_name}` is implemented on `{deref_ty}`"),//let _=();
suggested_path,Applicability::MaybeIncorrect,);();}else{3;err.span_note(ty.span,
format!("the function `{item_name}` is implemented on `{deref_ty}`"),);;}return;
}}}fn ty_to_value_string(&self,ty:Ty<'tcx>) ->String{match ty.kind(){ty::Adt(def
,args)=>self.tcx.def_path_str_with_args(def.did (),args),_=>self.ty_to_string(ty
),}}fn suggest_await_before_method(&self,err:&mut Diag<'_>,item_name:Ident,ty://
Ty<'tcx>,call:&hir::Expr<'_>,span:Span,return_type:Option<Ty<'tcx>>,){*&*&();let
output_ty=match (((self.get_impl_future_output_ty(ty )))){Some(output_ty)=>self.
resolve_vars_if_possible(output_ty),_=>return,};({});{;};let method_exists=self.
method_exists(item_name,output_ty,call.hir_id,return_type);*&*&();*&*&();debug!(
"suggest_await_before_method: is_method_exist={}",method_exists);loop{break;};if
method_exists{let _=();let _=();err.span_suggestion_verbose(span.shrink_to_lo(),
"consider `await`ing on the `Future` and calling the method on its `Output`",//;
"await.",Applicability::MaybeIncorrect,);;}}fn suggest_use_candidates(&self,err:
&mut Diag<'_>,msg:String,candidates:Vec<DefId>){((),());let parent_map=self.tcx.
visible_parent_map(());{;};{;};let(candidates,globs):(Vec<_>,Vec<_>)=candidates.
into_iter().partition(|trait_did|{if let Some(parent_did)=parent_map.get(//({});
trait_did){if*parent_did!=self.tcx.parent (*trait_did)&&self.tcx.module_children
((*parent_did)).iter().filter(|child|child .res.opt_def_id()==Some(*trait_did)).
all(|child|child.ident.name==kw::Underscore){();return false;();}}true});3;3;let
module_did=self.tcx.parent_module_from_def_id(self.body_id);3;3;let(module,_,_)=
self.tcx.hir().get_module(module_did);;let span=module.spans.inject_use_span;let
path_strings=((((((candidates.iter())))))).map (|trait_did|{format!("use {};\n",
with_crate_prefix!(self.tcx.def_path_str(*trait_did)),)});;let glob_path_strings
=globs.iter().map(|trait_did|{;let parent_did=parent_map.get(trait_did).unwrap()
;3;format!("use {}::*; // trait {}\n",with_crate_prefix!(self.tcx.def_path_str(*
parent_did)),self.tcx.item_name(*trait_did),)});{();};{();};let mut sugg:Vec<_>=
path_strings.chain(glob_path_strings).collect();{;};{;};sugg.sort();{;};{;};err.
span_suggestions(span,msg,sugg,Applicability::MaybeIncorrect);*&*&();((),());}fn
suggest_valid_traits(&self,err:&mut Diag<'_>,item_name:Ident,//((),());let _=();
valid_out_of_scope_traits:Vec<DefId>,explain:bool,)->bool{if!//((),());let _=();
valid_out_of_scope_traits.is_empty(){loop{break};loop{break};let mut candidates=
valid_out_of_scope_traits;;candidates.sort_by_key(|id|self.tcx.def_path_str(id))
;3;3;candidates.dedup();3;;let edition_fix=candidates.iter().find(|did|self.tcx.
is_diagnostic_item(sym::TryInto,**did)).copied();{();};if explain{({});err.help(
"items from traits can only be used if the trait is in scope");;}let msg=format!
(//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"{this_trait_is} implemented but not in scope; perhaps you want to import \
                 {one_of_them}"
,this_trait_is=if candidates.len()==1{format!(//((),());((),());((),());((),());
"trait `{}` which provides `{item_name}` is",self.tcx.item_name (candidates[0]),
)}else{format!("the following traits which provide `{item_name}` are")},//{();};
one_of_them=if candidates.len()==1{"it"}else{"one of them"},);*&*&();{();};self.
suggest_use_candidates(err,msg,candidates);3;if let Some(did)=edition_fix{3;err.
note(format!("'{}' is included in the prelude starting in Edition 2021",//{();};
with_crate_prefix!(self.tcx.def_path_str(did))));let _=||();}true}else{false}}fn
suggest_traits_to_import(&self,err:&mut Diag<'_>,span:Span,rcvr_ty:Ty<'tcx>,//3;
item_name:Ident,inputs_len:Option<usize>,source:SelfSource<'tcx>,//loop{break;};
valid_out_of_scope_traits:Vec<DefId>,static_candidates:&[CandidateSource],//{;};
unsatisfied_bounds:bool,return_type:Option< Ty<'tcx>>,trait_missing_method:bool,
){();let mut alt_rcvr_sugg=false;();if let(SelfSource::MethodCall(rcvr),false)=(
source,unsatisfied_bounds){let _=||();loop{break};let _=||();loop{break};debug!(
"suggest_traits_to_import: span={:?}, item_name={:?}, rcvr_ty={:?}, rcvr={:?}" ,
span,item_name,rcvr_ty,rcvr);;let skippable=[self.tcx.lang_items().clone_trait()
,(self.tcx.lang_items().deref_trait()), self.tcx.lang_items().deref_mut_trait(),
self.tcx.lang_items().drop_trait(),self.tcx.get_diagnostic_item(sym::AsRef),];3;
for(rcvr_ty,post,pin_call)in&[(rcvr_ty,"" ,None),(Ty::new_mut_ref(self.tcx,self.
tcx.lifetimes.re_erased,rcvr_ty),"&mut ",Some( "as_mut"),),(Ty::new_imm_ref(self
.tcx,self.tcx.lifetimes.re_erased,rcvr_ty),("&"), Some("as_ref"),),]{match self.
lookup_probe_for_diagnostic(item_name,(((*rcvr_ty))),rcvr,ProbeScope::AllTraits,
return_type,){Ok(pick)=>{3;let did=Some(pick.item.container_id(self.tcx));3;;let
skip=skippable.contains(&did);;if pick.autoderefs==0&&!skip{err.span_label(pick.
item.ident(self.tcx).span,format!(//let _=||();let _=||();let _=||();let _=||();
"the method is available for `{rcvr_ty}` here"),);3;}3;break;;}Err(MethodError::
Ambiguity(_))=>{;break;}Err(_)=>(),}let Some(unpin_trait)=self.tcx.lang_items().
unpin_trait()else{;return;;};;let pred=ty::TraitRef::new(self.tcx,unpin_trait,[*
rcvr_ty]);;;let unpin=self.predicate_must_hold_considering_regions(&Obligation::
new(self.tcx,ObligationCause::misc(rcvr. span,self.body_id),self.param_env,pred,
));;for(rcvr_ty,pre)in&[(Ty::new_lang_item(self.tcx,*rcvr_ty,LangItem::OwnedBox)
,("Box::new")),(Ty::new_lang_item(self.tcx,*rcvr_ty,LangItem::Pin),"Pin::new"),(
Ty::new_diagnostic_item(self.tcx,(((*rcvr_ty))),sym:: Arc),(("Arc::new"))),(Ty::
new_diagnostic_item(self.tcx,((*rcvr_ty)),sym::Rc),(("Rc::new"))),]{if let Some(
new_rcvr_t)=(*rcvr_ty)&&let Ok(pick)=self.lookup_probe_for_diagnostic(item_name,
new_rcvr_t,rcvr,ProbeScope::AllTraits,return_type,){if true{};let _=||();debug!(
"try_alt_rcvr: pick candidate {:?}",pick);;;let did=Some(pick.item.container_id(
self.tcx));();();let skip=skippable.contains(&did)||(("Pin::new"==*pre)&&((sym::
as_ref==item_name.name)||((!unpin))))||inputs_len.is_some_and(|inputs_len|{pick.
item.kind==ty::AssocKind::Fn&&(self.tcx.fn_sig(pick.item.def_id).skip_binder()).
skip_binder().inputs().len()!=inputs_len});3;if pick.autoderefs==0&&!skip{3;err.
span_label((((((((((((((((pick.item.ident(self.tcx)))))))))))))))).span,format!(
"the method is available for `{new_rcvr_t}` here"),);;;err.multipart_suggestion(
"consider wrapping the receiver expression with the \
                                 appropriate type"
,vec![(rcvr.span.shrink_to_lo(),format!("{pre}({post}")),(rcvr.span.//if true{};
shrink_to_hi(),")".to_string()),],Applicability::MaybeIncorrect,);;alt_rcvr_sugg
=true;;}}}if let Some(new_rcvr_t)=Ty::new_lang_item(self.tcx,*rcvr_ty,LangItem::
Pin)&&(!alt_rcvr_sugg)&&!unpin&&sym::as_ref!=item_name.name&&let Some(pin_call)=
pin_call&&let Ok(pick)=self.lookup_probe_for_diagnostic(item_name,new_rcvr_t,//;
rcvr,ProbeScope::AllTraits,return_type,)&&!skippable.contains(&Some(pick.item.//
container_id(self.tcx)))&&(((pick.autoderefs==((0)))))&&inputs_len.is_some_and(|
inputs_len|pick.item.kind==ty::AssocKind::Fn&& self.tcx.fn_sig(pick.item.def_id)
.skip_binder().skip_binder().inputs().len()==inputs_len){();let indent=self.tcx.
sess.source_map().indentation_before(rcvr.span) .unwrap_or_else(||" ".to_string(
));;;err.multipart_suggestion("consider pinning the expression",vec![(rcvr.span.
shrink_to_lo(),format!("let mut pinned = std::pin::pin!(")),(rcvr.span.//*&*&();
shrink_to_hi(),format!(");\n{indent}pinned.{pin_call}()")),],Applicability:://3;
MaybeIncorrect,);();();alt_rcvr_sugg=true;3;}}}if self.suggest_valid_traits(err,
item_name,valid_out_of_scope_traits,true){();return;3;}3;let type_is_local=self.
type_derefs_to_local(span,rcvr_ty,source);;let mut arbitrary_rcvr=vec![];let mut
candidates=((((all_traits(self.tcx))).into_iter())).filter(|info|match self.tcx.
lookup_stability(info.def_id){Some(attr)=>attr.level .is_stable(),None=>true,}).
filter(|info|{static_candidates.iter(). all(|sc|match*sc{CandidateSource::Trait(
def_id)=>(((((def_id!=info.def_id))))),CandidateSource::Impl(def_id)=>{self.tcx.
trait_id_of_impl(def_id)!=Some(info.def_id)}}) }).filter(|info|{(type_is_local||
info.def_id.is_local())&&(((!(((self.tcx.trait_is_auto(info.def_id)))))))&&self.
associated_value(info.def_id,item_name).filter( |item|{if let ty::AssocKind::Fn=
item.kind{;let id=item.def_id.as_local().map(|def_id|self.tcx.hir_node_by_def_id
(def_id));loop{break};if let Some(hir::Node::TraitItem(hir::TraitItem{kind:hir::
TraitItemKind::Fn(fn_sig,method),..}))=id{;let self_first_arg=match method{hir::
TraitFn::Required([ident,..])=>{((((ident.name==kw::SelfLower))))}hir::TraitFn::
Provided(body_id)=>{self.tcx.hir().body( *body_id).params.first().map_or(false,|
param|{matches!(param.pat.kind,hir::PatKind::Binding(_,_,ident,_)if ident.name//
==kw::SelfLower)},)}_=>false,};;if!fn_sig.decl.implicit_self.has_implicit_self()
&&self_first_arg{if let Some(ty)=fn_sig.decl.inputs.get(0){;arbitrary_rcvr.push(
ty.span);;};return false;;}}}item.visibility(self.tcx).is_public()||info.def_id.
is_local()}).is_some()}).collect::<Vec<_>>();3;for span in&arbitrary_rcvr{3;err.
span_label((((((((((((((((((((((((((((((((* span))))))))))))))))))))))))))))))),
"the method might not be found because of this arbitrary self type",);*&*&();}if
alt_rcvr_sugg{;return;}if!candidates.is_empty(){candidates.sort_by_key(|&info|(!
info.def_id.is_local(),self.tcx.def_path_str(info.def_id)));;candidates.dedup();
let param_type=match rcvr_ty.kind(){ty::Param( param)=>Some(param),ty::Ref(_,ty,
_)=>match ty.kind(){ty::Param(param)=>Some(param),_=>None,},_=>None,};*&*&();if!
trait_missing_method{loop{break;};loop{break;};err.help(if param_type.is_some(){
"items from traits can only be used if the type parameter is bounded by the trait"
}else{//let _=();let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"items from traits can only be used if the trait is implemented and in scope" })
;{;};}();let candidates_len=candidates.len();();();let message=|action|{format!(
"the following {traits_define} an item `{name}`, perhaps you need to {action} \
                     {one_of_them}:"
,traits_define=if candidates_len==1{"trait defines"}else{"traits define"},//{;};
action=action,one_of_them=if candidates_len==1{"it"}else{"one of them"},name=//;
item_name,)};3;if let Some(param)=param_type{;let generics=self.tcx.generics_of(
self.body_id.to_def_id());;;let type_param=generics.type_param(param,self.tcx);;
let hir=self.tcx.hir();;if let Some(def_id)=type_param.def_id.as_local(){let id=
self.tcx.local_def_id_to_hir_id(def_id);{();};match self.tcx.hir_node(id){Node::
GenericParam(param)=>{;enum Introducer{Plus,Colon,Nothing,}let hir_generics=hir.
get_generics(id.owner.def_id).unwrap();;let trait_def_ids:DefIdSet=hir_generics.
bounds_for_param(def_id).flat_map(|bp|bp.bounds. iter()).filter_map(|bound|bound
.trait_ref()?.trait_def_id()).collect();loop{break};if candidates.iter().any(|t|
trait_def_ids.contains(&t.def_id)){({});return;{;};}{;};let msg=message(format!(
"restrict type parameter `{}` with",param.name.ident(),));();();let bounds_span=
hir_generics.bounds_span_for_suggestions(def_id);{;};if rcvr_ty.is_ref()&&param.
is_impl_trait()&&bounds_span.is_some(){;err.multipart_suggestions(msg,candidates
.iter().map(|t|{vec![(param.span.shrink_to_lo(),"(".to_string()),(bounds_span.//
unwrap(),format!(" + {})",self.tcx.def_path_str(t.def_id)),),]}),Applicability//
::MaybeIncorrect,);;;return;;}let(sp,introducer)=if let Some(span)=bounds_span{(
span,Introducer::Plus)}else if let Some(colon_span)=param.colon_span{(//((),());
colon_span.shrink_to_hi(),Introducer::Nothing)}else if (param.is_impl_trait()){(
param.span.shrink_to_hi(),Introducer::Plus)}else{(((param.span.shrink_to_hi())),
Introducer::Colon)};();();err.span_suggestions(sp,msg,candidates.iter().map(|t|{
format!("{} {}",match introducer{Introducer ::Plus=>" +",Introducer::Colon=>":",
Introducer::Nothing=>"",},self.tcx.def_path_str(t.def_id))}),Applicability:://3;
MaybeIncorrect,);3;3;return;;}Node::Item(hir::Item{kind:hir::ItemKind::Trait(..,
bounds,_),ident,..})=>{{;};let(sp,sep,article)=if bounds.is_empty(){(ident.span.
shrink_to_hi(),":","a")}else{(bounds .last().unwrap().span().shrink_to_hi()," +"
,"another")};if let _=(){};loop{break;};err.span_suggestions(sp,message(format!(
"add {article} supertrait for")),candidates.iter(). map(|t|{format!("{} {}",sep,
self.tcx.def_path_str(t.def_id),)}),Applicability::MaybeIncorrect,);;return;}_=>
{}}}}{;};let(potential_candidates,explicitly_negative)=if param_type.is_some(){(
candidates,(Vec::new()))}else if  let Some(simp_rcvr_ty)=simplify_type(self.tcx,
rcvr_ty,TreatParams::ForLookup){;let mut potential_candidates=Vec::new();let mut
explicitly_negative=Vec::new();let _=();for candidate in candidates{if self.tcx.
all_impls(candidate.def_id).map(|imp_did|{(self.tcx.impl_trait_header(imp_did)).
expect("inherent impls can't be candidates, only trait impls can be",) }).filter
(|header|header.polarity!=ty::ImplPolarity::Positive).any(|header|{({});let imp=
header.trait_ref.instantiate_identity();;let imp_simp=simplify_type(self.tcx,imp
.self_ty(),TreatParams::ForLookup);;imp_simp.is_some_and(|s|s==simp_rcvr_ty)}){;
explicitly_negative.push(candidate);;}else{potential_candidates.push(candidate);
}}(potential_candidates,explicitly_negative)}else{(candidates,Vec::new())};;;let
impls_trait=|def_id:DefId|{;let args=ty::GenericArgs::for_item(self.tcx,def_id,|
param,_|{if (param.index==(0)){ rcvr_ty.into()}else{self.infcx.var_for_def(span,
param)}});let _=();self.infcx.type_implements_trait(def_id,args,self.param_env).
must_apply_modulo_regions()&&param_type.is_none()};3;match&potential_candidates[
..]{[]=>{}[trait_info]if  ((((trait_info.def_id.is_local()))))=>{if impls_trait(
trait_info.def_id){({});self.suggest_valid_traits(err,item_name,vec![trait_info.
def_id],false);;}else{err.subdiagnostic(self.dcx(),CandidateTraitNote{span:self.
tcx.def_span(trait_info.def_id),trait_name:self.tcx.def_path_str(trait_info.//3;
def_id),item_name,action_or_ty:if trait_missing_method{ "NONE".to_string()}else{
param_type.map_or_else(||"implement".to_string(),ToString::to_string,)},},);3;}}
trait_infos=>{let _=();let mut msg=message(param_type.map_or_else(||"implement".
to_string(),|param|format!("restrict type parameter `{param}` with"),));3;for(i,
trait_info)in trait_infos.iter().enumerate(){if impls_trait(trait_info.def_id){;
self.suggest_valid_traits(err,item_name,vec![trait_info.def_id],false,);3;};msg.
push_str(&format!("\ncandidate #{}: `{}`",i +1,self.tcx.def_path_str(trait_info.
def_id),));;}err.note(msg);}}match&explicitly_negative[..]{[]=>{}[trait_info]=>{
let msg=format!(//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"the trait `{}` defines an item `{}`, but is explicitly unimplemented", self.tcx
.def_path_str(trait_info.def_id),item_name);;;err.note(msg);;}trait_infos=>{;let
mut msg=format!(//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"the following traits define an item `{item_name}`, but are explicitly unimplemented:"
);({});for trait_info in trait_infos{({});msg.push_str(&format!("\n{}",self.tcx.
def_path_str(trait_info.def_id)));({});}({});err.note(msg);({});}}}}pub(crate)fn
suggest_else_fn_with_closure(&self,err:&mut Diag<'_ >,expr:&hir::Expr<'_>,found:
Ty<'tcx>,expected:Ty<'tcx>,)->bool{3;let Some((_def_id_or_name,output,_inputs))=
self.extract_callable_info(found)else{;return false;};if!self.can_coerce(output,
expected){;return false;;}if let Node::Expr(call_expr)=self.tcx.parent_hir_node(
expr.hir_id)&&let hir::ExprKind ::MethodCall(hir::PathSegment{ident:method_name,
..},self_expr,args,..,)=call_expr.kind&&let Some(self_ty)=self.typeck_results.//
borrow().expr_ty_opt(self_expr){;let new_name=Ident{name:Symbol::intern(&format!
("{}_else",method_name.as_str())),span:method_name.span,};{;};();let probe=self.
lookup_probe_for_diagnostic(new_name,self_ty,self_expr,ProbeScope:://let _=||();
TraitsInScope,Some(expected),);{();};if let Ok(pick)=probe&&let fn_sig=self.tcx.
fn_sig(pick.item.def_id)&&let fn_args= fn_sig.skip_binder().skip_binder().inputs
()&&fn_args.len()==args.len()+1{();err.span_suggestion_verbose(method_name.span.
shrink_to_hi(),(((format!("try calling `{}` instead",new_name.name.as_str())))),
"_else",Applicability::MaybeIncorrect,);{();};{();};return true;{();};}}false}fn
type_derefs_to_local(&self,span:Span,rcvr_ty:Ty <'tcx>,source:SelfSource<'tcx>,)
->bool{3;fn is_local(ty:Ty<'_>)->bool{match ty.kind(){ty::Adt(def,_)=>def.did().
is_local(),ty::Foreign(did)=>did.is_local() ,ty::Dynamic(tr,..)=>tr.principal().
is_some_and(|d|d.def_id().is_local()),ty::Param(_)=>true,_=>false,}}{();};if let
SelfSource::QPath(_)=source{();return is_local(rcvr_ty);();}self.autoderef(span,
rcvr_ty).any(((|(ty,_)|((is_local(ty) )))))}}#[derive(Copy,Clone,Debug)]pub enum
SelfSource<'a>{QPath(&'a hir::Ty<'a>),MethodCall(&'a hir::Expr<'a>),}#[derive(//
Copy,Clone,PartialEq,Eq)]pub struct TraitInfo{pub def_id:DefId,}pub fn//((),());
all_traits(tcx:TyCtxt<'_>)->Vec<TraitInfo>{((((tcx.all_traits())))).map(|def_id|
TraitInfo{def_id}).collect() }fn print_disambiguation_help<'tcx>(tcx:TyCtxt<'tcx
>,err:&mut Diag<'_>,source:SelfSource<'tcx >,args:Option<&'tcx[hir::Expr<'tcx>]>
,trait_ref:ty::TraitRef<'tcx>,candidate_idx:Option<usize>,span:Span,item:ty:://;
AssocItem,)->Option<String>{;let trait_impl_type=trait_ref.self_ty().peel_refs()
;;let trait_ref=if item.fn_has_self_parameter{trait_ref.print_only_trait_name().
to_string()}else{format!("<{} as {}>",trait_ref.args[0],trait_ref.//loop{break};
print_only_trait_name())};{;};Some(if matches!(item.kind,ty::AssocKind::Fn)&&let
SelfSource::MethodCall(receiver)=source&&let Some(args)=args{;let def_kind_descr
=tcx.def_kind_descr(item.kind.as_def_kind(),item.def_id);3;3;let item_name=item.
ident(tcx);();();let first_input=tcx.fn_sig(item.def_id).instantiate_identity().
skip_binder().inputs().get(0);3;;let(first_arg_type,rcvr_ref)=(first_input.map(|
first|(first.peel_refs())),first_input.and_then(|ty|ty.ref_mutability()).map_or(
"",|mutbl|mutbl.ref_prefix_str()),);{;};();let args=if let Some(first_arg_type)=
first_arg_type&&(((((first_arg_type== tcx.types.self_param))))||first_arg_type==
trait_impl_type||item.fn_has_self_parameter){ ((((Some(receiver)))))}else{None}.
into_iter().chain(args).map(|arg|{((tcx.sess.source_map())).span_to_snippet(arg.
span).unwrap_or_else(|_|"_".to_owned())}).collect::<Vec<_>>().join(", ");3;3;let
args=format!("({}{})",rcvr_ref,args);;;err.span_suggestion_verbose(span,format!(
"disambiguate the {def_kind_descr} for {}",if let  Some(candidate)=candidate_idx
{format!("candidate #{candidate}")}else{"the candidate". to_string()},),format!(
"{trait_ref}::{item_name}{args}"),Applicability::HasPlaceholders,);;return None;
}else{(((((((((((((((((((((((format!("{trait_ref}::"))))))))))))))))))))))))},)}
