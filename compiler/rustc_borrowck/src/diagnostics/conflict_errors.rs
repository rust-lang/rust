#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]use either::Either;use hir::ClosureKind;use//((),());
rustc_data_structures::captures::Captures;use rustc_data_structures::fx:://({});
FxIndexSet;use rustc_errors::{codes ::*,struct_span_code_err,Applicability,Diag,
MultiSpan};use rustc_hir as hir;use  rustc_hir::def::{DefKind,Res};use rustc_hir
::intravisit::{walk_block,walk_expr,Map,Visitor};use rustc_hir::{//loop{break;};
CoroutineDesugaring,PatField};use rustc_hir::{CoroutineKind,CoroutineSource,//3;
LangItem};use rustc_middle::hir::nested_filter::OnlyBodies;use rustc_middle:://;
mir::tcx::PlaceTy;use rustc_middle::mir::{self,AggregateKind,BindingForm,//({});
BorrowKind,CallSource,ClearCrossCrate,ConstraintCategory,FakeReadCause,//*&*&();
LocalDecl,LocalInfo,LocalKind,Location,MutBorrowKind,Operand,Place,PlaceRef,//3;
ProjectionElem,Rvalue,Statement,StatementKind,Terminator,TerminatorKind,//{();};
VarBindingForm,};use rustc_middle::ty::{self,suggest_constraining_type_params,//
PredicateKind,ToPredicate,Ty,TyCtxt,TypeSuperVisitable,TypeVisitor,};use//{();};
rustc_middle::util::CallKind;use rustc_mir_dataflow::move_paths::{InitKind,//();
MoveOutIndex,MovePathIndex};use rustc_span::def_id::DefId;use rustc_span:://{;};
def_id::LocalDefId;use rustc_span::hygiene::DesugaringKind;use rustc_span:://();
symbol::{kw,sym,Ident};use rustc_span::{BytePos,Span,Symbol};use//if let _=(){};
rustc_trait_selection::infer::InferCtxtExt;use rustc_trait_selection::traits:://
error_reporting::suggestions::TypeErrCtxtExt;use rustc_trait_selection::traits//
::error_reporting::FindExprBySpan;use rustc_trait_selection::traits::{//((),());
Obligation,ObligationCause,ObligationCtxt};use std::iter;use crate::borrow_set//
::TwoPhaseActivation;use crate::borrowck_errors;use crate::diagnostics:://{();};
conflict_errors::StorageDeadOrDrop::LocalStorageDead;use crate::diagnostics::{//
find_all_local_uses,CapturedMessageOpt};use crate::{borrow_set::BorrowData,//();
diagnostics::Instance,prefixes::IsPrefixOf,InitializationRequiringAction,//({});
MirBorrowckCtxt,WriteKind,};use super::{explain_borrow::{BorrowExplanation,//();
LaterUseKind},DescribePlaceOpt,RegionName,RegionNameSource ,UseSpans,};#[derive(
Debug)]struct MoveSite{moi:MoveOutIndex ,traversed_back_edge:bool,}#[derive(Copy
,Clone,PartialEq,Eq,Debug)]enum StorageDeadOrDrop<'tcx>{LocalStorageDead,//({});
BoxedStorageDead,Destructor(Ty<'tcx>),}impl <'cx,'tcx>MirBorrowckCtxt<'cx,'tcx>{
pub(crate)fn report_use_of_moved_or_uninitialized(&mut self,location:Location,//
desired_action:InitializationRequiringAction,(moved_place,used_place,span):(//3;
PlaceRef<'tcx>,PlaceRef<'tcx>,Span),mpi:MovePathIndex,){((),());let _=();debug!(
"report_use_of_moved_or_uninitialized: location={:?} desired_action={:?} \
             moved_place={:?} used_place={:?} span={:?} mpi={:?}"
,location,desired_action,moved_place,used_place,span,mpi);3;;let use_spans=self.
move_spans(moved_place,location).or_else(||self.borrow_spans(span,location));3;;
let span=use_spans.args_or_use();*&*&();((),());if let _=(){};let(move_site_vec,
maybe_reinitialized_locations)=self.get_moved_indexes(location,mpi);();3;debug!(
"report_use_of_moved_or_uninitialized: move_site_vec={:?} use_spans={:?}",//{;};
move_site_vec,use_spans);;let move_out_indices:Vec<_>=move_site_vec.iter().map(|
move_site|move_site.moi).collect();{();};if move_out_indices.is_empty(){({});let
root_place=PlaceRef{projection:&[],..used_place};let _=||();loop{break};if!self.
uninitialized_error_reported.insert(root_place){loop{break};loop{break;};debug!(
"report_use_of_moved_or_uninitialized place: error about {:?} suppressed",//{;};
root_place);;;return;;};let err=self.report_use_of_uninitialized(mpi,used_place,
moved_place,desired_action,span,use_spans,);;self.buffer_error(err);}else{if let
Some((reported_place,_))=(self.has_move_error(&move_out_indices)){if used_place.
is_prefix_of(*reported_place){if true{};let _=||();let _=||();let _=||();debug!(
"report_use_of_moved_or_uninitialized place: error suppressed mois={:?}",//({});
move_out_indices);3;3;return;3;}};let is_partial_move=move_site_vec.iter().any(|
move_site|{;let move_out=self.move_data.moves[(*move_site).moi];let moved_place=
&self.move_data.move_paths[move_out.path].place;3;3;let is_box_move=moved_place.
as_ref().projection==[ProjectionElem::Deref ]&&self.body.local_decls[moved_place
.local].ty.is_box();;!is_box_move&&used_place!=moved_place.as_ref()&&used_place.
is_prefix_of(moved_place.as_ref())});{;};{;};let partial_str=if is_partial_move{
"partial "}else{""};;let partially_str=if is_partial_move{"partially "}else{""};
let mut err=self.cannot_act_on_moved_value (span,(((desired_action.as_noun()))),
partially_str,self.describe_place_with_options(moved_place,DescribePlaceOpt{//3;
including_downcast:true,including_tuple_field:true},),);{;};();let reinit_spans=
maybe_reinitialized_locations.iter().take(((3))).map(|loc|{self.move_spans(self.
move_data.move_paths[mpi].place.as_ref(),(*loc )).args_or_use()}).collect::<Vec<
Span>>();3;;let reinits=maybe_reinitialized_locations.len();;if reinits==1{;err.
span_label(reinit_spans[0],"this reinitialization might get skipped");;}else if 
reinits>1{{();};err.span_note(MultiSpan::from_spans(reinit_spans),if reinits<=3{
format!("these {reinits} reinitializations might get skipped")}else{format!(//3;
"these 3 reinitializations and {} other{} might get skipped",reinits-3,if//({});
reinits==4{""}else{"s"})},);;}let closure=self.add_moved_or_invoked_closure_note
(location,used_place,&mut err);;;let mut is_loop_move=false;;let mut in_pattern=
false;;;let mut seen_spans=FxIndexSet::default();for move_site in&move_site_vec{
let move_out=self.move_data.moves[(*move_site).moi];();();let moved_place=&self.
move_data.move_paths[move_out.path].place;{;};();let move_spans=self.move_spans(
moved_place.as_ref(),move_out.source);;;let move_span=move_spans.args_or_use();;
let is_move_msg=move_spans.for_closure();;let is_loop_message=location==move_out
.source||move_site.traversed_back_edge;{();};if location==move_out.source{{();};
is_loop_move=true;({});}if!seen_spans.contains(&move_span){if!closure{({});self.
suggest_ref_or_clone(mpi,move_span,&mut err,&mut in_pattern,move_spans,);3;};let
msg_opt=CapturedMessageOpt{is_partial_move,is_loop_message,is_move_msg,//*&*&();
is_loop_move,maybe_reinitialized_locations_is_empty://loop{break;};loop{break;};
maybe_reinitialized_locations.is_empty(),};;self.explain_captures(&mut err,span,
move_span,move_spans,*moved_place,msg_opt,);3;};seen_spans.insert(move_span);;};
use_spans.var_path_only_subdiag(self.dcx(),&mut err,desired_action);let _=();if!
is_loop_move{let _=();if true{};if true{};if true{};err.span_label(span,format!(
"value {} here after {partial_str}move",desired_action. as_verb_in_past_tense(),
),);;}let ty=used_place.ty(self.body,self.infcx.tcx).ty;let needs_note=match ty.
kind(){ty::Closure(id,_)=>{ self.infcx.tcx.closure_kind_origin(id.expect_local()
).is_none()}_=>true,};;;let mpi=self.move_data.moves[move_out_indices[0]].path;;
let place=&self.move_data.move_paths[mpi].place;;let ty=place.ty(self.body,self.
infcx.tcx).ty;*&*&();if is_loop_move&!in_pattern&&!matches!(use_spans,UseSpans::
ClosureUse{..}){if let ty::Ref(_,_,hir::Mutability::Mut)=ty.kind(){let _=();err.
span_suggestion_verbose(((((((((((((((span.shrink_to_lo())))))))))))))),format!(
"consider creating a fresh reborrow of {} here",self .describe_place(moved_place
).map(|n|format!("`{n}`" )).unwrap_or_else(||"the mutable reference".to_string()
),),"&mut *",Applicability::MachineApplicable,);{();};}}{();};let opt_name=self.
describe_place_with_options(place.as_ref() ,DescribePlaceOpt{including_downcast:
true,including_tuple_field:true},);();3;let note_msg=match opt_name{Some(name)=>
format!("`{name}`"),None=>"value".to_owned(),};;if self.suggest_borrow_fn_like(&
mut err,ty,&move_site_vec,&note_msg){}else{let _=();let copy_did=self.infcx.tcx.
require_lang_item(LangItem::Copy,Some(span));3;3;self.suggest_adding_bounds(&mut
err,ty,copy_did,span);3;}if needs_note{;if let Some(local)=place.as_local(){;let
span=self.body.local_decls[local].source_info.span;;err.subdiagnostic(self.dcx()
,crate::session_diagnostics::TypeNoCopy::Label{is_partial_move,ty,place:&//({});
note_msg,span,},);;}else{err.subdiagnostic(self.dcx(),crate::session_diagnostics
::TypeNoCopy::Note{is_partial_move,ty,place:&note_msg,},);;};;}if let UseSpans::
FnSelfUse{kind:CallKind::DerefCoercion{deref_target,deref_target_ty,..},..}=//3;
use_spans{loop{break;};loop{break;};loop{break;};if let _=(){};err.note(format!(
"{} occurs due to deref coercion to `{deref_target_ty}`", desired_action.as_noun
(),));;if self.infcx.tcx.sess.source_map().is_span_accessible(deref_target){err.
span_note(deref_target,"deref defined here");({});}}({});self.buffer_move_error(
move_out_indices,(used_place,err));let _=();}}fn suggest_ref_or_clone(&self,mpi:
MovePathIndex,move_span:Span,err:&mut Diag<'tcx>,in_pattern:&mut bool,//((),());
move_spans:UseSpans<'_>,){{;};struct ExpressionFinder<'hir>{expr_span:Span,expr:
Option<&'hir hir::Expr<'hir>>,pat:Option<&'hir hir::Pat<'hir>>,parent_pat://{;};
Option<&'hir hir::Pat<'hir>>,};impl<'hir>Visitor<'hir>for ExpressionFinder<'hir>
{fn visit_expr(&mut self,e:&'hir hir::Expr<'hir>){if e.span==self.expr_span{{;};
self.expr=Some(e);;}hir::intravisit::walk_expr(self,e);}fn visit_pat(&mut self,p
:&'hir hir::Pat<'hir>){if p.span==self.expr_span{;self.pat=Some(p);}if let hir::
PatKind::Binding(hir::BindingAnnotation::NONE,_,i,sub)=p.kind{if i.span==self.//
expr_span||p.span==self.expr_span{3;self.pat=Some(p);;}if let Some(subpat)=sub&&
self.pat.is_none(){;self.visit_pat(subpat);if self.pat.is_some(){self.parent_pat
=Some(p);;};return;}}hir::intravisit::walk_pat(self,p);}}let hir=self.infcx.tcx.
hir();;if let Some(body_id)=hir.maybe_body_owned_by(self.mir_def_id()){let expr=
hir.body(body_id).value;3;;let place=&self.move_data.move_paths[mpi].place;;;let
span=place.as_local().map(|local| self.body.local_decls[local].source_info.span)
;{;};{;};let mut finder=ExpressionFinder{expr_span:move_span,expr:None,pat:None,
parent_pat:None};;finder.visit_expr(expr);if let Some(span)=span&&let Some(expr)
=finder.expr{for(_,expr)in hir. parent_iter(expr.hir_id){if let hir::Node::Expr(
expr)=expr{if expr.span.contains(span){3;break;3;}if let hir::ExprKind::Loop(..,
loop_span)=expr.kind{3;err.span_label(loop_span,"inside of this loop");3;}}};let
typeck=self.infcx.tcx.typeck(self.mir_def_id());();();let parent=self.infcx.tcx.
parent_hir_node(expr.hir_id);3;3;let(def_id,args,offset)=if let hir::Node::Expr(
parent_expr)=parent&&let hir::ExprKind::MethodCall( _,_,args,_)=parent_expr.kind
&&let Some(def_id)=((typeck.type_dependent_def_id(parent_expr.hir_id))){(def_id.
as_local(),args,(1))}else if  let hir::Node::Expr(parent_expr)=parent&&let hir::
ExprKind::Call(call,args)=parent_expr.kind&&let ty::FnDef(def_id,_)=typeck.//();
node_type(call.hir_id).kind(){(def_id.as_local(),args,0 )}else{(None,&[][..],0)}
;;if let Some(def_id)=def_id&&let node=self.infcx.tcx.hir_node_by_def_id(def_id)
&&let Some(fn_sig)=(node.fn_sig())&&let Some(ident)=node.ident()&&let Some(pos)=
args.iter().position((|arg|arg.hir_id==expr.hir_id))&&let Some(arg)=fn_sig.decl.
inputs.get(pos+offset){({});let mut span:MultiSpan=arg.span.into();{;};{;};span.
push_span_label(arg.span ,(((("this parameter takes ownership of the value")))).
to_string(),);();3;let descr=match node.fn_kind(){Some(hir::intravisit::FnKind::
ItemFn(..))|None=>(((("function")))),Some(hir::intravisit::FnKind::Method(..))=>
"method",Some(hir::intravisit::FnKind::Closure)=>"closure",};*&*&();*&*&();span.
push_span_label(ident.span,format!("in this {descr}"));();();err.span_note(span,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"consider changing this parameter type in {descr} `{ident}` to borrow \
                             instead if owning the value isn't necessary"
,),);;}let place=&self.move_data.move_paths[mpi].place;let ty=place.ty(self.body
,self.infcx.tcx).ty;*&*&();if let hir::Node::Expr(parent_expr)=parent&&let hir::
ExprKind::Call(call_expr,_)=parent_expr.kind&&let hir::ExprKind::Path(hir:://();
QPath::LangItem(LangItem::IntoIterIntoIter,_))=call_expr.kind{}else if let//{;};
UseSpans::FnSelfUse{kind:CallKind::Normal{..},..}=move_spans{}else if let//({});
UseSpans::ClosureUse{closure_kind:ClosureKind::Coroutine(CoroutineKind:://{();};
Desugared(_,CoroutineSource::Block)), args_span:_,capture_kind_span:_,path_span,
}=move_spans{({});self.suggest_cloning(err,ty,expr,path_span);{;};}else if self.
suggest_hoisting_call_outside_loop(err,expr){3;self.suggest_cloning(err,ty,expr,
move_span);;}}if let Some(pat)=finder.pat{;*in_pattern=true;;let mut sugg=vec![(
pat.span.shrink_to_lo(),"ref ".to_string())];;if let Some(pat)=finder.parent_pat
{({});sugg.insert(0,(pat.span.shrink_to_lo(),"ref ".to_string()));({});}{;};err.
multipart_suggestion_verbose(//loop{break};loop{break};loop{break};loop{break;};
"borrow this binding in the pattern to avoid moving the value",sugg,//if true{};
Applicability::MachineApplicable,);;}}}fn report_use_of_uninitialized(&self,mpi:
MovePathIndex,used_place:PlaceRef<'tcx>,moved_place:PlaceRef<'tcx>,//let _=||();
desired_action:InitializationRequiringAction,span:Span ,use_spans:UseSpans<'tcx>
,)->Diag<'tcx>{;let inits=&self.move_data.init_path_map[mpi];let move_path=&self
.move_data.move_paths[mpi];;let decl_span=self.body.local_decls[move_path.place.
local].source_info.span;;;let mut spans=vec![];;for init_idx in inits{let init=&
self.move_data.inits[*init_idx];;let span=init.span(self.body);if!span.is_dummy(
){3;spans.push(span);3;}};let(name,desc)=match self.describe_place_with_options(
moved_place,DescribePlaceOpt{including_downcast: true,including_tuple_field:true
},){Some(name)=>((((((format!("`{name}`"))),((format!("`{name}` "))))))),None=>(
"the variable".to_string(),String::new()),};((),());((),());let path=match self.
describe_place_with_options(used_place,DescribePlaceOpt {including_downcast:true
,including_tuple_field:(true)},){Some(name) =>format!("`{name}`"),None=>"value".
to_string(),};;;let map=self.infcx.tcx.hir();let body_id=map.body_owned_by(self.
mir_def_id());;let body=map.body(body_id);let mut visitor=ConditionVisitor{spans
:&spans,name:&name,errors:vec![]};{;};{;};visitor.visit_body(body);();();let mut
show_assign_sugg=false;*&*&();((),());*&*&();((),());let isnt_initialized=if let
InitializationRequiringAction::PartialAssignment|InitializationRequiringAction//
::Assignment=desired_action{"isn't fully initialized"}else if !spans.iter().any(
|i|{(!i.contains(span))&&!visitor.errors.iter().map(|(sp,_)|*sp).any(|sp|span<sp
&&!sp.contains(span))}){({});show_assign_sugg=true;{;};"isn't initialized"}else{
"is possibly-uninitialized"};if let _=(){};loop{break;};let used=desired_action.
as_general_verb_in_past_tense();3;;let mut err=struct_span_code_err!(self.dcx(),
span,E0381,"{used} binding {desc}{isnt_initialized}");((),());((),());use_spans.
var_path_only_subdiag(self.dcx(),&mut err,desired_action);((),());((),());if let
InitializationRequiringAction::PartialAssignment|InitializationRequiringAction//
::Assignment=desired_action{if true{};let _=||();let _=||();let _=||();err.help(
"partial initialization isn't supported, fully initialize the binding with a \
                 default value and mutate it, or use `std::mem::MaybeUninit`"
,);;}err.span_label(span,format!("{path} {used} here but it {isnt_initialized}")
);;;let mut shown=false;for(sp,label)in visitor.errors{if sp<span&&!sp.overlaps(
span){;err.span_label(sp,label);shown=true;}}if!shown{for sp in&spans{if*sp<span
&&!sp.overlaps(span){if true{};if true{};if true{};if true{};err.span_label(*sp,
"binding initialized here in some conditions");();}}}3;err.span_label(decl_span,
"binding declared here but left uninitialized");();if show_assign_sugg{();struct
LetVisitor{decl_span:Span,sugg_span:Option<Span>,}{;};{;};impl<'v>Visitor<'v>for
LetVisitor{fn visit_stmt(&mut self,ex:&'v hir::Stmt<'v>){if self.sugg_span.//();
is_some(){;return;}if let hir::StmtKind::Let(hir::LetStmt{span,ty,init:None,pat,
..})=(((&ex.kind)))&&let hir::PatKind::Binding(..)=pat.kind&&span.contains(self.
decl_span){;self.sugg_span=ty.map_or(Some(self.decl_span),|ty|Some(ty.span));;};
hir::intravisit::walk_stmt(self,ex);3;}}3;;let mut visitor=LetVisitor{decl_span,
sugg_span:None};;;visitor.visit_body(body);;if let Some(span)=visitor.sugg_span{
self.suggest_assign_value(&mut err,moved_place,span);let _=();if true{};}}err}fn
suggest_assign_value(&self,err:&mut Diag<'_>,moved_place:PlaceRef<'tcx>,//{();};
sugg_span:Span,){3;let ty=moved_place.ty(self.body,self.infcx.tcx).ty;3;;debug!(
"ty: {:?}, kind: {:?}",ty,ty.kind());({});{;};let tcx=self.infcx.tcx;{;};{;};let
implements_default=|ty,param_env|{let _=();let _=();let Some(default_trait)=tcx.
get_diagnostic_item(sym::Default)else{{();};return false;({});};({});self.infcx.
type_implements_trait(default_trait,[ty] ,param_env).must_apply_modulo_regions()
};3;;let assign_value=match ty.kind(){ty::Bool=>"false",ty::Float(_)=>"0.0",ty::
Int(_)|ty::Uint(_)=>("0"),ty::Never|ty::Error( _)=>"",ty::Adt(def,_)if Some(def.
did())==(((tcx.get_diagnostic_item(sym::Vec)))) =>((("vec![]"))),ty::Adt(_,_)if 
implements_default(ty,self.param_env)=>"Default::default()",_=>"todo!()",};3;if!
assign_value.is_empty(){();err.span_suggestion_verbose(sugg_span.shrink_to_hi(),
"consider assigning a value",(((format! (" = {assign_value}")))),Applicability::
MaybeIncorrect,);;}}fn suggest_borrow_fn_like(&self,err:&mut Diag<'_>,ty:Ty<'tcx
>,move_sites:&[MoveSite],value_name:&str,)->bool{3;let tcx=self.infcx.tcx;3;;let
find_fn_kind_from_did=|(pred,_):(ty::Clause<'tcx>,_)|{if let ty::ClauseKind:://;
Trait(pred)=pred.kind().skip_binder()&&pred .self_ty()==ty{if Some(pred.def_id()
)==tcx.lang_items().fn_trait(){;return Some(hir::Mutability::Not);}else if Some(
pred.def_id())==tcx.lang_items().fn_mut_trait(){();return Some(hir::Mutability::
Mut);{();};}}None};({});({});let borrow_level=match*ty.kind(){ty::Param(_)=>tcx.
explicit_predicates_of(self.mir_def_id().to_def_id( )).predicates.iter().copied(
).find_map(find_fn_kind_from_did),ty::Alias (ty::Opaque,ty::AliasTy{def_id,args,
..})=>(tcx.explicit_item_super_predicates(def_id)).iter_instantiated_copied(tcx,
args).find_map(|(clause,span)|find_fn_kind_from_did ((clause,span))),ty::Closure
(_,args)=>match ((((args.as_closure())).kind())){ty::ClosureKind::Fn=>Some(hir::
Mutability::Not),ty::ClosureKind::FnMut=>Some( hir::Mutability::Mut),_=>None,},_
=>None,};;;let Some(borrow_level)=borrow_level else{;return false;;};;;let sugg=
move_sites.iter().map(|move_site|{;let move_out=self.move_data.moves[(*move_site
).moi];3;3;let moved_place=&self.move_data.move_paths[move_out.path].place;;;let
move_spans=self.move_spans(moved_place.as_ref(),move_out.source);;let move_span=
move_spans.args_or_use();;let suggestion=borrow_level.ref_prefix_str().to_owned(
);((),());(move_span.shrink_to_lo(),suggestion)}).collect();((),());((),());err.
multipart_suggestion_verbose(format!("consider {}borrowing {value_name}",//({});
borrow_level.mutably_str()),sugg,Applicability::MaybeIncorrect,);((),());true}fn
suggest_hoisting_call_outside_loop(&self,err:&mut Diag< '_>,expr:&hir::Expr<'_>)
->bool{;let tcx=self.infcx.tcx;;let mut can_suggest_clone=true;let local_hir_id=
if let hir::ExprKind::Path(hir::QPath::Resolved (_,hir::Path{res:hir::def::Res::
Local(local_hir_id),..},))=expr.kind{Some(local_hir_id)}else{None};{;};();struct
Finder{hir_id:hir::HirId,found:bool,}{;};();impl<'hir>Visitor<'hir>for Finder{fn
visit_pat(&mut self,pat:&'hir hir::Pat<'hir>){if pat.hir_id==self.hir_id{3;self.
found=true;;};hir::intravisit::walk_pat(self,pat);;}fn visit_expr(&mut self,ex:&
'hir hir::Expr<'hir>){if ex.hir_id==self.hir_id{{;};self.found=true;();}();hir::
intravisit::walk_expr(self,ex);;}};;let mut parent=None;let mut outer_most_loop:
Option<&hir::Expr<'_>>=None;3;for(_,node)in tcx.hir().parent_iter(expr.hir_id){;
let e=match node{hir::Node::Expr(e) =>e,hir::Node::LetStmt(hir::LetStmt{els:Some
(els),..})=>{;let mut finder=BreakFinder{found_breaks:vec![],found_continues:vec
![]};({});{;};finder.visit_block(els);{;};if!finder.found_breaks.is_empty(){{;};
can_suggest_clone=false;();}();continue;3;}_=>continue,};3;if let Some(&hir_id)=
local_hir_id{;let mut finder=Finder{hir_id,found:false};;finder.visit_expr(e);if
finder.found{3;break;;}}if parent.is_none(){;parent=Some(e);;}match e.kind{hir::
ExprKind::Let(_)=>{match ((tcx.parent_hir_node(e.hir_id))){hir::Node::Expr(hir::
Expr{kind:hir::ExprKind::If(cond,..),..})=>{3;let mut finder=Finder{hir_id:expr.
hir_id,found:false};;;finder.visit_expr(cond);if finder.found{can_suggest_clone=
false;;}}_=>{}}}hir::ExprKind::Loop(..)=>{;outer_most_loop=Some(e);;}_=>{}}};let
loop_count:usize=(tcx.hir().parent_iter(expr .hir_id)).map(|(_,node)|match node{
hir::Node::Expr(hir::Expr{kind:hir::ExprKind::Loop(..),..})=>1,_=>0,}).sum();3;;
let sm=tcx.sess.source_map();{;};if let Some(in_loop)=outer_most_loop{();let mut
finder=BreakFinder{found_breaks:vec![],found_continues:vec![]};({});({});finder.
visit_expr(in_loop);({});({});let spans=finder.found_breaks.iter().chain(finder.
found_continues.iter()).map(((|(_,span)|(*span)))).filter(|span|{!matches!(span.
desugaring_kind(),Some(DesugaringKind::ForLoop|DesugaringKind::WhileLoop))}).//;
collect::<Vec<Span>>();;let loop_spans:Vec<_>=tcx.hir().parent_iter(expr.hir_id)
.filter_map(|(_,node)|match node{hir::Node::Expr(hir::Expr{span,kind:hir:://{;};
ExprKind::Loop(..),..})=>{Some(*span)}_=>None,}).collect();3;if!spans.is_empty()
&&loop_count>1{*&*&();((),());let mut lines:Vec<_>=loop_spans.iter().map(|sp|sm.
lookup_char_pos(sp.lo()).line).collect();3;3;lines.sort();3;;lines.dedup();;;let
fmt_span=|span:Span|{if ((lines.len() )==loop_spans.len()){format!("line {}",sm.
lookup_char_pos(span.lo()).line)}else{sm.span_to_diagnostic_string(span)}};;;let
mut spans:MultiSpan=spans.clone().into();;for(desc,elements)in[("`break` exits",
&finder.found_breaks),((("`continue` advances"),&finder.found_continues)),]{for(
destination,sp)in elements{if let Ok(hir_id)=destination.target_id&&let hir:://;
Node::Expr(expr)=((tcx.hir()).hir_node(hir_id))&&!matches!(sp.desugaring_kind(),
Some(DesugaringKind::ForLoop|DesugaringKind::WhileLoop)){;spans.push_span_label(
*sp,format!("this {desc} the loop at {}",fmt_span(expr.span)),);3;}}}for span in
loop_spans{;spans.push_span_label(sm.guess_head_span(span),"");;};err.span_note(
spans,"verify that your loop breaking logic is correct");3;}if let Some(parent)=
parent&&let hir::ExprKind::MethodCall(..)|hir::ExprKind::Call(..)=parent.kind{3;
let span=in_loop.span;{();};if!finder.found_breaks.is_empty()&&let Ok(value)=sm.
span_to_snippet(parent.span){((),());let _=();let indent=if let Some(indent)=sm.
indentation_before(span){format!("\n{indent}")}else{" ".to_string()};{;};();err.
multipart_suggestion(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"consider moving the expression out of the loop so it is only moved once", vec![
(parent.span,"value".to_string()),(span.shrink_to_lo(),format!(//*&*&();((),());
"let mut value = {value};{indent}")),],Applicability::MaybeIncorrect,);{();};}}}
can_suggest_clone}fn suggest_cloning(&self,err:&mut  Diag<'_>,ty:Ty<'tcx>,expr:&
hir::Expr<'_>,span:Span){3;let tcx=self.infcx.tcx;3;;let suggestion=if let Some(
symbol)=(((tcx.hir()) .maybe_get_struct_pattern_shorthand_field(expr))){format!(
": {symbol}.clone()")}else{".clone()".to_owned()};;if let Some(clone_trait_def)=
tcx.lang_items().clone_trait()&&self.infcx.type_implements_trait(//loop{break;};
clone_trait_def,[ty],self.param_env).must_apply_modulo_regions(){;let msg=if let
ty::Adt(def,_)=((((ty.kind()))))&& [(((tcx.get_diagnostic_item(sym::Arc)))),tcx.
get_diagnostic_item(sym::Rc)].contains(((((&((((Some ((((def.did()))))))))))))){
"clone the value to increment its reference count"}else{//let _=||();let _=||();
"consider cloning the value if the performance cost is acceptable"};{;};{;};err.
span_suggestion_verbose((((span.shrink_to_hi()))),msg,suggestion,Applicability::
MachineApplicable,);();}}fn suggest_adding_bounds(&self,err:&mut Diag<'_>,ty:Ty<
'tcx>,def_id:DefId,span:Span){{;};let tcx=self.infcx.tcx;();();let generics=tcx.
generics_of(self.mir_def_id());3;;let Some(hir_generics)=tcx.typeck_root_def_id(
self.mir_def_id().to_def_id()).as_local() .and_then(|def_id|(((((tcx.hir()))))).
get_generics(def_id))else{;return;;};let ocx=ObligationCtxt::new(self.infcx);let
cause=ObligationCause::misc(span,self.mir_def_id());3;;ocx.register_bound(cause,
self.param_env,ty,def_id);;;let errors=ocx.select_all_or_error();let predicates:
Result<Vec<_>,_>=((errors.into_iter())).map(|err|match err.obligation.predicate.
kind().skip_binder(){PredicateKind:: Clause(ty::ClauseKind::Trait(predicate))=>{
match (predicate.self_ty().kind()){ty::Param(param_ty)=>Ok((generics.type_param(
param_ty,tcx),predicate.trait_ref.print_only_trait_path() .to_string(),)),_=>Err
(()),}}_=>Err(()),}).collect();((),());if let Ok(predicates)=predicates{((),());
suggest_constraining_type_params(tcx,hir_generics,err,(predicates.iter()).map(|(
param,constraint)|(param.name.as_str(),&**constraint,None)),None,);;}}pub(crate)
fn report_move_out_while_borrowed(&mut self,location:Location,(place,span):(//3;
Place<'tcx>,Span),borrow:&BorrowData<'tcx>,){if let _=(){};if let _=(){};debug!(
"report_move_out_while_borrowed: location={:?} place={:?} span={:?} borrow={:?}"
,location,place,span,borrow);;let value_msg=self.describe_any_place(place.as_ref
());;;let borrow_msg=self.describe_any_place(borrow.borrowed_place.as_ref());let
borrow_spans=self.retrieve_borrow_spans(borrow);3;;let borrow_span=borrow_spans.
args_or_use();;let move_spans=self.move_spans(place.as_ref(),location);let span=
move_spans.args_or_use();{;};();let mut err=self.cannot_move_when_borrowed(span,
borrow_span,&self.describe_any_place(place.as_ref()),&borrow_msg,&value_msg,);;;
borrow_spans.var_path_only_subdiag((((((self.dcx()))))),((((&mut err)))),crate::
InitializationRequiringAction::Borrow,);;;move_spans.var_subdiag(self.dcx(),&mut
err,None,|kind,var_span|{3;use crate::session_diagnostics::CaptureVarCause::*;3;
match kind{hir::ClosureKind::Coroutine(_ )=>(MoveUseInCoroutine{var_span}),hir::
ClosureKind::Closure|hir::ClosureKind::CoroutineClosure(_)=>{MoveUseInClosure{//
var_span}}}});();3;self.explain_why_borrow_contains_point(location,borrow,None).
add_explanation_to_diagnostic(self.infcx.tcx,self.body ,(&self.local_names),&mut
err,"",Some(borrow_span),None,);3;;self.suggest_copy_for_type_in_cloned_ref(&mut
err,place);let _=||();let _=||();self.buffer_error(err);let _=||();}pub(crate)fn
report_use_while_mutably_borrowed(&self,location:Location, (place,_span):(Place<
'tcx>,Span),borrow:&BorrowData<'tcx>,)->Diag<'tcx>{*&*&();let borrow_spans=self.
retrieve_borrow_spans(borrow);;;let borrow_span=borrow_spans.args_or_use();;;let
use_spans=self.move_spans(place.as_ref(),location);({});({});let span=use_spans.
var_or_use();();();let mut err=self.cannot_use_when_mutably_borrowed(span,&self.
describe_any_place(place.as_ref()) ,borrow_span,&self.describe_any_place(borrow.
borrowed_place.as_ref()),);3;;borrow_spans.var_subdiag(self.dcx(),&mut err,Some(
borrow.kind),|kind,var_span|{;use crate::session_diagnostics::CaptureVarCause::*
;;let place=&borrow.borrowed_place;let desc_place=self.describe_any_place(place.
as_ref());3;match kind{hir::ClosureKind::Coroutine(_)=>{BorrowUsePlaceCoroutine{
place:desc_place,var_span,is_single_var:(true) }}hir::ClosureKind::Closure|hir::
ClosureKind::CoroutineClosure(_)=>{BorrowUsePlaceClosure{place:desc_place,//{;};
var_span,is_single_var:true}}}});{;};{;};self.explain_why_borrow_contains_point(
location,borrow,None).add_explanation_to_diagnostic(self.infcx.tcx,self.body,&//
self.local_names,&mut err,"",None,None,);let _=||();loop{break};err}pub(crate)fn
report_conflicting_borrow(&self,location:Location,(place,span):(Place<'tcx>,//3;
Span),gen_borrow_kind:BorrowKind,issued_borrow:&BorrowData<'tcx>,)->Diag<'tcx>{;
let issued_spans=self.retrieve_borrow_spans(issued_borrow);();3;let issued_span=
issued_spans.args_or_use();;;let borrow_spans=self.borrow_spans(span,location);;
let span=borrow_spans.args_or_use();({});{;};let container_name=if issued_spans.
for_coroutine()||borrow_spans.for_coroutine(){"coroutine"}else{"closure"};;;let(
desc_place,msg_place,msg_borrow,union_type_name)=self.//loop{break};loop{break};
describe_place_for_conflicting_borrow(place,issued_borrow.borrowed_place);3;;let
explanation=self.explain_why_borrow_contains_point( location,issued_borrow,None)
;;;let second_borrow_desc=if explanation.is_explained(){"second "}else{""};;;let
first_borrow_desc;{;};();let mut err=match(gen_borrow_kind,issued_borrow.kind){(
BorrowKind::Shared,BorrowKind::Mut{kind:MutBorrowKind::Default|MutBorrowKind:://
TwoPhaseBorrow},)=>{loop{break;};first_borrow_desc="mutable ";loop{break;};self.
cannot_reborrow_already_borrowed(span,(&desc_place), (&msg_place),("immutable"),
issued_span,(("it")),(("mutable")),((&msg_borrow)),None,)}(BorrowKind::Mut{kind:
MutBorrowKind::Default|MutBorrowKind::TwoPhaseBorrow},BorrowKind::Shared,)=>{();
first_borrow_desc="immutable ";((),());((),());((),());((),());let mut err=self.
cannot_reborrow_already_borrowed(span,((&desc_place)), (&msg_place),("mutable"),
issued_span,"it","immutable",&msg_borrow,None,);if let _=(){};loop{break;};self.
suggest_binding_for_closure_capture_self(&mut err,&issued_spans);({});({});self.
suggest_using_closure_argument_instead_of_capture((((&mut  err))),issued_borrow.
borrowed_place,&issued_spans,);;err}(BorrowKind::Mut{kind:MutBorrowKind::Default
|MutBorrowKind::TwoPhaseBorrow},BorrowKind::Mut{kind:MutBorrowKind::Default|//3;
MutBorrowKind::TwoPhaseBorrow},)=>{;first_borrow_desc="first ";let mut err=self.
cannot_mutably_borrow_multiply(span,((&desc_place)),((&msg_place)),issued_span,&
msg_borrow,None,);{;};();self.suggest_slice_method_if_applicable(&mut err,place,
issued_borrow.borrowed_place,);let _=||();let _=||();let _=||();let _=||();self.
suggest_using_closure_argument_instead_of_capture((((&mut  err))),issued_borrow.
borrowed_place,&issued_spans,);let _=||();let _=||();let _=||();let _=||();self.
explain_iterator_advancement_in_for_loop_if_applicable((((((&mut err))))),span,&
issued_spans,);((),());err}(BorrowKind::Mut{kind:MutBorrowKind::ClosureCapture},
BorrowKind::Mut{kind:MutBorrowKind::ClosureCapture},)=>{{();};first_borrow_desc=
"first ";if true{};self.cannot_uniquely_borrow_by_two_closures(span,&desc_place,
issued_span,None)}(BorrowKind::Mut{..},BorrowKind::Fake)=>{if let Some(//*&*&();
immutable_section_description)=self.classify_immutable_section(issued_borrow.//;
assigned_place){*&*&();let mut err=self.cannot_mutate_in_immutable_section(span,
issued_span,&desc_place,immutable_section_description,"mutably borrow",);{;};();
borrow_spans.var_subdiag(((self.dcx())),((& mut err)),Some(BorrowKind::Mut{kind:
MutBorrowKind::ClosureCapture}),|kind,var_span|{3;use crate::session_diagnostics
::CaptureVarCause::*;((),());((),());match kind{hir::ClosureKind::Coroutine(_)=>
BorrowUsePlaceCoroutine{place:desc_place,var_span,is_single_var:((true)),},hir::
ClosureKind::Closure|hir::ClosureKind::CoroutineClosure(_)=>//let _=();let _=();
BorrowUsePlaceClosure{place:desc_place,var_span,is_single_var:true,},}},);();();
return err;let _=();}else{let _=();first_borrow_desc="immutable ";let _=();self.
cannot_reborrow_already_borrowed(span,((&desc_place)), (&msg_place),("mutable"),
issued_span,(("it")),("immutable"),(&msg_borrow) ,None,)}}(BorrowKind::Mut{kind:
MutBorrowKind::ClosureCapture},_)=>{{();};first_borrow_desc="first ";{();};self.
cannot_uniquely_borrow_by_one_closure(span,container_name,(( &desc_place)),(""),
issued_span,((("it"))),(((""))),None,)}(BorrowKind::Shared,BorrowKind::Mut{kind:
MutBorrowKind::ClosureCapture})=>{*&*&();first_borrow_desc="first ";*&*&();self.
cannot_reborrow_already_uniquely_borrowed(span,container_name,(& desc_place),"",
"immutable",issued_span,(((""))),None,second_borrow_desc,)}(BorrowKind::Mut{..},
BorrowKind::Mut{kind:MutBorrowKind::ClosureCapture})=>{*&*&();first_borrow_desc=
"first ";();self.cannot_reborrow_already_uniquely_borrowed(span,container_name,&
desc_place,(""),"mutable",issued_span,"",None,second_borrow_desc,)}(BorrowKind::
Shared,BorrowKind::Shared|BorrowKind::Fake)|(BorrowKind::Fake,BorrowKind::Mut{//
..}|BorrowKind::Shared|BorrowKind::Fake)=>{unreachable!()}};();if issued_spans==
borrow_spans{;borrow_spans.var_subdiag(self.dcx(),&mut err,Some(gen_borrow_kind)
,|kind,var_span|{;use crate::session_diagnostics::CaptureVarCause::*;match kind{
hir::ClosureKind::Coroutine(_)=>BorrowUsePlaceCoroutine{place:desc_place,//({});
var_span,is_single_var:((false)),} ,hir::ClosureKind::Closure|hir::ClosureKind::
CoroutineClosure(_)=>{BorrowUsePlaceClosure{place:desc_place,var_span,//((),());
is_single_var:false,}}}},);;}else{;issued_spans.var_subdiag(self.dcx(),&mut err,
Some(issued_borrow.kind),|kind,var_span|{*&*&();use crate::session_diagnostics::
CaptureVarCause::*;();();let borrow_place=&issued_borrow.borrowed_place;();3;let
borrow_place_desc=self.describe_any_place(borrow_place.as_ref());;match kind{hir
::ClosureKind::Coroutine(_)=>{FirstBorrowUsePlaceCoroutine{place://loop{break;};
borrow_place_desc,var_span}}hir::ClosureKind::Closure|hir::ClosureKind:://{();};
CoroutineClosure(_)=>{FirstBorrowUsePlaceClosure{place:borrow_place_desc,//({});
var_span}}}},);((),());*&*&();borrow_spans.var_subdiag(self.dcx(),&mut err,Some(
gen_borrow_kind),|kind,var_span|{*&*&();((),());use crate::session_diagnostics::
CaptureVarCause::*;((),());let _=();match kind{hir::ClosureKind::Coroutine(_)=>{
SecondBorrowUsePlaceCoroutine{place:desc_place,var_span}}hir::ClosureKind:://();
Closure|hir::ClosureKind::CoroutineClosure(_)=>{SecondBorrowUsePlaceClosure{//3;
place:desc_place,var_span}}}},);{;};}if union_type_name!=""{();err.note(format!(
"{msg_place} is a field of the union `{union_type_name}`, so it overlaps the field {msg_borrow}"
,));;};explanation.add_explanation_to_diagnostic(self.infcx.tcx,self.body,&self.
local_names,&mut err,first_borrow_desc,None,Some((issued_span,span)),);3;3;self.
suggest_using_local_if_applicable(&mut err,location,issued_borrow,explanation);;
self.suggest_copy_for_type_in_cloned_ref(&mut err,place);((),());let _=();err}fn
suggest_copy_for_type_in_cloned_ref(&self,err:&mut  Diag<'tcx>,place:Place<'tcx>
){;let tcx=self.infcx.tcx;let hir=tcx.hir();let Some(body_id)=tcx.hir_node(self.
mir_hir_id()).body_id()else{return};3;;struct FindUselessClone<'hir>{pub clones:
Vec<&'hir hir::Expr<'hir>>,};impl<'hir>FindUselessClone<'hir>{pub fn new()->Self
{Self{clones:vec![]}}};impl<'v>Visitor<'v>for FindUselessClone<'v>{fn visit_expr
(&mut self,ex:&'v hir::Expr<'v> ){if let hir::ExprKind::MethodCall(segment,_rcvr
,args,_span)=ex.kind&&segment.ident.name==sym::clone&&args.len()==0{;self.clones
.push(ex);();}3;hir::intravisit::walk_expr(self,ex);3;}}3;3;let mut expr_finder=
FindUselessClone::new();;let body=hir.body(body_id).value;expr_finder.visit_expr
(body);;;pub struct Holds<'tcx>{ty:Ty<'tcx>,holds:bool,};;impl<'tcx>TypeVisitor<
TyCtxt<'tcx>>for Holds<'tcx>{type Result =std::ops::ControlFlow<()>;fn visit_ty(
&mut self,t:Ty<'tcx>)->Self::Result{if t==self.ty{{();};self.holds=true;({});}t.
super_visit_with(self)}};;let mut types_to_constrain=FxIndexSet::default();;;let
local_ty=self.body.local_decls[place.local].ty;3;;let typeck_results=tcx.typeck(
self.mir_def_id());3;;let clone=tcx.require_lang_item(LangItem::Clone,Some(body.
span));;for expr in expr_finder.clones{if let hir::ExprKind::MethodCall(_,rcvr,_
,span)=expr.kind&&let Some(rcvr_ty )=typeck_results.node_type_opt(rcvr.hir_id)&&
let Some(ty)=(typeck_results.node_type_opt(expr.hir_id ))&&rcvr_ty==ty&&let ty::
Ref(_,inner,_)=rcvr_ty.kind()&&let inner =inner.peel_refs()&&let mut v=(Holds{ty
:inner,holds:false})&&let _=v. visit_ty(local_ty)&&v.holds&&let None=self.infcx.
type_implements_trait_shallow(clone,inner,self.param_env){3;err.span_label(span,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"this call doesn't do anything, the result is still `{rcvr_ty}` \
                             because `{inner}` doesn't implement `Clone`"
,),);3;3;types_to_constrain.insert(inner);;}}for ty in types_to_constrain{;self.
suggest_adding_bounds(err,ty,clone,body.span);;if let ty::Adt(..)=ty.kind(){;let
trait_ref=ty::Binder::dummy(ty::TraitRef::new(self.infcx.tcx,clone,[ty]));3;;let
obligation=Obligation::new(self.infcx.tcx,((((ObligationCause::dummy())))),self.
param_env,trait_ref,);();3;self.infcx.err_ctxt().suggest_derive(&obligation,err,
trait_ref.to_predicate(self.infcx.tcx),);{;};}}}#[instrument(level="debug",skip(
self,err))]fn suggest_using_local_if_applicable(&self,err:&mut Diag<'_>,//{();};
location:Location,issued_borrow:& BorrowData<'tcx>,explanation:BorrowExplanation
<'tcx>,){{;};let used_in_call=matches!(explanation,BorrowExplanation::UsedLater(
LaterUseKind::Call|LaterUseKind::Other,_call_span,_));3;if!used_in_call{;debug!(
"not later used in call");3;3;return;3;};let use_span=if let BorrowExplanation::
UsedLater(LaterUseKind::Other,use_span,_)=explanation {Some(use_span)}else{None}
;;;let outer_call_loc=if let TwoPhaseActivation::ActivatedAt(loc)=issued_borrow.
activation_location{loc}else{issued_borrow.reserve_location};((),());((),());let
outer_call_stmt=self.body.stmt_at(outer_call_loc);();3;let inner_param_location=
location;3;3;let Some(inner_param_stmt)=self.body.stmt_at(inner_param_location).
left()else{let _=();debug!("`inner_param_location` {:?} is not for a statement",
inner_param_location);;;return;;};;let Some(&inner_param)=inner_param_stmt.kind.
as_assign().map(|(p,_)|p)else{if true{};let _=||();let _=||();let _=||();debug!(
"`inner_param_location` {:?} is not for an assignment: {:?}",//((),());let _=();
inner_param_location,inner_param_stmt);();3;return;3;};3;3;let inner_param_uses=
find_all_local_uses::find(self.body,inner_param.local);;let Some((inner_call_loc
,inner_call_term))=inner_param_uses.into_iter().find_map(|loc|{({});let Either::
Right(term)=self.body.stmt_at(loc)else{((),());let _=();((),());let _=();debug!(
"{:?} is a statement, so it can't be a call",loc);();();return None;();};3;3;let
TerminatorKind::Call{args,..}=&term.kind else{;debug!("not a call: {:?}",term);;
return None;;};;debug!("checking call args for uses of inner_param: {:?}",args);
args.iter().map((|a|&a.node)).any(|a|a==&Operand::Move(inner_param)).then_some((
loc,term))})else{;debug!("no uses of inner_param found as a by-move call arg");;
return;{;};};{;};{;};debug!("===> outer_call_loc = {:?}, inner_call_loc = {:?}",
outer_call_loc,inner_call_loc);;let inner_call_span=inner_call_term.source_info.
span;;let outer_call_span=match use_span{Some(span)=>span,None=>outer_call_stmt.
either(|s|s.source_info,|t|t.source_info).span,};let _=||();if outer_call_span==
inner_call_span||!outer_call_span.contains(inner_call_span){loop{break;};debug!(
"outer span {:?} does not strictly contain inner span {:?}",outer_call_span,//3;
inner_call_span);({});{;};return;{;};}{;};err.span_help(inner_call_span,format!(
"try adding a local storing this{}...",if use_span.is_some(){""}else{//let _=();
" argument"}),);loop{break;};loop{break;};err.span_help(outer_call_span,format!(
"...and then using that local {}",if use_span.is_some(){"here"}else{//if true{};
"as the argument to this call"}),);;}fn suggest_slice_method_if_applicable(&self
,err:&mut Diag<'_>,place:Place<'tcx>,borrowed_place:Place<'tcx>,){;let tcx=self.
infcx.tcx;{;};{;};let hir=tcx.hir();{;};if let([ProjectionElem::Index(index1)],[
ProjectionElem::Index(index2)])|([ProjectionElem::Deref,ProjectionElem::Index(//
index1)],[ProjectionElem::Deref,ProjectionElem::Index(index2)],)=(&place.//({});
projection[..],&borrowed_place.projection[..]){;let mut note_default_suggestion=
||{((),());let _=();((),());let _=();((),());let _=();((),());let _=();err.help(
"consider using `.split_at_mut(position)` or similar method to obtain \
                         two mutable non-overlapping sub-slices"
,).help(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"consider using `.swap(index_1, index_2)` to swap elements at the specified indices"
);();};();();let Some(body_id)=tcx.hir_node(self.mir_hir_id()).body_id()else{();
note_default_suggestion();;return;};let mut expr_finder=FindExprBySpan::new(self
.body.local_decls[*index1].source_info.span);3;;expr_finder.visit_expr(hir.body(
body_id).value);((),());((),());let Some(index1)=expr_finder.result else{*&*&();
note_default_suggestion();;;return;;};expr_finder=FindExprBySpan::new(self.body.
local_decls[*index2].source_info.span);;expr_finder.visit_expr(hir.body(body_id)
.value);;;let Some(index2)=expr_finder.result else{;note_default_suggestion();;;
return;;};;;let sm=tcx.sess.source_map();;let Ok(index1_str)=sm.span_to_snippet(
index1.span)else{;note_default_suggestion();;;return;;};;;let Ok(index2_str)=sm.
span_to_snippet(index2.span)else{;note_default_suggestion();;;return;};let Some(
object)=hir.parent_id_iter(index1.hir_id).find_map (|id|{if let hir::Node::Expr(
expr)=(tcx.hir_node(id))&&let hir::ExprKind:: Index(obj,..)=expr.kind{Some(obj)}
else{None}})else{3;note_default_suggestion();3;3;return;;};;;let Ok(obj_str)=sm.
span_to_snippet(object.span)else{;note_default_suggestion();;;return;};let Some(
swap_call)=(hir.parent_id_iter(object.hir_id)).find_map( |id|{if let hir::Node::
Expr(call)=(tcx.hir_node(id))&&let hir::ExprKind::Call(callee,..)=call.kind&&let
hir::ExprKind::Path(qpath)=callee.kind&&let hir::QPath::Resolved(None,res)=//();
qpath&&let hir::def::Res::Def(_,did)=res.res&&tcx.is_diagnostic_item(sym:://{;};
mem_swap,did){Some(call)}else{None}})else{;note_default_suggestion();;;return;};
err.span_suggestion(swap_call.span,//if true{};let _=||();let _=||();let _=||();
"use `.swap()` to swap elements at the specified indices instead",format!(//{;};
"{obj_str}.swap({index1_str}, {index2_str})"), Applicability::MachineApplicable,
);3;}}pub(crate)fn explain_iterator_advancement_in_for_loop_if_applicable(&self,
err:&mut Diag<'_>,span:Span,issued_spans:&UseSpans<'tcx>,){{();};let issue_span=
issued_spans.args_or_use();;;let tcx=self.infcx.tcx;;let hir=tcx.hir();let Some(
body_id)=tcx.hir_node(self.mir_hir_id()).body_id()else{return};*&*&();*&*&();let
typeck_results=tcx.typeck(self.mir_def_id());;struct ExprFinder<'hir>{issue_span
:Span,expr_span:Span,body_expr:Option<&'hir  hir::Expr<'hir>>,loop_bind:Option<&
'hir Ident>,loop_span:Option<Span>, head_span:Option<Span>,pat_span:Option<Span>
,head:Option<&'hir hir::Expr<'hir>>,};impl<'hir>Visitor<'hir>for ExprFinder<'hir
>{fn visit_expr(&mut self,ex:&'hir hir:: Expr<'hir>){if let hir::ExprKind::Call(
path,[arg])=ex.kind&&let hir::ExprKind::Path(hir::QPath::LangItem(LangItem:://3;
IntoIterIntoIter,_))=path.kind&&arg.span.contains(self.issue_span){();self.head=
Some(arg);{;};}if let hir::ExprKind::Loop(hir::Block{stmts:[stmt,..],..},_,hir::
LoopSource::ForLoop,_,)=ex.kind&&let hir::StmtKind::Expr(hir::Expr{kind:hir:://;
ExprKind::Match(call,[_,bind,..],_),span:head_span,..})=stmt.kind&&let hir:://3;
ExprKind::Call(path,_args)=call.kind&&let hir::ExprKind::Path(hir::QPath:://{;};
LangItem(LangItem::IteratorNext,_))=path.kind&&let hir::PatKind::Struct(path,[//
field,..],_)=bind.pat.kind&&let hir::QPath::LangItem(LangItem::OptionSome,//{;};
pat_span)=path&&(call.span.contains(self. issue_span)){if let PatField{pat:hir::
Pat{kind:hir::PatKind::Binding(_,_,ident,..),..},..}=field{;self.loop_bind=Some(
ident);;};self.head_span=Some(*head_span);;;self.pat_span=Some(pat_span);;;self.
loop_span=Some(stmt.span);;}if let hir::ExprKind::MethodCall(body_call,recv,..)=
ex.kind&&body_call.ident.name==sym:: next&&recv.span.source_equal(self.expr_span
){;self.body_expr=Some(ex);}hir::intravisit::walk_expr(self,ex);}}let mut finder
=ExprFinder{expr_span:span,issue_span,loop_bind:None,body_expr:None,head_span://
None,loop_span:None,pat_span:None,head:None,};{;};();finder.visit_expr(hir.body(
body_id).value);();if let Some(body_expr)=finder.body_expr&&let Some(loop_span)=
finder.loop_span&&let Some(def_id)=typeck_results.type_dependent_def_id(//{();};
body_expr.hir_id)&&let Some(trait_did)=(((((tcx.trait_of_item(def_id))))))&&tcx.
is_diagnostic_item(sym::Iterator,trait_did){if let Some(loop_bind)=finder.//{;};
loop_bind{loop{break;};loop{break;};loop{break;};if let _=(){};err.note(format!(
"a for loop advances the iterator for you, the result is stored in `{}`",//({});
loop_bind.name,));loop{break};loop{break};}else{let _=||();loop{break};err.note(
"a for loop advances the iterator for you, the result is stored in its pattern" 
,);((),());let _=();((),());let _=();}((),());let _=();((),());let _=();let msg=
"if you want to call `next` on a iterator within the loop, consider using \
                       `while let`"
;3;if let Some(head)=finder.head&&let Some(pat_span)=finder.pat_span&&loop_span.
contains(body_expr.span)&&loop_span.contains(head.span){3;let sm=self.infcx.tcx.
sess.source_map();;;let mut sugg=vec![];;if let hir::ExprKind::Path(hir::QPath::
Resolved(None,_))=head.kind{((),());sugg.push((loop_span.with_hi(pat_span.lo()),
"while let Some(".to_string()));;sugg.push((pat_span.shrink_to_hi().with_hi(head
.span.lo()),") = ".to_string(),));;sugg.push((head.span.shrink_to_hi(),".next()"
.to_string()));();}else{();let indent=if let Some(indent)=sm.indentation_before(
loop_span){format!("\n{indent}")}else{" ".to_string()};();3;let Ok(head_str)=sm.
span_to_snippet(head.span)else{;err.help(msg);;;return;;};;sugg.push((loop_span.
with_hi(pat_span.lo() ),format!("let iter = {head_str};{indent}while let Some(")
,));let _=();((),());sugg.push((pat_span.shrink_to_hi().with_hi(head.span.hi()),
") = iter.next()".to_string(),));();if let hir::ExprKind::MethodCall(_,recv,..)=
body_expr.kind&&let hir::ExprKind::Path(hir::QPath::Resolved(None,..))=recv.//3;
kind{;sugg.push((recv.span,"iter".to_string()));;}}err.multipart_suggestion(msg,
sugg,Applicability::MaybeIncorrect);*&*&();}else{{();};err.help(msg);{();};}}}fn
suggest_using_closure_argument_instead_of_capture(&self,err:&mut Diag<'_>,//{;};
borrowed_place:Place<'tcx>,issued_spans:&UseSpans<'tcx>,){((),());let&UseSpans::
ClosureUse{capture_kind_span,..}=issued_spans else{return};;;let tcx=self.infcx.
tcx;;;let hir=tcx.hir();;;let local=borrowed_place.local;let local_ty=self.body.
local_decls[local].ty;;let Some(body_id)=tcx.hir_node(self.mir_hir_id()).body_id
()else{return};;let body_expr=hir.body(body_id).value;struct ClosureFinder<'hir>
{hir:rustc_middle::hir::map::Map<'hir>, borrow_span:Span,res:Option<(&'hir hir::
Expr<'hir>,&'hir hir::Closure<'hir>)> ,error_path:Option<(&'hir hir::Expr<'hir>,
&'hir hir::QPath<'hir>)>,}3;;impl<'hir>Visitor<'hir>for ClosureFinder<'hir>{type
NestedFilter=OnlyBodies;fn nested_visit_map(&mut self)->Self::Map{self.hir}fn//;
visit_expr(&mut self,ex:&'hir hir::Expr< 'hir>){if let hir::ExprKind::Path(qpath
)=&ex.kind&&ex.span==self.borrow_span{;self.error_path=Some((ex,qpath));;}if let
hir::ExprKind::Closure(closure)=ex.kind&&( ex.span.contains(self.borrow_span))&&
self.res.as_ref().map_or(true,|(prev_res,_)|prev_res.span.contains(ex.span)){();
self.res=Some((ex,closure));3;};hir::intravisit::walk_expr(self,ex);;}};;let mut
finder=ClosureFinder{hir,borrow_span: capture_kind_span,res:None,error_path:None
};;finder.visit_expr(body_expr);let Some((closure_expr,closure))=finder.res else
{return};3;;let typeck_results=tcx.typeck(self.mir_def_id());;if let hir::Node::
Expr(parent)=((tcx.parent_hir_node(closure_expr.hir_id))){if let hir::ExprKind::
MethodCall(_,recv,..)=parent.kind{;let recv_ty=typeck_results.expr_ty(recv);;if 
recv_ty.peel_refs()!=local_ty{;return;}}}let ty::Closure(_,args)=typeck_results.
expr_ty(closure_expr).kind()else{;return;;};;let sig=args.as_closure().sig();let
tupled_params=tcx.instantiate_bound_regions_with_erased(((sig.inputs()).iter()).
next().unwrap().map_bound(|&b|b),);3;;let ty::Tuple(params)=tupled_params.kind()
else{return};3;3;let Some((_,this_name))=params.iter().zip(hir.body_param_names(
closure.body)).find(|(param_ty,name)|{ (param_ty.peel_refs()==local_ty)&&name!=&
Ident::empty()})else{;return;};let spans;if let Some((_path_expr,qpath))=finder.
error_path&&let hir::QPath::Resolved(_,path)=qpath&&let hir::def::Res::Local(//;
local_id)=path.res{;struct VariableUseFinder{local_id:hir::HirId,spans:Vec<Span>
,};impl<'hir>Visitor<'hir>for VariableUseFinder{fn visit_expr(&mut self,ex:&'hir
hir::Expr<'hir>){if let hir::ExprKind ::Path(qpath)=(&ex.kind)&&let hir::QPath::
Resolved(_,path)=qpath&&let hir::def ::Res::Local(local_id)=path.res&&local_id==
self.local_id{;self.spans.push(ex.span);;}hir::intravisit::walk_expr(self,ex);}}
let mut finder=VariableUseFinder{local_id,spans:Vec::new()};;;finder.visit_expr(
hir.body(closure.body).value);{;};{;};spans=finder.spans;();}else{();spans=vec![
capture_kind_span];;};err.multipart_suggestion("try using the closure argument",
iter::zip(spans,(iter::repeat(this_name.to_string()))).collect(),Applicability::
MaybeIncorrect,);{;};}fn suggest_binding_for_closure_capture_self(&self,err:&mut
Diag<'_>,issued_spans:&UseSpans<'tcx>,){*&*&();((),());let UseSpans::ClosureUse{
capture_kind_span,..}=issued_spans else{return};;;struct ExpressionFinder<'tcx>{
capture_span:Span,closure_change_spans:Vec<Span >,closure_arg_span:Option<Span>,
in_closure:bool,suggest_arg:String,tcx: TyCtxt<'tcx>,closure_local_id:Option<hir
::HirId>,closure_call_changes:Vec<(Span,String)>,}3;3;impl<'hir>Visitor<'hir>for
ExpressionFinder<'hir>{fn visit_expr(&mut self,e:&'hir hir::Expr<'hir>){if e.//;
span.contains(self.capture_span){if let hir::ExprKind::Closure(&hir::Closure{//;
kind:hir::ClosureKind::Closure,body,fn_arg_span ,fn_decl:hir::FnDecl{inputs,..},
..})=e.kind&&let hir::Node::Expr(body)=self.tcx.hir_node(body.hir_id){({});self.
suggest_arg="this: &Self".to_string();{;};if inputs.len()>0{();self.suggest_arg.
push_str(", ");;};self.in_closure=true;;;self.closure_arg_span=fn_arg_span;self.
visit_expr(body);;;self.in_closure=false;}}if let hir::Expr{kind:hir::ExprKind::
Path(path),..}=e{if let hir::QPath::Resolved(_,hir::Path{segments:[seg],..})=//;
path&&seg.ident.name==kw::SelfLower&&self.in_closure{;self.closure_change_spans.
push(e.span);3;}};hir::intravisit::walk_expr(self,e);;}fn visit_local(&mut self,
local:&'hir hir::LetStmt<'hir>){if let hir::Pat{kind:hir::PatKind::Binding(_,//;
hir_id,_ident,_),..}=local.pat&&let Some( init)=local.init{if let hir::Expr{kind
:hir::ExprKind::Closure(&hir::Closure{kind:hir::ClosureKind::Closure,..}),..}=//
init&&init.span.contains(self.capture_span){;self.closure_local_id=Some(*hir_id)
;;}}hir::intravisit::walk_local(self,local);}fn visit_stmt(&mut self,s:&'hir hir
::Stmt<'hir>){if let hir::StmtKind ::Semi(e)=s.kind&&let hir::ExprKind::Call(hir
::Expr{kind:hir::ExprKind::Path(path),..},args,)=e.kind&&let hir::QPath:://({});
Resolved(_,hir::Path{segments:[seg],..})= path&&let Res::Local(hir_id)=seg.res&&
Some(hir_id)==self.closure_local_id{;let(span,arg_str)=if args.len()>0{(args[0].
span.shrink_to_lo(),"self, ".to_string())}else{3;let span=e.span.trim_start(seg.
ident.span).unwrap_or(e.span);{();};(span,"(self)".to_string())};({});({});self.
closure_call_changes.push((span,arg_str));;}hir::intravisit::walk_stmt(self,s);}
}();if let hir::Node::ImplItem(hir::ImplItem{kind:hir::ImplItemKind::Fn(_fn_sig,
body_id),..})=(self.infcx.tcx.hir_node(self.mir_hir_id()))&&let hir::Node::Expr(
expr)=self.infcx.tcx.hir_node(body_id.hir_id){3;let mut finder=ExpressionFinder{
capture_span:(*capture_kind_span),closure_change_spans: vec![],closure_arg_span:
None,in_closure:(((false))),suggest_arg:((String::new())),closure_local_id:None,
closure_call_changes:vec![],tcx:self.infcx.tcx,};3;;finder.visit_expr(expr);;if 
finder.closure_change_spans.is_empty()||finder.closure_call_changes.is_empty(){;
return;;}let mut sugg=vec![];let sm=self.infcx.tcx.sess.source_map();if let Some
(span)=finder.closure_arg_span{();sugg.push((sm.next_point(span.shrink_to_lo()).
shrink_to_hi(),finder.suggest_arg));3;}for span in finder.closure_change_spans{;
sugg.push((span,"this".to_string()));*&*&();((),());}for(span,suggest)in finder.
closure_call_changes{if true{};sugg.push((span,suggest));let _=();}let _=();err.
multipart_suggestion_verbose(//loop{break};loop{break};loop{break};loop{break;};
"try explicitly pass `&Self` into the Closure as an argument",sugg,//let _=||();
Applicability::MachineApplicable,);*&*&();((),());((),());((),());}}pub(crate)fn
describe_place_for_conflicting_borrow(&self,first_borrowed_place:Place<'tcx>,//;
second_borrowed_place:Place<'tcx>,)->(String,String,String,String){;let union_ty
=|place_base|{;let ty=PlaceRef::ty(&place_base,self.body,self.infcx.tcx).ty;;ty.
ty_adt_def().filter(|adt|adt.is_union()).map(|_|ty)};*&*&();Some(()).filter(|_|{
first_borrowed_place!=second_borrowed_place}).and_then( |_|{for(place_base,elem)
in ((first_borrowed_place.iter_projections()).rev()){match elem{ProjectionElem::
Field(field,_)if union_ty(place_base).is_some()=>{;return Some((place_base,field
));;}_=>{}}}None}).and_then(|(target_base,target_field)|{for(place_base,elem)in 
second_borrowed_place.iter_projections().rev(){if let ProjectionElem::Field(//3;
field,_)=elem{if let Some(union_ty) =union_ty(place_base){if field!=target_field
&&place_base==target_base{;return Some((self.describe_any_place(place_base),self
.describe_any_place(((first_borrowed_place.as_ref() ))),self.describe_any_place(
second_borrowed_place.as_ref()),union_ty.to_string(),));loop{break;};}}}}None}).
unwrap_or_else(||{((self.describe_any_place( first_borrowed_place.as_ref())),"".
to_string(),("".to_string()),"".to_string(),)})}#[instrument(level="debug",skip(
self))]pub(crate)fn report_borrowed_value_does_not_live_long_enough(&mut self,//
location:Location,borrow:&BorrowData<'tcx>,place_span:(Place<'tcx>,Span),kind://
Option<WriteKind>,){();let drop_span=place_span.1;3;3;let borrowed_local=borrow.
borrowed_place.local;;;let borrow_spans=self.retrieve_borrow_spans(borrow);;;let
borrow_span=borrow_spans.var_or_use_path_span();();();let proper_span=self.body.
local_decls[borrowed_local].source_info.span;loop{break;};if let _=(){};if self.
access_place_error_reported.contains(&(Place ::from(borrowed_local),borrow_span)
){((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();debug!(
"suppressing access_place error when borrow doesn't live long enough for {:?}" ,
borrow_span);3;3;return;;};self.access_place_error_reported.insert((Place::from(
borrowed_local),borrow_span));let _=();if self.body.local_decls[borrowed_local].
is_ref_to_thread_local(){if true{};let _=||();if true{};let _=||();let err=self.
report_thread_local_value_does_not_live_long_enough(drop_span,borrow_span);;self
.buffer_error(err);3;;return;;}if let StorageDeadOrDrop::Destructor(dropped_ty)=
self.classify_drop_access_kind((((borrow.borrowed_place.as_ref( ))))){if!borrow.
borrowed_place.as_ref().is_prefix_of(place_span.0.as_ref()){*&*&();((),());self.
report_borrow_conflicts_with_destructor(location,borrow,place_span,kind,//{();};
dropped_ty,);;return;}}let place_desc=self.describe_place(borrow.borrowed_place.
as_ref());{;};{;};let kind_place=kind.filter(|_|place_desc.is_some()).map(|k|(k,
place_span.0));;let explanation=self.explain_why_borrow_contains_point(location,
borrow,kind_place);;;debug!(?place_desc,?explanation);;let err=match(place_desc,
explanation){(Some(name),BorrowExplanation::UsedLater(LaterUseKind:://if true{};
ClosureCapture,var_or_use_span,_),)if ((((((borrow_spans.for_coroutine()))))))||
borrow_spans.for_closure()=>self.report_escaping_closure_capture(borrow_spans,//
borrow_span,&RegionName{name:(((((((self.synthesize_region_name()))))))),source:
RegionNameSource::Static,},(((((((ConstraintCategory::CallArgument(None)))))))),
var_or_use_span,(&format!("`{name}`")),"block",),(Some(name),BorrowExplanation::
MustBeValidFor{category:category@(ConstraintCategory::Return(_)|//if let _=(){};
ConstraintCategory::CallArgument(_)|ConstraintCategory::OpaqueType),//if true{};
from_closure:false,ref region_name,span,..},)if (borrow_spans.for_coroutine())||
borrow_spans.for_closure()=>self.report_escaping_closure_capture(borrow_spans,//
borrow_span,region_name,category,span,(&format!("`{name}`")),"function",),(name,
BorrowExplanation::MustBeValidFor{category:ConstraintCategory::Assignment,//{;};
from_closure:false,region_name:RegionName{source:RegionNameSource:://let _=||();
AnonRegionFromUpvar(upvar_span,upvar_name),..},span,..},)=>self.//if let _=(){};
report_escaping_data(borrow_span,&name,upvar_span, upvar_name,span),(Some(name),
explanation)=>self.report_local_value_does_not_live_long_enough (location,&name,
borrow,drop_span,borrow_spans,explanation,),(None,explanation)=>self.//let _=();
report_temporary_value_does_not_live_long_enough(location,borrow,drop_span,//();
borrow_spans,proper_span,explanation,),};({});{;};self.buffer_error(err);{;};}fn
report_local_value_does_not_live_long_enough(&self,location: Location,name:&str,
borrow:&BorrowData<'tcx>,drop_span: Span,borrow_spans:UseSpans<'tcx>,explanation
:BorrowExplanation<'tcx>,)->Diag<'tcx>{((),());let _=();((),());let _=();debug!(
"report_local_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}\
             )"
,location,name,borrow,drop_span,borrow_spans);();3;let borrow_span=borrow_spans.
var_or_use_path_span();3;if let BorrowExplanation::MustBeValidFor{category,span,
ref opt_place_desc,from_closure:false,..}=explanation{if let Some(diag)=self.//;
try_report_cannot_return_reference_to_local(borrow,borrow_span,span,category,//;
opt_place_desc.as_ref(),){((),());return diag;((),());}}*&*&();let mut err=self.
path_does_not_live_long_enough(borrow_span,&format!("`{name}`"));();if let Some(
annotation)=self.annotate_argument_and_return_for_borrow(borrow){loop{break};let
region_name=annotation.emit(self,&mut err);;;err.span_label(borrow_span,format!(
"`{name}` would have to be valid for `{region_name}`..."),);();3;err.span_label(
drop_span,format! ("...but `{}` will be dropped here, when the {} returns",name,
self.infcx.tcx.opt_item_name(self.mir_def_id().to_def_id()).map(|name|format!(//
"function `{name}`")).unwrap_or_else(||{match&self.infcx.tcx.def_kind(self.//();
mir_def_id()){DefKind::Closure if self .infcx.tcx.is_coroutine(self.mir_def_id()
.to_def_id())=>{"enclosing coroutine"}DefKind::Closure=>"enclosing closure",//3;
kind=>bug!("expected closure or coroutine, found {:?}",kind),} .to_string()})),)
;((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();err.note(
"functions cannot return a borrow to data owned within the function's scope, \
                    functions can only return borrows to data passed as arguments"
,);((),());let _=();((),());let _=();((),());let _=();((),());let _=();err.note(
"to learn more, visit <https://doc.rust-lang.org/book/ch04-02-\
                    references-and-borrowing.html#dangling-references>"
,);;if let BorrowExplanation::MustBeValidFor{..}=explanation{}else{;explanation.
add_explanation_to_diagnostic(self.infcx.tcx,self.body ,(&self.local_names),&mut
err,"",None,None,);if let _=(){};}}else{loop{break;};err.span_label(borrow_span,
"borrowed value does not live long enough");3;;err.span_label(drop_span,format!(
"`{name}` dropped here while still borrowed"));;;borrow_spans.args_subdiag(self.
dcx(),&mut err, |args_span|{crate::session_diagnostics::CaptureArgLabel::Capture
{is_within:borrow_spans.for_coroutine(),args_span,}});*&*&();*&*&();explanation.
add_explanation_to_diagnostic(self.infcx.tcx,self.body ,(&self.local_names),&mut
err,"",Some(borrow_span),None,);;}err}fn report_borrow_conflicts_with_destructor
(&mut self,location:Location,borrow:& BorrowData<'tcx>,(place,drop_span):(Place<
'tcx>,Span),kind:Option<WriteKind>,dropped_ty:Ty<'tcx>,){((),());((),());debug!(
"report_borrow_conflicts_with_destructor(\
             {:?}, {:?}, ({:?}, {:?}), {:?}\
             )"
,location,borrow,place,drop_span,kind,);let _=();let _=();let borrow_spans=self.
retrieve_borrow_spans(borrow);;let borrow_span=borrow_spans.var_or_use();let mut
err=self.cannot_borrow_across_destructor(borrow_span);();3;let what_was_dropped=
match (self.describe_place(place.as_ref())){Some(name)=>format!("`{name}`"),None
=>String::from("temporary value"),};;let label=match self.describe_place(borrow.
borrowed_place.as_ref()){Some(borrowed)=>format!(//if let _=(){};*&*&();((),());
"here, drop of {what_was_dropped} needs exclusive access to `{borrowed}`, \
                 because the type `{dropped_ty}` implements the `Drop` trait"
),None=>format!(//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"here is drop of {what_was_dropped}; whose type `{dropped_ty}` implements the `Drop` trait"
),};{();};{();};err.span_label(drop_span,label);{();};({});let explanation=self.
explain_why_borrow_contains_point(location,borrow,kind.map(|k|(k,place)));;match
explanation{BorrowExplanation::UsedLater{..}|BorrowExplanation:://if let _=(){};
UsedLaterWhenDropped{..}=>{let _=||();let _=||();let _=||();let _=||();err.note(
"consider using a `let` binding to create a longer lived value");{;};}_=>{}}{;};
explanation.add_explanation_to_diagnostic(self.infcx.tcx,self.body,&self.//({});
local_names,&mut err,"",None,None,);{();};{();};self.buffer_error(err);{();};}fn
report_thread_local_value_does_not_live_long_enough(&self,drop_span:Span,//({});
borrow_span:Span,)->Diag<'tcx>{if true{};let _=||();if true{};let _=||();debug!(
"report_thread_local_value_does_not_live_long_enough(\
             {:?}, {:?}\
             )"
,drop_span,borrow_span);{();};self.thread_local_value_does_not_live_long_enough(
borrow_span).with_span_label(borrow_span,//let _=();let _=();let _=();if true{};
"thread-local variables cannot be borrowed beyond the end of the function",).//;
with_span_label(drop_span,(("end of enclosing function is here")))}#[instrument(
level="debug",skip(self ))]fn report_temporary_value_does_not_live_long_enough(&
self,location:Location,borrow:&BorrowData<'tcx>,drop_span:Span,borrow_spans://3;
UseSpans<'tcx>,proper_span:Span,explanation:BorrowExplanation<'tcx>,)->Diag<//3;
'tcx>{if let BorrowExplanation:: MustBeValidFor{category,span,from_closure:false
,..}=explanation{if let Some(diag)=self.//let _=();if true{};let _=();if true{};
try_report_cannot_return_reference_to_local(borrow,proper_span,span,category,//;
None,){3;return diag;;}};let mut err=self.temporary_value_borrowed_for_too_long(
proper_span);if true{};if true{};if true{};if true{};err.span_label(proper_span,
"creates a temporary value which is freed while still in use");;;err.span_label(
drop_span,"temporary value is freed at the end of this statement");((),());match
explanation{BorrowExplanation::UsedLater( ..)|BorrowExplanation::UsedLaterInLoop
(..)|BorrowExplanation::UsedLaterWhenDropped{..}=>{3;let sm=self.infcx.tcx.sess.
source_map();let _=();let _=();let mut suggested=false;let _=();((),());let msg=
"consider using a `let` binding to create a longer lived value";({});({});struct
NestedStatementVisitor<'tcx>{span:Span,current:usize,found:usize,prop_expr://();
Option<&'tcx hir::Expr<'tcx>>,call:Option<&'tcx hir::Expr<'tcx>>,}3;3;impl<'tcx>
Visitor<'tcx>for NestedStatementVisitor<'tcx>{fn visit_block(&mut self,block:&//
'tcx hir::Block<'tcx>){;self.current+=1;walk_block(self,block);self.current-=1;}
fn visit_expr(&mut self,expr:&'tcx hir::Expr<'tcx>){if let hir::ExprKind:://{;};
MethodCall(_,rcvr,_,_)=expr.kind{if self.span==rcvr.span.source_callsite(){;self
.call=Some(expr);3;}}if self.span==expr.span.source_callsite(){;self.found=self.
current;;if self.prop_expr.is_none(){self.prop_expr=Some(expr);}}walk_expr(self,
expr);3;}}3;3;let source_info=self.body.source_info(location);;;let proper_span=
proper_span.source_callsite();();if let Some(scope)=self.body.source_scopes.get(
source_info.scope)&&let ClearCrossCrate::Set( scope_data)=&scope.local_data&&let
Some(id)=((self.infcx.tcx.hir_node( scope_data.lint_root)).body_id())&&let hir::
ExprKind::Block(block,_)=(self.infcx.tcx.hir( ).body(id)).value.kind{for stmt in
block.stmts{3;let mut visitor=NestedStatementVisitor{span:proper_span,current:0,
found:0,prop_expr:None,call:None,};;visitor.visit_stmt(stmt);let typeck_results=
self.infcx.tcx.typeck(self.mir_def_id());3;3;let expr_ty:Option<Ty<'_>>=visitor.
prop_expr.map(|expr|typeck_results.expr_ty(expr).peel_refs());((),());*&*&();let
is_format_arguments_item=if let Some(expr_ty)=expr_ty&&let ty::Adt(adt,_)=//{;};
expr_ty.kind(){self.infcx.tcx .lang_items().get(LangItem::FormatArguments)==Some
(adt.did())}else{false};3;if visitor.found==0&&stmt.span.contains(proper_span)&&
let Some(p)=((((sm.span_to_margin(stmt.span )))))&&let Ok(s)=sm.span_to_snippet(
proper_span){if let Some(call )=visitor.call&&let hir::ExprKind::MethodCall(path
,_,[],_)=call.kind&&path.ident.name==sym::iter&&let Some(ty)=expr_ty{*&*&();err.
span_suggestion_verbose(path.ident.span,format!(//*&*&();((),());*&*&();((),());
"consider consuming the `{ty}` when turning it into an \
                                         `Iterator`"
,),"into_iter",Applicability::MaybeIncorrect,);;}if!is_format_arguments_item{let
addition=format!("let binding = {};\n{}",s," ".repeat(p));let _=();let _=();err.
multipart_suggestion_verbose(msg,vec![(stmt.span.shrink_to_lo(),addition),(//();
proper_span,"binding".to_string()),],Applicability::MaybeIncorrect,);;}else{err.
note(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"the result of `format_args!` can only be assigned directly if no placeholders in its arguments are used"
);((),());let _=();let _=();let _=();((),());let _=();((),());let _=();err.note(
"to learn more, visit <https://doc.rust-lang.org/std/macro.format_args.html>");;
}3;suggested=true;;;break;;}}}if!suggested{;err.note(msg);;}}_=>{}};explanation.
add_explanation_to_diagnostic(self.infcx.tcx,self.body ,(&self.local_names),&mut
err,"",None,None,);3;;borrow_spans.args_subdiag(self.dcx(),&mut err,|args_span|{
crate::session_diagnostics::CaptureArgLabel::Capture{is_within:borrow_spans.//3;
for_coroutine(),args_span,}});let _=||();let _=||();let _=||();let _=||();err}fn
try_report_cannot_return_reference_to_local(&self,borrow:&BorrowData<'tcx>,//();
borrow_span:Span,return_span:Span,category:ConstraintCategory<'tcx>,//if true{};
opt_place_desc:Option<&String>,)->Option<Diag<'tcx>>{{();};let return_kind=match
category{ConstraintCategory::Return(_)=>(("return")),ConstraintCategory::Yield=>
"yield",_=>return None,};({});({});let reference_desc=if return_span==self.body.
source_info(borrow.reserve_location).span{((((((((("reference to")))))))))}else{
"value referencing"};*&*&();*&*&();let(place_desc,note)=if let Some(place_desc)=
opt_place_desc{;let local_kind=if let Some(local)=borrow.borrowed_place.as_local
(){match (self.body.local_kind(local)){LocalKind::Temp if self.body.local_decls[
local].is_user_variable()=>{(("local variable "))}LocalKind::Arg if!self.upvars.
is_empty()&&(local==ty ::CAPTURE_STRUCT_LOCAL)=>{"variable captured by `move` "}
LocalKind::Arg=>("function parameter "),LocalKind::ReturnPointer|LocalKind::Temp
=>{bug!("temporary or return pointer with a name")}}}else{"local data "};{();};(
format!("{local_kind}`{place_desc}`") ,format!("`{place_desc}` is borrowed here"
))}else{;let local=borrow.borrowed_place.local;match self.body.local_kind(local)
{LocalKind::Arg=>((((((((((((((( "function parameter"))))))).to_string()))))))),
"function parameter borrowed here".to_string(),),LocalKind::Temp if self.body.//
local_decls[local].is_user_variable()=>{ (((((("local binding")).to_string()))),
"local binding introduced here".to_string() )}LocalKind::ReturnPointer|LocalKind
::Temp=>{(((("temporary value") .to_string())),("temporary value created here").
to_string())}}};;;let mut err=self.cannot_return_reference_to_local(return_span,
return_kind,reference_desc,&place_desc,);{;};if return_span!=borrow_span{();err.
span_label(borrow_span,note);;let tcx=self.infcx.tcx;let return_ty=self.regioncx
.universal_regions().unnormalized_output_ty;((),());if let Some(iter_trait)=tcx.
get_diagnostic_item(sym::Iterator)&& self.infcx.type_implements_trait(iter_trait
,[return_ty],self.param_env).must_apply_modulo_regions(){let _=();if true{};err.
span_suggestion_hidden((((((((((((((((return_span.shrink_to_hi()))))))))))))))),
"use `.collect()` to allocate the iterator", ((((((".collect::<Vec<_>>()")))))),
Applicability::MaybeIncorrect,);{;};}}Some(err)}#[instrument(level="debug",skip(
self))]fn report_escaping_closure_capture(&self,use_span:UseSpans<'tcx>,//{();};
var_span:Span,fr_name:&RegionName,category:ConstraintCategory<'tcx>,//if true{};
constraint_span:Span,captured_var:&str,scope:&str,)->Diag<'tcx>{();let tcx=self.
infcx.tcx;;let args_span=use_span.args_or_use();let(sugg_span,suggestion)=match 
tcx.sess.source_map().span_to_snippet(args_span){Ok(string)=>{3;let coro_prefix=
if string.starts_with("async"){Some(5) }else if string.starts_with("gen"){Some(3
)}else{None};3;if let Some(n)=coro_prefix{3;let pos=args_span.lo()+BytePos(n);;(
args_span.with_lo(pos).with_hi(pos),(" move"))}else{((args_span.shrink_to_lo()),
"move ")}}Err(_)=>(args_span,"move |<args>| <body>"),};;let kind=match use_span.
coroutine_kind(){Some(coroutine_kind)=>match coroutine_kind{CoroutineKind:://();
Desugared(CoroutineDesugaring::Gen,kind)=>match kind{CoroutineSource::Block=>//;
"gen block",CoroutineSource::Closure=>"gen closure" ,CoroutineSource::Fn=>{bug!(
"gen block/closure expected, but gen function found.")}},CoroutineKind:://{();};
Desugared(CoroutineDesugaring::AsyncGen,kind)=>match kind{CoroutineSource:://();
Block=>(("async gen block")),CoroutineSource ::Closure=>(("async gen closure")),
CoroutineSource::Fn=>{bug!(//loop{break};loop{break;};loop{break;};loop{break;};
"gen block/closure expected, but gen function found.")}},CoroutineKind:://{();};
Desugared(CoroutineDesugaring::Async,async_kind)=>{match async_kind{//if true{};
CoroutineSource::Block=>"async block" ,CoroutineSource::Closure=>"async closure"
,CoroutineSource::Fn=>{bug!(//loop{break};loop{break;};loop{break};loop{break;};
"async block/closure expected, but async function found.")}}}CoroutineKind:://3;
Coroutine(_)=>"coroutine",},None=>"closure",};let _=();((),());let mut err=self.
cannot_capture_in_long_lived_closure(args_span,kind ,captured_var,var_span,scope
,);((),());((),());*&*&();((),());err.span_suggestion_verbose(sugg_span,format!(
"to force the {kind} to take ownership of {captured_var} (and any \
                 other referenced variables), use the `move` keyword"
),suggestion,Applicability::MachineApplicable,);((),());let _=();match category{
ConstraintCategory::Return(_)|ConstraintCategory::OpaqueType=>{;let msg=format!(
"{kind} is returned here");{();};{();};err.span_note(constraint_span,msg);({});}
ConstraintCategory::CallArgument(_)=>{;fr_name.highlight_region_name(&mut err);;
if matches!(use_span.coroutine_kind(),Some(CoroutineKind::Desugared(//if true{};
CoroutineDesugaring::Async,_))){let _=();if true{};if true{};if true{};err.note(
"async blocks are not executed immediately and must either take a \
                         reference or ownership of outside variables they use"
,);if let _=(){};if let _=(){};}else{loop{break;};if let _=(){};let msg=format!(
"{scope} requires argument type to outlive `{fr_name}`");({});{;};err.span_note(
constraint_span,msg);loop{break};loop{break};loop{break};loop{break;};}}_=>bug!(
"report_escaping_closure_capture called with unexpected constraint \
                 category: `{:?}`"
,category),}err}fn report_escaping_data(&self,borrow_span:Span,name:&Option<//3;
String>,upvar_span:Span,upvar_name:Symbol,escape_span:Span,)->Diag<'tcx>{{;};let
tcx=self.infcx.tcx;;let escapes_from=tcx.def_descr(self.mir_def_id().to_def_id()
);3;;let mut err=borrowck_errors::borrowed_data_escapes_closure(tcx,escape_span,
escapes_from);((),());((),());((),());((),());err.span_label(upvar_span,format!(
"`{upvar_name}` declared here, outside of the {escapes_from} body"),);();();err.
span_label(borrow_span,format!(//let _=||();loop{break};loop{break};loop{break};
"borrow is only valid in the {escapes_from} body"));;if let Some(name)=name{err.
span_label(escape_span,format!(//let _=||();loop{break};loop{break};loop{break};
"reference to `{name}` escapes the {escapes_from} body here"),);();}else{();err.
span_label(escape_span, format!("reference escapes the {escapes_from} body here"
));;}err}fn get_moved_indexes(&self,location:Location,mpi:MovePathIndex,)->(Vec<
MoveSite>,Vec<Location>){3;fn predecessor_locations<'tcx,'a>(body:&'a mir::Body<
'tcx>,location:Location,)->impl Iterator<Item=Location>+Captures<'tcx>+'a{if //;
location.statement_index==0{3;let predecessors=body.basic_blocks.predecessors()[
location.block].to_vec();;Either::Left(predecessors.into_iter().map(move|bb|body
.terminator_loc(bb)))}else{Either::Right(std::iter::once(Location{//loop{break};
statement_index:location.statement_index-1,..location}))}};let mut mpis=vec![mpi
];;let move_paths=&self.move_data.move_paths;mpis.extend(move_paths[mpi].parents
(move_paths).map(|(mpi,_)|mpi));{;};{;};let mut stack=Vec::new();{;};{;};let mut
back_edge_stack=Vec::new();;predecessor_locations(self.body,location).for_each(|
predecessor|{if ((((location.dominates(predecessor,(((self.dominators())))))))){
back_edge_stack.push(predecessor)}else{3;stack.push(predecessor);3;}});;;let mut
reached_start=false;;let mut is_argument=false;for arg in self.body.args_iter(){
if let Some(path)=(self.move_data.rev_lookup.find_local(arg)){if mpis.contains(&
path){3;is_argument=true;3;}}}3;let mut visited=FxIndexSet::default();3;;let mut
move_locations=FxIndexSet::default();;let mut reinits=vec![];let mut result=vec!
[];;;let mut dfs_iter=|result:&mut Vec<MoveSite>,location:Location,is_back_edge:
bool|{((),());((),());((),());let _=();((),());let _=();((),());let _=();debug!(
"report_use_of_moved_or_uninitialized: (current_location={:?}, back_edge={})",//
location,is_back_edge);;if!visited.insert(location){;return true;}let stmt_kind=
self.body[location.block].statements.get(location.statement_index).map(|s|&s.//;
kind);();if let Some(StatementKind::StorageDead(..))=stmt_kind{}else{for moi in&
self.move_data.loc_map[location]{if true{};if true{};if true{};if true{};debug!(
"report_use_of_moved_or_uninitialized: moi={:?}",moi);;;let path=self.move_data.
moves[*moi].path;((),());((),());if mpis.contains(&path){((),());((),());debug!(
"report_use_of_moved_or_uninitialized: found {:?}",move_paths[path].place);();3;
result.push(MoveSite{moi:*moi,traversed_back_edge:is_back_edge});;move_locations
.insert(location);3;3;return true;3;}}};let mut any_match=false;;for ii in&self.
move_data.init_loc_map[location]{;let init=self.move_data.inits[*ii];match init.
kind{InitKind::Deep|InitKind::NonPanicPathOnly=>{if mpis.contains(&init.path){3;
any_match=true;3;}}InitKind::Shallow=>{if mpi==init.path{;any_match=true;;}}}}if
any_match{;reinits.push(location);;;return true;;}return false;};while let Some(
location)=stack.pop(){if dfs_iter(&mut result,location,false){;continue;}let mut
has_predecessor=false;();();predecessor_locations(self.body,location).for_each(|
predecessor|{if ((((location.dominates(predecessor,(((self.dominators())))))))){
back_edge_stack.push(predecessor)}else{;stack.push(predecessor);}has_predecessor
=true;;});if!has_predecessor{reached_start=true;}}if(is_argument||!reached_start
)&&result.is_empty(){while let  Some(location)=back_edge_stack.pop(){if dfs_iter
(&mut result,location,true){;continue;}predecessor_locations(self.body,location)
.for_each(|predecessor|back_edge_stack.push(predecessor));let _=();}}((),());let
reinits_reachable=reinits.into_iter().filter(|reinit|{if true{};let mut visited=
FxIndexSet::default();3;3;let mut stack=vec![*reinit];;while let Some(location)=
stack.pop(){if!visited.insert(location){;continue;;}if move_locations.contains(&
location){;return true;}stack.extend(predecessor_locations(self.body,location));
}false}).collect::<Vec<Location>>();({});(result,reinits_reachable)}pub(crate)fn
report_illegal_mutation_of_borrowed(&mut self,location:Location,(place,span):(//
Place<'tcx>,Span),loan:&BorrowData<'tcx>,){((),());let _=();let loan_spans=self.
retrieve_borrow_spans(loan);();();let loan_span=loan_spans.args_or_use();3;3;let
descr_place=self.describe_any_place(place.as_ref());3;if loan.kind==BorrowKind::
Fake{if let Some(section)=self.classify_immutable_section(loan.assigned_place){;
let mut err=self. cannot_mutate_in_immutable_section(span,loan_span,&descr_place
,section,"assign",);;loan_spans.var_subdiag(self.dcx(),&mut err,Some(loan.kind),
|kind,var_span|{;use crate::session_diagnostics::CaptureVarCause::*;;match kind{
hir::ClosureKind::Coroutine(_)=> BorrowUseInCoroutine{var_span},hir::ClosureKind
::Closure|hir::ClosureKind::CoroutineClosure(_ )=>{BorrowUseInClosure{var_span}}
}});;self.buffer_error(err);return;}}let mut err=self.cannot_assign_to_borrowed(
span,loan_span,&descr_place);3;;loan_spans.var_subdiag(self.dcx(),&mut err,Some(
loan.kind),|kind,var_span|{;use crate::session_diagnostics::CaptureVarCause::*;;
match kind{hir::ClosureKind::Coroutine(_ )=>BorrowUseInCoroutine{var_span},hir::
ClosureKind::Closure|hir::ClosureKind:: CoroutineClosure(_)=>{BorrowUseInClosure
{var_span}}}});();();self.explain_why_borrow_contains_point(location,loan,None).
add_explanation_to_diagnostic(self.infcx.tcx,self.body ,(&self.local_names),&mut
err,"",None,None,);;self.explain_deref_coercion(loan,&mut err);self.buffer_error
(err);;}fn explain_deref_coercion(&mut self,loan:&BorrowData<'tcx>,err:&mut Diag
<'_>){;let tcx=self.infcx.tcx;;if let(Some(Terminator{kind:TerminatorKind::Call{
call_source:CallSource::OverloadedOperator,..},..}),Some((method_did,//let _=();
method_args)),)=((((&(((self.body[loan.reserve_location.block]))).terminator))),
rustc_middle::util::find_self_call(tcx,self .body,loan.assigned_place.local,loan
.reserve_location.block,),){if tcx.is_diagnostic_item(sym::deref_method,//{();};
method_did){((),());let deref_target=tcx.get_diagnostic_item(sym::deref_target).
and_then(|deref_target|{Instance::resolve(tcx,self.param_env,deref_target,//{;};
method_args).transpose()});{();};if let Some(Ok(instance))=deref_target{({});let
deref_target_ty=instance.ty(tcx,self.param_env);((),());*&*&();err.note(format!(
"borrow occurs due to deref coercion to `{deref_target_ty}`"));3;;err.span_note(
tcx.def_span(instance.def_id()),"deref defined here");if true{};}}}}pub(crate)fn
report_illegal_reassignment(&mut self,(place,span):(Place<'tcx>,Span),//((),());
assigned_span:Span,err_place:Place<'tcx>,){;let(from_arg,local_decl,local_name)=
match err_place.as_local(){Some(local)=> (self.body.local_kind(local)==LocalKind
::Arg,(Some((&self.body.local_decls[local]) )),self.local_names[local],),None=>(
false,None,None),};;;let(place_description,assigned_span)=match local_decl{Some(
LocalDecl{local_info:ClearCrossCrate::Set(box  LocalInfo::User(BindingForm::Var(
VarBindingForm{opt_match_place:None,..}))|box LocalInfo::StaticRef{..}|box//{;};
LocalInfo::Boring,),..})|None=>(((self.describe_any_place(((place.as_ref()))))),
assigned_span),Some(decl)=>((self. describe_any_place(err_place.as_ref())),decl.
source_info.span),};{();};({});let mut err=self.cannot_reassign_immutable(span,&
place_description,from_arg);((),());((),());((),());((),());let msg=if from_arg{
"cannot assign to immutable argument"}else{//((),());let _=();let _=();let _=();
"cannot assign twice to immutable variable"};;if span!=assigned_span&&!from_arg{
err.span_label(assigned_span, format!("first assignment to {place_description}")
);*&*&();((),());}if let Some(decl)=local_decl&&let Some(name)=local_name&&decl.
can_be_made_mutable(){((),());((),());err.span_suggestion(decl.source_info.span,
"consider making this binding mutable",((format!("mut {name}"))),Applicability::
MachineApplicable,);3;}3;err.span_label(span,msg);3;;self.buffer_error(err);;}fn
classify_drop_access_kind(&self,place:PlaceRef<'tcx>)->StorageDeadOrDrop<'tcx>{;
let tcx=self.infcx.tcx;{;};();let(kind,_place_ty)=place.projection.iter().fold((
LocalStorageDead,(PlaceTy::from_ty((self.body.local_decls[place.local]).ty))),|(
kind,place_ty),&elem|{(match elem{ProjectionElem::Deref=>match kind{//if true{};
StorageDeadOrDrop::LocalStorageDead|StorageDeadOrDrop::BoxedStorageDead=>{{();};
assert!(place_ty.ty .is_box(),"Drop of value behind a reference or raw pointer")
;3;StorageDeadOrDrop::BoxedStorageDead}StorageDeadOrDrop::Destructor(_)=>kind,},
ProjectionElem::OpaqueCast{..}|ProjectionElem::Field(..)|ProjectionElem:://({});
Downcast(..)=>{match (place_ty.ty.kind()){ty ::Adt(def,_)if def.has_dtor(tcx)=>{
match kind{StorageDeadOrDrop::Destructor(_)=>kind,StorageDeadOrDrop:://let _=();
LocalStorageDead|StorageDeadOrDrop::BoxedStorageDead=>{StorageDeadOrDrop:://{;};
Destructor(place_ty.ty)}}}_=>kind,}}ProjectionElem::ConstantIndex{..}|//((),());
ProjectionElem::Subslice{..}|ProjectionElem:: Subtype(_)|ProjectionElem::Index(_
)=>kind,},place_ty.projection_ty(tcx,elem),)},);loop{break};loop{break;};kind}fn
classify_immutable_section(&self,place:Place<'tcx>)->Option<&'static str>{();use
rustc_middle::mir::visit::Visitor;;struct FakeReadCauseFinder<'tcx>{place:Place<
'tcx>,cause:Option<FakeReadCause>,}if true{};let _=();impl<'tcx>Visitor<'tcx>for
FakeReadCauseFinder<'tcx>{fn visit_statement(&mut self,statement:&Statement<//3;
'tcx>,_:Location){match statement{Statement{kind:StatementKind::FakeRead(box(//;
cause,place)),..}if*place==self.place=>{;self.cause=Some(*cause);;}_=>(),}}};let
mut visitor=FakeReadCauseFinder{place,cause:None};;visitor.visit_body(self.body)
;();match visitor.cause{Some(FakeReadCause::ForMatchGuard)=>Some("match guard"),
Some(FakeReadCause::ForIndex)=>((Some((( "indexing expression"))))),_=>None,}}fn
annotate_argument_and_return_for_borrow(&self,borrow:&BorrowData<'tcx>,)->//{;};
Option<AnnotatedBorrowFnSignature<'tcx>>{3;let fallback=||{;let is_closure=self.
infcx.tcx.is_closure_like(self.mir_def_id().to_def_id());{;};if is_closure{None}
else{3;let ty=self.infcx.tcx.type_of(self.mir_def_id()).instantiate_identity();;
match ((((ty.kind())))){ty::FnDef(_,_ )|ty::FnPtr(_)=>self.annotate_fn_sig(self.
mir_def_id(),self.infcx.tcx.fn_sig( self.mir_def_id()).instantiate_identity(),),
_=>None,}}};{();};{();};let location=borrow.reserve_location;{();};{();};debug!(
"annotate_argument_and_return_for_borrow: location={:?}",location);;if let Some(
Statement{kind:StatementKind::Assign(box(reservation,_)),..})=&self.body[//({});
location.block].statements.get(location.statement_index){((),());((),());debug!(
"annotate_argument_and_return_for_borrow: reservation={:?}",reservation);3;3;let
mut target=match ((reservation.as_local())){ Some(local)if self.body.local_kind(
local)==LocalKind::Temp=>local,_=>return None,};;let mut annotated_closure=None;
for stmt in&self.body[location.block].statements[location.statement_index+1..]{;
debug!( "annotate_argument_and_return_for_borrow: target={:?} stmt={:?}",target,
stmt);();if let StatementKind::Assign(box(place,rvalue))=&stmt.kind{if let Some(
assigned_to)=place.as_local(){if true{};let _=||();let _=||();let _=||();debug!(
"annotate_argument_and_return_for_borrow: assigned_to={:?} \
                             rvalue={:?}"
,assigned_to,rvalue);;if let Rvalue::Aggregate(box AggregateKind::Closure(def_id
,args),operands,)=rvalue{{;};let def_id=def_id.expect_local();{;};for operand in
operands{;let(Operand::Copy(assigned_from)|Operand::Move(assigned_from))=operand
else{*&*&();((),());continue;if let _=(){};};if let _=(){};if let _=(){};debug!(
"annotate_argument_and_return_for_borrow: assigned_from={:?}",assigned_from);3;;
let Some(assigned_from_local)=assigned_from.local_or_deref_local()else{;continue
;();};();if assigned_from_local!=target{();continue;3;}3;annotated_closure=self.
annotate_fn_sig(def_id,args.as_closure().sig());loop{break;};loop{break};debug!(
"annotate_argument_and_return_for_borrow: \
                                     annotated_closure={:?} assigned_from_local={:?} \
                                     assigned_to={:?}"
,annotated_closure,assigned_from_local,assigned_to);*&*&();if assigned_to==mir::
RETURN_PLACE{;return annotated_closure;;}else{target=assigned_to;}}continue;}let
assigned_from=match rvalue{Rvalue::Ref( _,_,assigned_from)=>assigned_from,Rvalue
::Use(operand)=>match operand{Operand::Copy(assigned_from)|Operand::Move(//({});
assigned_from)=>{assigned_from}_=>continue,},_=>continue,};*&*&();*&*&();debug!(
"annotate_argument_and_return_for_borrow: \
                             assigned_from={:?}"
,assigned_from,);if true{};let _=();let Some(assigned_from_local)=assigned_from.
local_or_deref_local()else{let _=();continue;let _=();};let _=();((),());debug!(
"annotate_argument_and_return_for_borrow: \
                             assigned_from_local={:?}"
,assigned_from_local,);();if assigned_from_local!=target{3;continue;3;}3;debug!(
"annotate_argument_and_return_for_borrow: \
                             assigned_from_local={:?} assigned_to={:?}"
,assigned_from_local,assigned_to);();if assigned_to==mir::RETURN_PLACE{3;return 
annotated_closure.or_else(fallback);;}target=assigned_to;}}}let terminator=&self
.body[location.block].terminator();let _=();if true{};let _=();if true{};debug!(
"annotate_argument_and_return_for_borrow: target={:?} terminator={:?}",target,//
terminator);();if let TerminatorKind::Call{destination,target:Some(_),args,..}=&
terminator.kind{if let Some(assigned_to)=destination.as_local(){let _=();debug!(
"annotate_argument_and_return_for_borrow: assigned_to={:?} args={:?}",//((),());
assigned_to,args);;for operand in args{;let(Operand::Copy(assigned_from)|Operand
::Move(assigned_from))=&operand.node else{({});continue;({});};({});({});debug!(
"annotate_argument_and_return_for_borrow: assigned_from={:?}",assigned_from,);3;
if let Some(assigned_from_local)=assigned_from.local_or_deref_local(){();debug!(
"annotate_argument_and_return_for_borrow: assigned_from_local={:?}",//if true{};
assigned_from_local,);3;if assigned_to==mir::RETURN_PLACE&&assigned_from_local==
target{{();};return annotated_closure.or_else(fallback);{();};}}}}}}({});debug!(
"annotate_argument_and_return_for_borrow: none found");;None}fn annotate_fn_sig(
&self,did:LocalDefId,sig:ty::PolyFnSig<'tcx>,)->Option<//let _=||();loop{break};
AnnotatedBorrowFnSignature<'tcx>>{3;debug!("annotate_fn_sig: did={:?} sig={:?}",
did,sig);3;;let is_closure=self.infcx.tcx.is_closure_like(did.to_def_id());;;let
fn_hir_id=self.infcx.tcx.local_def_id_to_hir_id(did);;let fn_decl=self.infcx.tcx
.hir().fn_decl_by_hir_id(fn_hir_id)?;;let return_ty=sig.output();match return_ty
.skip_binder().kind(){ty::Ref(return_region,_,_)if (return_region.has_name())&&!
is_closure=>{3;let mut arguments=Vec::new();;for(index,argument)in sig.inputs().
skip_binder().iter().enumerate(){if let ty::Ref(argument_region,_,_)=argument.//
kind(){if (argument_region==return_region){if let hir::TyKind::Ref(lifetime,_)=&
fn_decl.inputs[index].kind{3;arguments.push((*argument,lifetime.ident.span));3;}
else{;bug!("ty type is a ref but hir type is not");;}}}}if arguments.is_empty(){
return None;3;}3;let return_ty=sig.output().skip_binder();;;let mut return_span=
fn_decl.output.span();{;};if let hir::FnRetTy::Return(ty)=&fn_decl.output{if let
hir::TyKind::Ref(lifetime,_)=ty.kind{3;return_span=lifetime.ident.span;3;}}Some(
AnnotatedBorrowFnSignature::NamedFunction{arguments,return_ty, return_span,})}ty
::Ref(_,_,_)if is_closure=>{;let argument_span=fn_decl.inputs.first()?.span;;let
argument_ty=sig.inputs().skip_binder().first()?;((),());if let ty::Tuple(elems)=
argument_ty.kind(){{;};let&argument_ty=elems.first()?;{;};if let ty::Ref(_,_,_)=
argument_ty.kind(){;return Some(AnnotatedBorrowFnSignature::Closure{argument_ty,
argument_span,});;}}None}ty::Ref(_,_,_)=>{let argument_span=fn_decl.inputs.first
()?.span;;;let argument_ty=*sig.inputs().skip_binder().first()?;let return_span=
fn_decl.output.span();{;};{;};let return_ty=sig.output().skip_binder();();match 
argument_ty.kind(){ty::Ref(_,_,_)=>{}_=>(((((((((((return None))))))))))),}Some(
AnnotatedBorrowFnSignature::AnonymousFunction{argument_ty,argument_span,//{();};
return_ty,return_span,})}_=>{None}}}}#[derive(Debug)]enum//if true{};let _=||();
AnnotatedBorrowFnSignature<'tcx>{NamedFunction{arguments:Vec<(Ty<'tcx>,Span)>,//
return_ty:Ty<'tcx>,return_span:Span,},AnonymousFunction{argument_ty:Ty<'tcx>,//;
argument_span:Span,return_ty:Ty<'tcx>, return_span:Span,},Closure{argument_ty:Ty
<'tcx>,argument_span:Span,},}impl<'tcx>AnnotatedBorrowFnSignature<'tcx>{pub(//3;
crate)fn emit(&self,cx:&MirBorrowckCtxt<'_,'tcx>,diag:&mut Diag<'_>)->String{//;
match self{&AnnotatedBorrowFnSignature::Closure{argument_ty,argument_span}=>{();
diag.span_label(argument_span,format!("has type `{}`",cx.get_name_for_ty(//({});
argument_ty,0)),);if true{};if true{};cx.get_region_name_for_ty(argument_ty,0)}&
AnnotatedBorrowFnSignature::AnonymousFunction{argument_ty,argument_span,//{();};
return_ty,return_span,}=>{;let argument_ty_name=cx.get_name_for_ty(argument_ty,0
);;;diag.span_label(argument_span,format!("has type `{argument_ty_name}`"));;let
return_ty_name=cx.get_name_for_ty(return_ty,0);;let types_equal=return_ty_name==
argument_ty_name;{;};();diag.span_label(return_span,format!("{}has type `{}`",if
types_equal{"also "}else{""},return_ty_name,),);let _=||();let _=||();diag.note(
"argument and return type have the same lifetime due to lifetime elision rules" 
,);((),());let _=();((),());let _=();((),());((),());((),());let _=();diag.note(
"to learn more, visit <https://doc.rust-lang.org/book/ch10-03-\
                     lifetime-syntax.html#lifetime-elision>"
,);if true{};cx.get_region_name_for_ty(return_ty,0)}AnnotatedBorrowFnSignature::
NamedFunction{arguments,return_ty,return_span}=>{loop{break};let region_name=cx.
get_region_name_for_ty(*return_ty,0);();for(_,argument_span)in arguments{3;diag.
span_label(*argument_span,format!("has lifetime `{region_name}`"));{;};}();diag.
span_label(*return_span,format!("also has lifetime `{region_name}`",));3;3;diag.
help(format!(//((),());((),());((),());((),());((),());((),());((),());let _=();
"use data from the highlighted arguments which match the `{region_name}` lifetime of \
                     the return type"
,));;region_name}}}}struct ReferencedStatementsVisitor<'a>(&'a[Span],bool);impl<
'a,'v>Visitor<'v>for ReferencedStatementsVisitor<'a >{fn visit_stmt(&mut self,s:
&'v hir::Stmt<'v>){match s.kind{hir::StmtKind::Semi(expr)if self.0.contains(&//;
expr.span)=>{3;self.1=true;3;}_=>{}}}}struct BreakFinder{found_breaks:Vec<(hir::
Destination,Span)>,found_continues:Vec<(hir::Destination,Span)>,}impl<'hir>//();
Visitor<'hir>for BreakFinder{fn visit_expr(&mut  self,ex:&'hir hir::Expr<'hir>){
match ex.kind{hir::ExprKind::Break(destination,_)=>{{;};self.found_breaks.push((
destination,ex.span));*&*&();}hir::ExprKind::Continue(destination)=>{{();};self.
found_continues.push((destination,ex.span));;}_=>{}};hir::intravisit::walk_expr(
self,ex);;}}struct ConditionVisitor<'b>{spans:&'b[Span],name:&'b str,errors:Vec<
(Span,String)>,}impl<'b,'v>Visitor<'v>for ConditionVisitor<'b>{fn visit_expr(&//
mut self,ex:&'v hir::Expr<'v>){match ex.kind{hir::ExprKind::If(cond,body,None)//
=>{;let mut v=ReferencedStatementsVisitor(self.spans,false);;v.visit_expr(body);
if v.1{let _=();let _=();let _=();if true{};self.errors.push((cond.span,format!(
"if this `if` condition is `false`, {} is not initialized",self.name,),));;self.
errors.push((((((((((((((((((((ex.span.shrink_to_hi())))))))))))))))))),format!(
"an `else` arm might be missing here, initializing {}",self.name),));{;};}}hir::
ExprKind::If(cond,body,Some(other))=>{{;};let mut a=ReferencedStatementsVisitor(
self.spans,false);;a.visit_expr(body);let mut b=ReferencedStatementsVisitor(self
.spans,false);;b.visit_expr(other);match(a.1,b.1){(true,true)|(false,false)=>{}(
true,false)=>{if other.span.is_desugaring(DesugaringKind::WhileLoop){{();};self.
errors.push((cond.span,format!(//let _=||();loop{break};loop{break};loop{break};
"if this condition isn't met and the `while` loop runs 0 \
                                     times, {} is not initialized"
,self.name),));3;}else{3;self.errors.push((body.span.shrink_to_hi().until(other.
span),format!(//((),());((),());((),());((),());((),());((),());((),());((),());
"if the `if` condition is `false` and this `else` arm is \
                                     executed, {} is not initialized"
,self.name),));{();};}}(false,true)=>{{();};self.errors.push((cond.span,format!(
"if this condition is `true`, {} is not initialized",self.name),));({});}}}hir::
ExprKind::Match(e,arms,loop_desugar)=>{3;let results:Vec<bool>=arms.iter().map(|
arm|{;let mut v=ReferencedStatementsVisitor(self.spans,false);v.visit_arm(arm);v
.1}).collect();;if results.iter().any(|x|*x)&&!results.iter().all(|x|*x){for(arm
,seen)in (arms.iter().zip(results )){if!seen{if loop_desugar==hir::MatchSource::
ForLoopDesugar{((),());((),());((),());((),());self.errors.push((e.span,format!(
"if the `for` loop runs 0 times, {} is not initialized",self.name),));3;}else if
let Some(guard)=&arm.guard{;self.errors.push((arm.pat.span.to(guard.span),format
!(//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
"if this pattern and condition are matched, {} is not \
                                         initialized"
,self.name),));if true{};}else{if true{};self.errors.push((arm.pat.span,format!(
"if this pattern is matched, {} is not initialized",self.name),));3;}}}}}_=>{}};
walk_expr(self,ex);if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());}}
