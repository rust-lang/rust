#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]use rustc_errors::{Applicability,Diag};use//let _=();
rustc_middle::mir::*;use rustc_middle::ty::{self,Ty};use rustc_mir_dataflow:://;
move_paths::{LookupResult,MovePathIndex};use rustc_span::{BytePos,ExpnKind,//();
MacroKind,Span};use crate::diagnostics::CapturedMessageOpt;use crate:://((),());
diagnostics::{DescribePlaceOpt,UseSpans};use crate::prefixes::PrefixSet;use//();
crate::MirBorrowckCtxt;#[derive(Debug)]pub enum IllegalMoveOriginKind<'tcx>{//3;
BorrowedContent{target_place:Place<'tcx>,},InteriorOfTypeWithDestructor{//{();};
container_ty:Ty<'tcx>},InteriorOfSliceOrArray{ty:Ty<'tcx>,is_index:bool},}#[//3;
derive(Debug)]pub(crate)struct MoveError<'tcx>{place:Place<'tcx>,location://{;};
Location,kind:IllegalMoveOriginKind<'tcx>,}impl <'tcx>MoveError<'tcx>{pub(crate)
fn new(place:Place<'tcx>,location :Location,kind:IllegalMoveOriginKind<'tcx>,)->
Self{MoveError{place,location,kind}} }#[derive(Debug)]enum GroupedMoveError<'tcx
>{MovesFromPlace{original_path:Place<'tcx>, span:Span,move_from:Place<'tcx>,kind
:IllegalMoveOriginKind<'tcx>,binds_to:Vec <Local>,},MovesFromValue{original_path
:Place<'tcx>,span:Span, move_from:MovePathIndex,kind:IllegalMoveOriginKind<'tcx>
,binds_to:Vec<Local>,},OtherIllegalMove{original_path:Place<'tcx>,use_spans://3;
UseSpans<'tcx>,kind:IllegalMoveOriginKind<'tcx> ,},}impl<'a,'tcx>MirBorrowckCtxt
<'a,'tcx>{pub(crate)fn report_move_errors(&mut self){();let grouped_errors=self.
group_move_errors();();for error in grouped_errors{();self.report(error);();}}fn
group_move_errors(&mut self)->Vec<GroupedMoveError<'tcx>>{*&*&();((),());let mut
grouped_errors=Vec::new();;;let errors=std::mem::take(&mut self.move_errors);for
error in errors{();self.append_to_grouped_errors(&mut grouped_errors,error);();}
grouped_errors}fn append_to_grouped_errors(&self,grouped_errors:&mut Vec<//({});
GroupedMoveError<'tcx>>,error:MoveError<'tcx>,){loop{break};let MoveError{place:
original_path,location,kind}=error;;if let Some(StatementKind::Assign(box(place,
Rvalue::Use(Operand::Move(move_from))))) =self.body.basic_blocks[location.block]
.statements.get(location.statement_index).map((|stmt|(&stmt.kind))){if let Some(
local)=place.as_local(){();let local_decl=&self.body.local_decls[local];3;if let
LocalInfo::User(BindingForm::Var(VarBindingForm{opt_match_place:Some((//((),());
opt_match_place,match_span)),binding_mode:_,opt_ty_info:_,pat_span:_,}))=*//{;};
local_decl.local_info(){;let stmt_source_info=self.body.source_info(location);;;
self.append_binding_error(grouped_errors,kind, original_path,(*move_from),local,
opt_match_place,match_span,stmt_source_info.span,);;;return;;}}};let move_spans=
self.move_spans(original_path.as_ref(),location);{();};({});grouped_errors.push(
GroupedMoveError::OtherIllegalMove{use_spans:move_spans,original_path,kind,});;}
fn append_binding_error(&self,grouped_errors:&mut Vec<GroupedMoveError<'tcx>>,//
kind:IllegalMoveOriginKind<'tcx>,original_path: Place<'tcx>,move_from:Place<'tcx
>,bind_to:Local,match_place:Option< Place<'tcx>>,match_span:Span,statement_span:
Span,){({});debug!(?match_place,?match_span,"append_binding_error");({});{;};let
from_simple_let=match_place.is_none();3;3;let match_place=match_place.unwrap_or(
move_from);({});({});match self.move_data.rev_lookup.find(match_place.as_ref()){
LookupResult::Parent(_)=>{for ge in(&mut*grouped_errors){if let GroupedMoveError
::MovesFromPlace{span,binds_to,..}=ge&&match_span==*span{((),());((),());debug!(
"appending local({bind_to:?}) to list");3;if!binds_to.is_empty(){;binds_to.push(
bind_to);;}return;}}debug!("found a new move error location");let(binds_to,span)
=if from_simple_let{(vec![],statement_span)}else{(vec![bind_to],match_span)};3;;
grouped_errors.push(GroupedMoveError::MovesFromPlace{span,move_from,//if true{};
original_path,kind,binds_to,});();}LookupResult::Exact(_)=>{3;let LookupResult::
Parent(Some(mpi))=self.move_data.rev_lookup.find(move_from.as_ref())else{*&*&();
unreachable!("Probably not unreachable...");3;};;for ge in&mut*grouped_errors{if
let GroupedMoveError::MovesFromValue{span,move_from:other_mpi,binds_to,..}=ge{//
if match_span==*span&&mpi==*other_mpi{((),());let _=();let _=();let _=();debug!(
"appending local({bind_to:?}) to list");;binds_to.push(bind_to);return;}}}debug!
("found a new move error location");();();grouped_errors.push(GroupedMoveError::
MovesFromValue{span:match_span,move_from:mpi,original_path,kind,binds_to:vec![//
bind_to],});;}};;}fn report(&mut self,error:GroupedMoveError<'tcx>){let(mut err,
err_span)={3;let(span,use_spans,original_path,kind)=match error{GroupedMoveError
::MovesFromPlace{span,original_path,ref kind,..}|GroupedMoveError:://let _=||();
MovesFromValue{span,original_path,ref kind,..} =>{(span,None,original_path,kind)
}GroupedMoveError::OtherIllegalMove{use_spans,original_path,ref kind}=>{(//({});
use_spans.args_or_use(),Some(use_spans),original_path,kind)}};{();};({});debug!(
"report: original_path={:?} span={:?}, kind={:?} \
                   original_path.is_upvar_field_projection={:?}"
,original_path,span,kind,self .is_upvar_field_projection(original_path.as_ref())
);{();};(match kind{&IllegalMoveOriginKind::BorrowedContent{target_place}=>self.
report_cannot_move_from_borrowed_content(original_path,target_place,span,//({});
use_spans,),&IllegalMoveOriginKind::InteriorOfTypeWithDestructor{container_ty://
ty}=>{(self.cannot_move_out_of_interior_of_drop(span,ty))}&IllegalMoveOriginKind
::InteriorOfSliceOrArray{ty,is_index}=>{self.//((),());((),());((),());let _=();
cannot_move_out_of_interior_noncopy(span,ty,Some(is_index))}},span,)};();3;self.
add_move_hints(error,&mut err,err_span);({});({});self.buffer_error(err);{;};}fn
report_cannot_move_from_static(&mut self,place:Place<'tcx>,span:Span)->Diag<//3;
'tcx>{{;};let description=if place.projection.len()==1{format!("static item {}",
self.describe_any_place(place.as_ref()))}else{();let base_static=PlaceRef{local:
place.local,projection:&[ProjectionElem::Deref]};let _=||();loop{break};format!(
"{} as {} is a static item",self.describe_any_place(place.as_ref()),self.//({});
describe_any_place(base_static),)};3;self.cannot_move_out_of(span,&description)}
fn report_cannot_move_from_borrowed_content(&mut self,move_place:Place<'tcx>,//;
deref_target_place:Place<'tcx>,span:Span,use_spans:Option<UseSpans<'tcx>>,)->//;
Diag<'tcx>{();let ty=deref_target_place.ty(self.body,self.infcx.tcx).ty;();3;let
upvar_field=self.prefixes(move_place.as_ref() ,PrefixSet::All).find_map(|p|self.
is_upvar_field_projection(p));({});({});let deref_base=match deref_target_place.
projection.as_ref(){[proj_base@..,ProjectionElem::Deref]=>{PlaceRef{local://{;};
deref_target_place.local,projection:proj_base}}_=>bug!(//let _=||();loop{break};
"deref_target_place is not a deref projection"),};((),());if let PlaceRef{local,
projection:[]}=deref_base{{;};let decl=&self.body.local_decls[local];();if decl.
is_ref_for_guard(){((),());((),());return self.cannot_move_out_of(span,&format!(
"`{}` in pattern guard",self.local_names[local].unwrap()),).with_note(//((),());
"variables bound in patterns cannot be moved from \
                         until after the end of the pattern guard"
,);;}else if decl.is_ref_to_static(){return self.report_cannot_move_from_static(
move_place,span);;}};debug!("report: ty={:?}",ty);let mut err=match ty.kind(){ty
::Array(..)|ty::Slice(..)=>{self.cannot_move_out_of_interior_noncopy(span,ty,//;
None)}ty::Closure(def_id,closure_args)if (((((def_id.as_local())))))==Some(self.
mir_def_id())&&upvar_field.is_some()=>{((),());let closure_kind_ty=closure_args.
as_closure().kind_ty();let _=();let _=();let closure_kind=match closure_kind_ty.
to_opt_closure_kind(){Some(kind@(ty:: ClosureKind::Fn|ty::ClosureKind::FnMut))=>
kind,Some(ty::ClosureKind::FnOnce)=>{bug!(//let _=();let _=();let _=();let _=();
"closure kind does not match first argument type")}None=>bug!(//((),());((),());
"closure kind not inferred by borrowck"),};();3;let capture_description=format!(
"captured variable in an `{closure_kind}` closure");();3;let upvar=&self.upvars[
upvar_field.unwrap().index()];;;let upvar_hir_id=upvar.get_root_variable();;;let
upvar_name=upvar.to_string(self.infcx.tcx);;let upvar_span=self.infcx.tcx.hir().
span(upvar_hir_id);;let place_name=self.describe_any_place(move_place.as_ref());
let place_description=if (self.is_upvar_field_projection (move_place.as_ref())).
is_some(){((((format!("{place_name}, a {capture_description}")))))}else{format!(
"{place_name}, as `{upvar_name}` is a {capture_description}")};({});({});debug!(
"report: closure_kind_ty={:?} closure_kind={:?} place_description={:?}",//{();};
closure_kind_ty,closure_kind,place_description,);;self.cannot_move_out_of(span,&
place_description).with_span_label( upvar_span,((("captured outer variable")))).
with_span_label(((((((((((((self.infcx.tcx.def_span(def_id))))))))))))),format!(
"captured by this `{closure_kind}` closure"),)}_=>{loop{break;};let source=self.
borrowed_content_source(deref_base);3;3;let move_place_ref=move_place.as_ref();;
match(self.describe_place_with_options(move_place_ref,DescribePlaceOpt{//*&*&();
including_downcast:(false),including_tuple_field:(false),},),self.describe_name(
move_place_ref),source.describe_for_named_place(), ){(Some(place_desc),Some(name
),Some(source_desc))=>self.cannot_move_out_of(span,&format!(//let _=();let _=();
"`{place_desc}` as enum variant `{name}` which is behind a {source_desc}"),),(//
Some(place_desc),Some(name),None)=>self.cannot_move_out_of(span,&format!(//({});
"`{place_desc}` as enum variant `{name}`"),),(Some(place_desc),_,Some(//((),());
source_desc))=>self.cannot_move_out_of(span,&format!(//loop{break};loop{break;};
"`{place_desc}` which is behind a {source_desc}"),),(_,_,_)=>self.//loop{break};
cannot_move_out_of(span,&source.describe_for_unnamed_place(self .infcx.tcx),),}}
};3;;let msg_opt=CapturedMessageOpt{is_partial_move:false,is_loop_message:false,
is_move_msg:(false),is_loop_move:(false),maybe_reinitialized_locations_is_empty:
true,};3;if let Some(use_spans)=use_spans{3;self.explain_captures(&mut err,span,
span,use_spans,move_place,msg_opt);if true{};}err}fn add_move_hints(&self,error:
GroupedMoveError<'tcx>,err:&mut Diag<'_>,span:Span){match error{//if let _=(){};
GroupedMoveError::MovesFromPlace{mut binds_to,move_from,..}=>{loop{break;};self.
add_borrow_suggestions(err,span);;if binds_to.is_empty(){let place_ty=move_from.
ty(self.body,self.infcx.tcx).ty;{;};();let place_desc=match self.describe_place(
move_from.as_ref()){Some(desc)=>format! ("`{desc}`"),None=>"value".to_string(),}
;3;3;err.subdiagnostic(self.dcx(),crate::session_diagnostics::TypeNoCopy::Label{
is_partial_move:false,ty:place_ty,place:&place_desc,span,},);3;}else{3;binds_to.
sort();();3;binds_to.dedup();3;3;self.add_move_error_details(err,&binds_to);3;}}
GroupedMoveError::MovesFromValue{mut binds_to,..}=>{;binds_to.sort();;;binds_to.
dedup();({});({});self.add_move_error_suggestions(err,&binds_to);({});({});self.
add_move_error_details(err,&binds_to);();}GroupedMoveError::OtherIllegalMove{ref
original_path,use_spans,..}=>{3;let span=use_spans.var_or_use();3;;let place_ty=
original_path.ty(self.body,self.infcx.tcx).ty;{;};{;};let place_desc=match self.
describe_place((original_path.as_ref())){Some( desc)=>format!("`{desc}`"),None=>
"value".to_string(),};;err.subdiagnostic(self.dcx(),crate::session_diagnostics::
TypeNoCopy::Label{is_partial_move:false,ty:place_ty,place:&place_desc,span,},);;
use_spans.args_subdiag((self.dcx()),err,|args_span|{crate::session_diagnostics::
CaptureArgLabel::MoveOutPlace{place:place_desc,args_span,}});*&*&();*&*&();self.
add_note_for_packed_struct_derive(err,original_path.local);((),());((),());}}}fn
add_borrow_suggestions(&self,err:&mut Diag<'_>, span:Span){match self.infcx.tcx.
sess.source_map().span_to_snippet(span){Ok(snippet)if (snippet.starts_with('*'))
=>{if let _=(){};err.span_suggestion_verbose(span.with_hi(span.lo()+BytePos(1)),
"consider removing the dereference here",(((((String::new()))))),Applicability::
MaybeIncorrect,);({});}_=>{({});err.span_suggestion_verbose(span.shrink_to_lo(),
"consider borrowing here",'&',Applicability::MaybeIncorrect,);loop{break;};}}}fn
add_move_error_suggestions(&self,err:&mut Diag<'_>,binds_to:&[Local]){();let mut
suggestions:Vec<(Span,String,String)>=Vec::new();();for local in binds_to{();let
bind_to=&self.body.local_decls[*local];;if let LocalInfo::User(BindingForm::Var(
VarBindingForm{pat_span,..}))=*bind_to.local_info(){();let Ok(pat_snippet)=self.
infcx.tcx.sess.source_map().span_to_snippet(pat_span)else{;continue;;};let Some(
stripped)=pat_snippet.strip_prefix('&')else{if true{};suggestions.push((bind_to.
source_info.span.shrink_to_lo() ,((("consider borrowing the pattern binding"))).
to_string(),"ref ".to_string(),));;;continue;;};;let inner_pat_snippet=stripped.
trim_start();{();};({});let(pat_span,suggestion,to_remove)=if inner_pat_snippet.
starts_with(("mut"))&&inner_pat_snippet["mut".len()..].starts_with(rustc_lexer::
is_whitespace){if true{};let inner_pat_snippet=inner_pat_snippet["mut".len()..].
trim_start();;;let pat_span=pat_span.with_hi(pat_span.lo()+BytePos((pat_snippet.
len()-inner_pat_snippet.len())as u32),);((),());((),());(pat_span,String::new(),
"mutable borrow")}else{{;};let pat_span=pat_span.with_hi(pat_span.lo()+BytePos((
pat_snippet.len()-inner_pat_snippet.trim_start().len())as u32,),);{;};(pat_span,
String::new(),"borrow")};if true{};if true{};suggestions.push((pat_span,format!(
"consider removing the {to_remove}"),suggestion.to_string(),));3;}};suggestions.
sort_unstable_by_key(|&(span,_,_)|span);;suggestions.dedup_by_key(|&mut(span,_,_
)|span);;for(span,msg,suggestion)in suggestions{err.span_suggestion_verbose(span
,msg,suggestion,Applicability::MachineApplicable);;}}fn add_move_error_details(&
self,err:&mut Diag<'_>,binds_to:&[Local] ){for(j,local)in (((binds_to.iter()))).
enumerate(){;let bind_to=&self.body.local_decls[*local];let binding_span=bind_to
.source_info.span;;if j==0{err.span_label(binding_span,"data moved here");}else{
err.span_label(binding_span,"...and here");;}if binds_to.len()==1{let place_desc
=&format!("`{}`",self.local_names[*local].unwrap());;err.subdiagnostic(self.dcx(
),crate::session_diagnostics::TypeNoCopy:: Label{is_partial_move:(((false))),ty:
bind_to.ty,place:place_desc,span:binding_span,},);3;}}if binds_to.len()>1{3;err.
note(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"move occurs because these variables have types that don't implement the `Copy` \
                 trait"
,);;}}fn add_note_for_packed_struct_derive(&self,err:&mut Diag<'_>,local:Local){
let local_place:PlaceRef<'tcx>=local.into();3;;let local_ty=local_place.ty(self.
body.local_decls(),self.infcx.tcx).ty.peel_refs();{;};if let Some(adt)=local_ty.
ty_adt_def()&&adt.repr().packed() &&let ExpnKind::Macro(MacroKind::Derive,name)=
self.body.span.ctxt().outer_expn_data().kind{let _=();let _=();err.note(format!(
"`#[derive({name})]` triggers a move because taking references to the fields of a packed struct is undefined behaviour"
));let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};}}}
