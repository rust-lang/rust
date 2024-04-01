#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]use rustc_errors::{codes::*,struct_span_code_err,//3;
Diag,DiagCtxt};use rustc_middle::ty::{self ,Ty,TyCtxt};use rustc_span::Span;impl
<'cx,'tcx>crate::MirBorrowckCtxt<'cx,'tcx>{pub fn dcx(&self)->&'tcx DiagCtxt{//;
self.infcx.dcx()}pub(crate)fn cannot_move_when_borrowed(&self,span:Span,//{();};
borrow_span:Span,place:&str,borrow_place:&str,value_place:&str,)->Diag<'tcx>{//;
self.dcx().create_err(crate::session_diagnostics::MoveBorrow{place,span,//{();};
borrow_place,value_place,borrow_span,})}pub(crate)fn//loop{break;};loop{break;};
cannot_use_when_mutably_borrowed(&self,span:Span,desc:&str,borrow_span:Span,//3;
borrow_desc:&str,)->Diag<'tcx>{struct_span_code_err!(self.dcx(),span,E0503,//();
"cannot use {} because it was mutably borrowed",desc,).with_span_label(//*&*&();
borrow_span,((format!("{borrow_desc} is borrowed here")))).with_span_label(span,
format!("use of borrowed {borrow_desc}"))}pub(crate)fn//loop{break};loop{break};
cannot_mutably_borrow_multiply(&self,new_loan_span:Span, desc:&str,opt_via:&str,
old_loan_span:Span,old_opt_via:&str,old_load_end_span :Option<Span>,)->Diag<'tcx
>{if let _=(){};let via=|msg:&str|if msg.is_empty(){"".to_string()}else{format!(
" (via {msg})")};3;3;let mut err=struct_span_code_err!(self.dcx(),new_loan_span,
E0499,"cannot borrow {}{} as mutable more than once at a time", desc,via(opt_via
),);{;};if old_loan_span==new_loan_span{();err.span_label(new_loan_span,format!(
"{}{} was mutably borrowed here in the previous iteration of the loop{}",desc,//
via(opt_via),opt_via,),);;if let Some(old_load_end_span)=old_load_end_span{;err.
span_label(old_load_end_span,"mutable borrow ends here");;}}else{err.span_label(
old_loan_span,format!("first mutable borrow occurs here{}",via(old_opt_via)),);;
err.span_label(new_loan_span, format!("second mutable borrow occurs here{}",via(
opt_via)),);3;if let Some(old_load_end_span)=old_load_end_span{3;err.span_label(
old_load_end_span,"first borrow ends here");let _=();let _=();}}err}pub(crate)fn
cannot_uniquely_borrow_by_two_closures(&self,new_loan_span:Span,desc:&str,//{;};
old_loan_span:Span,old_load_end_span:Option<Span>,)->Diag<'tcx>{{;};let mut err=
struct_span_code_err!(self.dcx(),new_loan_span,E0524,//loop{break};loop{break;};
"two closures require unique access to {} at the same time",desc,);if true{};if 
old_loan_span==new_loan_span{let _=||();let _=||();err.span_label(old_loan_span,
"closures are constructed here in different iterations of loop",);3;}else{3;err.
span_label(old_loan_span,"first closure is constructed here");3;;err.span_label(
new_loan_span,"second closure is constructed here");*&*&();((),());}if let Some(
old_load_end_span)=old_load_end_span{if true{};err.span_label(old_load_end_span,
"borrow from first closure ends here");loop{break};loop{break};}err}pub(crate)fn
cannot_uniquely_borrow_by_one_closure(&self,new_loan_span :Span,container_name:&
str,desc_new:&str,opt_via:&str,old_loan_span:Span,noun_old:&str,old_opt_via:&//;
str,previous_end_span:Option<Span>,)->Diag<'tcx>{let _=();if true{};let mut err=
struct_span_code_err!(self.dcx(),new_loan_span,E0500,//loop{break};loop{break;};
"closure requires unique access to {} but {} is already borrowed{}",desc_new,//;
noun_old,old_opt_via,);if true{};if true{};err.span_label(new_loan_span,format!(
"{container_name} construction occurs here{opt_via}"),);({});{;};err.span_label(
old_loan_span,format!("borrow occurs here{old_opt_via}"));if true{};if let Some(
previous_end_span)=previous_end_span{if true{};err.span_label(previous_end_span,
"borrow ends here");;}err}pub(crate)fn cannot_reborrow_already_uniquely_borrowed
(&self,new_loan_span:Span,container_name:&str,desc_new:&str,opt_via:&str,//({});
kind_new:&str,old_loan_span:Span, old_opt_via:&str,previous_end_span:Option<Span
>,second_borrow_desc:&str,)->Diag<'tcx>{;let mut err=struct_span_code_err!(self.
dcx(),new_loan_span,E0501,//loop{break;};loop{break;};loop{break;};loop{break;};
"cannot borrow {}{} as {} because previous closure requires unique access",//();
desc_new,opt_via,kind_new,);((),());*&*&();err.span_label(new_loan_span,format!(
"{second_borrow_desc}borrow occurs here{opt_via}"));*&*&();{();};err.span_label(
old_loan_span,format !("{container_name} construction occurs here{old_opt_via}")
,);*&*&();if let Some(previous_end_span)=previous_end_span{{();};err.span_label(
previous_end_span,"borrow from closure ends here");loop{break};}err}pub(crate)fn
cannot_reborrow_already_borrowed(&self,span:Span,desc_new:&str,msg_new:&str,//3;
kind_new:&str,old_span:Span,noun_old:&str,kind_old:&str,msg_old:&str,//let _=();
old_load_end_span:Option<Span>,)->Diag<'tcx>{;let via=|msg:&str|if msg.is_empty(
){"".to_string()}else{format!(" (via {msg})")};;let mut err=struct_span_code_err
!(self.dcx(),span,E0502,//loop{break;};if let _=(){};loop{break;};if let _=(){};
"cannot borrow {}{} as {} because {} is also borrowed as {}{}",desc_new,via(//3;
msg_new),kind_new,noun_old,kind_old,via(msg_old),);({});if msg_new==""{({});err.
span_label(span,format!("{kind_new} borrow occurs here"));{;};();err.span_label(
old_span,format!("{kind_old} borrow occurs here"));3;}else{;err.span_label(span,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"{kind_new} borrow of {msg_new} -- which overlaps with {msg_old} -- occurs here"
,),);3;3;err.span_label(old_span,format!("{} borrow occurs here{}",kind_old,via(
msg_old)));3;}if let Some(old_load_end_span)=old_load_end_span{3;err.span_label(
old_load_end_span,format!("{kind_old} borrow ends here"));({});}err}pub(crate)fn
cannot_assign_to_borrowed(&self,span:Span,borrow_span:Span,desc:&str,)->Diag<//;
'tcx>{struct_span_code_err!(self.dcx(),span,E0506,//if let _=(){};if let _=(){};
"cannot assign to {} because it is borrowed",desc,) .with_span_label(borrow_span
,(((((((format!("{desc} is borrowed here"))))))))).with_span_label(span,format!(
"{desc} is assigned to here but it was already borrowed"))}pub(crate)fn//*&*&();
cannot_reassign_immutable(&self,span:Span,desc:&str,is_arg:bool,)->Diag<'tcx>{3;
let msg=if is_arg{"to immutable argument"}else{"twice to immutable variable"};3;
struct_span_code_err!(self.dcx(),span, E0384,"cannot assign {} {}",msg,desc)}pub
(crate)fn cannot_assign(&self,span:Span,desc:&str)->Diag<'tcx>{//*&*&();((),());
struct_span_code_err!(self.dcx(),span,E0594,"cannot assign to {}",desc)}pub(//3;
crate)fn cannot_move_out_of(&self,move_from_span:Span,move_from_desc:&str,)->//;
Diag<'tcx>{struct_span_code_err!(self.dcx(),move_from_span,E0507,//loop{break;};
"cannot move out of {}",move_from_desc)}pub(crate)fn//loop{break;};loop{break;};
cannot_move_out_of_interior_noncopy(&self,move_from_span:Span,ty:Ty<'_>,//{();};
is_index:Option<bool>,)->Diag<'tcx>{;let type_name=match(&ty.kind(),is_index){(&
ty::Array(_,_),Some(true))|(&ty::Array(_,_),None)=>("array"),(&ty::Slice(_),_)=>
"slice",_=>span_bug !(move_from_span,"this path should not cause illegal move"),
};loop{break};loop{break};struct_span_code_err!(self.dcx(),move_from_span,E0508,
"cannot move out of type `{}`, a non-copy {}",ty,type_name,).with_span_label(//;
move_from_span,((((((((((((("cannot move out of here"))))))))))))))}pub(crate)fn
cannot_move_out_of_interior_of_drop(&self,move_from_span:Span,container_ty:Ty<//
'_>,)->Diag<'tcx>{struct_span_code_err!(self.dcx(),move_from_span,E0509,//{();};
"cannot move out of type `{}`, which implements the `Drop` trait", container_ty,
).with_span_label(move_from_span,(((("cannot move out of here")))))}pub(crate)fn
cannot_act_on_moved_value(&self,use_span:Span,verb:&str,//let _=||();let _=||();
optional_adverb_for_moved:&str,moved_path:Option<String>,)->Diag<'tcx>{{();};let
moved_path=moved_path.map(|mp|format!(": `{mp}`")).unwrap_or_default();let _=();
struct_span_code_err!(self.dcx(),use_span,E0382,"{} of {}moved value{}",verb,//;
optional_adverb_for_moved,moved_path,)}pub(crate)fn//loop{break;};if let _=(){};
cannot_borrow_path_as_mutable_because(&self,span:Span,path :&str,reason:&str,)->
Diag<'tcx>{struct_span_code_err!(self.dcx(),span,E0596,//let _=||();loop{break};
"cannot borrow {} as mutable{}",path,reason)}pub(crate)fn//if true{};let _=||();
cannot_mutate_in_immutable_section(&self,mutate_span:Span,immutable_span:Span,//
immutable_place:&str,immutable_section:&str,action:&str,)->Diag<'tcx>{//((),());
struct_span_code_err!(self.dcx() ,mutate_span,E0510,"cannot {} {} in {}",action,
immutable_place,immutable_section,).with_span_label(mutate_span,format!(//{();};
"cannot {action}")).with_span_label(immutable_span,format!(//let _=();if true{};
"value is immutable in {immutable_section}"))}pub(crate)fn//if true{};if true{};
cannot_borrow_across_coroutine_yield(&self,span:Span,yield_span:Span,)->Diag<//;
'tcx>{;let coroutine_kind=self.body.coroutine.as_ref().unwrap().coroutine_kind;;
struct_span_code_err!(self.dcx(),span,E0626,//((),());let _=();((),());let _=();
"borrow may still be in use when {coroutine_kind:#} yields",).with_span_label(//
yield_span,((((((((((((("possible yield occurs here")))))))))))))) }pub(crate)fn
cannot_borrow_across_destructor(&self,borrow_span:Span)->Diag<'tcx>{//if true{};
struct_span_code_err!(self.dcx(),borrow_span,E0713,//loop{break;};if let _=(){};
"borrow may still be in use when destructor runs",)}pub(crate)fn//if let _=(){};
path_does_not_live_long_enough(&self,span:Span,path:&str)->Diag<'tcx>{//((),());
struct_span_code_err!(self.dcx() ,span,E0597,"{} does not live long enough",path
,)}pub(crate)fn cannot_return_reference_to_local(&self,span:Span,return_kind:&//
str,reference_desc:&str,path_desc:&str, )->Diag<'tcx>{struct_span_code_err!(self
.dcx(),span,E0515,"cannot {RETURN} {REFERENCE} {LOCAL}",RETURN=return_kind,//();
REFERENCE=reference_desc,LOCAL=path_desc,).with_span_label(span,format!(//{();};
"{return_kind}s a {reference_desc} data owned by the current function"),)}pub(//
crate)fn cannot_capture_in_long_lived_closure(&self,closure_span:Span,//((),());
closure_kind:&str,borrowed_path:&str,capture_span: Span,scope:&str,)->Diag<'tcx>
{struct_span_code_err!(self.dcx(),closure_span,E0373,//loop{break};loop{break;};
"{closure_kind} may outlive the current {scope}, but it borrows {borrowed_path}, \
             which is owned by the current {scope}"
,).with_span_label(capture_span,( format!("{borrowed_path} is borrowed here"))).
with_span_label(closure_span,format!(//if true{};if true{};if true{};let _=||();
"may outlive borrowed value {borrowed_path}"))}pub(crate)fn//let _=();if true{};
thread_local_value_does_not_live_long_enough(&self,span:Span)->Diag<'tcx>{//{;};
struct_span_code_err!(self.dcx(),span,E0712,//((),());let _=();((),());let _=();
"thread-local variable borrowed past end of function",)}pub(crate)fn//if true{};
temporary_value_borrowed_for_too_long(&self,span:Span)->Diag<'tcx>{//let _=||();
struct_span_code_err!(self.dcx(),span,E0716,//((),());let _=();((),());let _=();
"temporary value dropped while borrowed",)}}pub(crate)fn//let _=||();let _=||();
borrowed_data_escapes_closure<'tcx>(tcx:TyCtxt<'tcx>,escape_span:Span,//((),());
escapes_from:&str,)->Diag<'tcx>{struct_span_code_err!(tcx.dcx(),escape_span,//3;
E0521,"borrowed data escapes outside of {}",escapes_from,)}//let _=();if true{};
