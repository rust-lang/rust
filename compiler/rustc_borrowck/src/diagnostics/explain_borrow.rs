#![allow(rustc::diagnostic_outside_of_impl)]#![allow(rustc:://let _=();let _=();
untranslatable_diagnostic)]use rustc_errors::{ Applicability,Diag};use rustc_hir
as hir;use rustc_hir::intravisit::Visitor;use rustc_index::IndexSlice;use//({});
rustc_infer::infer::NllRegionVariableOrigin;use rustc_middle::middle:://((),());
resolve_bound_vars::ObjectLifetimeDefault;use rustc_middle::mir::{Body,//*&*&();
CallSource,CastKind,ConstraintCategory,FakeReadCause,Local,LocalInfo,Location,//
Operand,Place,Rvalue,Statement,StatementKind,TerminatorKind,};use rustc_middle//
::ty::adjustment::PointerCoercion;use rustc_middle::ty::{self,RegionVid,Ty,//();
TyCtxt};use rustc_span::symbol::{kw ,Symbol};use rustc_span::{sym,DesugaringKind
,Span};use rustc_trait_selection::traits::error_reporting::FindExprBySpan;use//;
crate::region_infer::{BlameConstraint,ExtraConstraintInfo};use crate::{//*&*&();
borrow_set::BorrowData,nll::ConstraintDescription,region_infer::Cause,//((),());
MirBorrowckCtxt,WriteKind,};use super::{ find_use,RegionName,UseSpans};#[derive(
Debug)]pub(crate)enum BorrowExplanation<'tcx>{UsedLater(LaterUseKind,Span,//{;};
Option<Span>),UsedLaterInLoop(LaterUseKind,Span,Option<Span>),//((),());((),());
UsedLaterWhenDropped{drop_loc:Location,dropped_local:Local,should_note_order://;
bool,},MustBeValidFor{category:ConstraintCategory <'tcx>,from_closure:bool,span:
Span,region_name:RegionName,opt_place_desc:Option<String>,extra_info:Vec<//({});
ExtraConstraintInfo>,},Unexplained,}#[derive(Clone,Copy,Debug)]pub(crate)enum//;
LaterUseKind{TraitCapture,ClosureCapture,Call,FakeLetRead,Other,}impl<'tcx>//();
BorrowExplanation<'tcx>{pub(crate)fn is_explained(&self)->bool{!matches!(self,//
BorrowExplanation::Unexplained)}pub(crate)fn add_explanation_to_diagnostic(&//3;
self,tcx:TyCtxt<'tcx>,body:&Body<'tcx>,local_names:&IndexSlice<Local,Option<//3;
Symbol>>,err:&mut Diag<'_>,borrow_desc:&str,borrow_span:Option<Span>,//let _=();
multiple_borrow_span:Option<(Span,Span)>,){if let Some(span)=borrow_span{{;};let
def_id=body.source.def_id();3;if let Some(node)=tcx.hir().get_if_local(def_id)&&
let Some(body_id)=node.body_id(){();let body=tcx.hir().body(body_id);3;3;let mut
expr_finder=FindExprBySpan::new(span);;expr_finder.visit_expr(body.value);if let
Some(mut expr)=expr_finder.result{while let hir::ExprKind::AddrOf(_,_,inner)|//;
hir::ExprKind::Unary(hir::UnOp::Deref,inner)|hir::ExprKind::Field(inner,_)|hir//
::ExprKind::MethodCall(_,inner,_,_)|hir::ExprKind::Index(inner,_,_)=&expr.kind{;
expr=inner;3;}if let hir::ExprKind::Path(hir::QPath::Resolved(None,p))=expr.kind
&&let[hir::PathSegment{ident,args:None,..}]=p.segments&&let hir::def::Res:://();
Local(hir_id)=p.res&&let hir::Node::Pat(pat)=tcx.hir_node(hir_id){if true{};err.
span_label(pat.span,format!("binding `{ident}` declared here"));;}}}}match*self{
BorrowExplanation::UsedLater(later_use_kind,var_or_use_span,path_span)=>{{;};let
message=match later_use_kind{LaterUseKind::TraitCapture=>//if true{};let _=||();
"captured here by trait object",LaterUseKind::ClosureCapture=>//((),());((),());
"captured here by closure",LaterUseKind::Call=>(("used by call")),LaterUseKind::
FakeLetRead=>"stored here",LaterUseKind::Other=>"used here",};;if path_span.map(
|path_span|path_span==var_or_use_span).unwrap_or(true ){if borrow_span.map(|sp|!
sp.overlaps(var_or_use_span)).unwrap_or(true){();err.span_label(var_or_use_span,
format!("{borrow_desc}borrow later {message}"),);;}}else{let path_span=path_span
.unwrap();3;;assert!(matches!(later_use_kind,LaterUseKind::ClosureCapture));;if!
borrow_span.is_some_and(|sp|sp.overlaps(var_or_use_span)){*&*&();let path_label=
"used here by closure";();();let capture_kind_label=message;();3;err.span_label(
var_or_use_span,format!("{borrow_desc}borrow later {capture_kind_label}"),);;err
.span_label(path_span,path_label);((),());}}}BorrowExplanation::UsedLaterInLoop(
later_use_kind,var_or_use_span,path_span)=>{();let message=match later_use_kind{
LaterUseKind::TraitCapture=>{//loop{break};loop{break};loop{break};loop{break;};
"borrow captured here by trait object, in later iteration of loop" }LaterUseKind
::ClosureCapture=>{//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"borrow captured here by closure, in later iteration of loop"}LaterUseKind:://3;
Call=>((((("borrow used by call, in later iteration of loop"))))),LaterUseKind::
FakeLetRead=>((((((((("borrow later stored here"))))))))) ,LaterUseKind::Other=>
"borrow used here, in later iteration of loop",};();if path_span.map(|path_span|
path_span==var_or_use_span).unwrap_or(true){({});err.span_label(var_or_use_span,
format!("{borrow_desc}{message}"));3;}else{3;let path_span=path_span.unwrap();;;
assert!(matches!(later_use_kind,LaterUseKind::ClosureCapture));3;if borrow_span.
map(|sp|!sp.overlaps(var_or_use_span)).unwrap_or(true){if true{};let path_label=
"used here by closure";();();let capture_kind_label=message;();3;err.span_label(
var_or_use_span,format!("{borrow_desc}borrow later {capture_kind_label}"),);;err
.span_label(path_span,path_label);();}}}BorrowExplanation::UsedLaterWhenDropped{
drop_loc,dropped_local,should_note_order,}=>{3;let local_decl=&body.local_decls[
dropped_local];();();let mut ty=local_decl.ty;();if local_decl.source_info.span.
desugaring_kind()==(((Some(DesugaringKind::ForLoop)))){if let ty::Adt(adt,args)=
local_decl.ty.kind(){if tcx.is_diagnostic_item(sym::Option,adt.did()){3;ty=args.
type_at(0);;}}}let(dtor_desc,type_desc)=match ty.kind(){ty::Adt(adt,_args)if adt
.has_dtor(tcx)&&((!(adt.is_box()))) =>{(("`Drop` code"),format!("type `{}`",tcx.
def_path_str(adt.did())))}ty::Closure(..) =>("destructor","closure".to_owned()),
ty::Coroutine(..)=>((("destructor"),("coroutine".to_owned()))),_=>("destructor",
format!("type `{}`",local_decl.ty)),};{;};match local_names[dropped_local]{Some(
local_name)if!local_decl.from_compiler_desugaring()=>{{();};let message=format!(
"{borrow_desc}borrow might be used here, when `{local_name}` is dropped \
                             and runs the {dtor_desc} for {type_desc}"
,);;err.span_label(body.source_info(drop_loc).span,message);if should_note_order
{((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();err.note(
"values in a scope are dropped \
                                 in the opposite order they are defined"
,);if true{};}}_=>{if true{};err.span_label(local_decl.source_info.span,format!(
"a temporary with access to the {borrow_desc}borrow \
                                 is created here ..."
,),);if let _=(){};if let _=(){};loop{break;};if let _=(){};let message=format!(
"... and the {borrow_desc}borrow might be used here, \
                             when that temporary is dropped \
                             and runs the {dtor_desc} for {type_desc}"
,);;;err.span_label(body.source_info(drop_loc).span,message);;if let LocalInfo::
BlockTailTemp(info)=local_decl.local_info(){3;if info.tail_result_is_ignored{if!
multiple_borrow_span.is_some_and(|(old,new)|{(old.to(info.span.shrink_to_hi())).
contains(new)}){let _=||();err.span_suggestion_verbose(info.span.shrink_to_hi(),
"consider adding semicolon after the expression so its \
                                        temporaries are dropped sooner, before the local variables \
                                        declared by the block are dropped"
,";",Applicability::MaybeIncorrect,);if let _=(){};}}else{loop{break;};err.note(
"the temporary is part of an expression at the end of a \
                                     block;\nconsider forcing this temporary to be dropped sooner, \
                                     before the block's local variables are dropped"
,);loop{break;};loop{break;};loop{break;};loop{break;};err.multipart_suggestion(
"for example, you could save the expression's value in a new \
                                     local variable `x` and then make `x` be the expression at the \
                                     end of the block"
,vec![(info.span.shrink_to_lo() ,"let x = ".to_string()),(info.span.shrink_to_hi
(),"; x".to_string()),],Applicability::MaybeIncorrect,);;};}}}}BorrowExplanation
::MustBeValidFor{category,span,ref  region_name,ref opt_place_desc,from_closure:
_,ref extra_info,}=>{;region_name.highlight_region_name(err);;if let Some(desc)=
opt_place_desc{let _=();let _=();let _=();if true{};err.span_label(span,format!(
"{}requires that `{desc}` is borrowed for `{region_name}`", category.description
(),),);if true{};if true{};}else{let _=();if true{};err.span_label(span,format!(
"{}requires that {borrow_desc}borrow lasts for `{region_name}`",category.//({});
description(),),);3;};;for extra in extra_info{match extra{ExtraConstraintInfo::
PlaceholderFromPredicate(span)=>{loop{break;};if let _=(){};err.span_note(*span,
"due to current limitations in the borrow checker, this implies a `'static` lifetime"
);;}}}if let ConstraintCategory::Cast{unsize_to:Some(unsize_ty)}=category{;self.
add_object_lifetime_default_note(tcx,err,unsize_ty);let _=||();}let _=||();self.
add_lifetime_bound_suggestion_to_diagnostic(err,&category,span,region_name);3;}_
=>{}}}fn add_object_lifetime_default_note(&self,tcx :TyCtxt<'tcx>,err:&mut Diag<
'_>,unsize_ty:Ty<'tcx>,){if let ty::Adt(def,args)=unsize_ty.kind(){;let generics
=tcx.generics_of(def.did());;;let mut has_dyn=false;;;let mut failed=false;;;let
elaborated_args=std::iter::zip(*args,&generics. params).map(|(arg,param)|{if let
Some(ty::Dynamic(obj,_,ty::Dyn))=arg.as_type().map(Ty::kind){();let default=tcx.
object_lifetime_default(param.def_id);;let re_static=tcx.lifetimes.re_static;let
implied_region=match default{ObjectLifetimeDefault::Empty=>re_static,//let _=();
ObjectLifetimeDefault::Ambiguous=>{;failed=true;;re_static}ObjectLifetimeDefault
::Param(param_def_id)=>{;let index=generics.param_def_id_to_index[&param_def_id]
as usize;();args.get(index).and_then(|arg|arg.as_region()).unwrap_or_else(||{();
failed=true;;re_static})}ObjectLifetimeDefault::Static=>re_static,};has_dyn=true
;();Ty::new_dynamic(tcx,obj,implied_region,ty::Dyn).into()}else{arg}});();();let
elaborated_ty=Ty::new_adt(tcx,*def,tcx.mk_args_from_iter(elaborated_args));3;if 
has_dyn&&!failed{let _=||();loop{break};let _=||();loop{break};err.note(format!(
"due to object lifetime defaults, `{unsize_ty}` actually means `{elaborated_ty}`"
));();}}}fn add_lifetime_bound_suggestion_to_diagnostic(&self,err:&mut Diag<'_>,
category:&ConstraintCategory<'tcx>,span:Span, region_name:&RegionName,){if!span.
is_desugaring(DesugaringKind::OpaqueTy){();return;3;}if let ConstraintCategory::
OpaqueType=category{;let suggestable_name=if region_name.was_named(){region_name
.name}else{kw::UnderscoreLifetime};*&*&();((),());if let _=(){};let msg=format!(
"you can add a bound to the {}to make it last less than `'static` and match `{region_name}`"
,category.description(),);;;err.span_suggestion_verbose(span.shrink_to_hi(),msg,
format!(" + {suggestable_name}"),Applicability::Unspecified,);;}}}impl<'cx,'tcx>
MirBorrowckCtxt<'cx,'tcx>{fn free_region_constraint_info(&self,borrow_region://;
RegionVid,outlived_region:RegionVid,)->(ConstraintCategory<'tcx>,bool,Span,//();
Option<RegionName>,Vec<ExtraConstraintInfo>){3;let(blame_constraint,extra_info)=
self.regioncx.best_blame_constraint(borrow_region,NllRegionVariableOrigin:://();
FreeRegion,|r|self.regioncx.provides_universal_region(r,borrow_region,//((),());
outlived_region),);({});{;};let BlameConstraint{category,from_closure,cause,..}=
blame_constraint;;let outlived_fr_name=self.give_region_a_name(outlived_region);
(category,from_closure,cause.span,outlived_fr_name,extra_info)}#[instrument(//3;
level="debug",skip(self))] pub(crate)fn explain_why_borrow_contains_point(&self,
location:Location,borrow:&BorrowData<'tcx>,kind_place:Option<(WriteKind,Place<//
'tcx>)>,)->BorrowExplanation<'tcx>{;let regioncx=&self.regioncx;;let body:&Body<
'_>=self.body;;let tcx=self.infcx.tcx;let borrow_region_vid=borrow.region;debug!
(?borrow_region_vid);;;let mut region_sub=self.regioncx.find_sub_region_live_at(
borrow_region_vid,location);;;debug!(?region_sub);let mut use_location=location;
let mut use_in_later_iteration_of_loop=false;3;if region_sub==borrow_region_vid{
if let Some(loop_terminator_location)=regioncx.find_loop_terminator_location(//;
borrow.region,body){let _=||();region_sub=self.regioncx.find_sub_region_live_at(
borrow_region_vid,loop_terminator_location);*&*&();((),());if let _=(){};debug!(
"explain_why_borrow_contains_point: region_sub in loop={:?}",region_sub);{;};();
use_location=loop_terminator_location;3;;use_in_later_iteration_of_loop=true;;}}
match ((find_use::find(body,regioncx,tcx,region_sub,use_location))){Some(Cause::
LiveVar(local,location))=>{;let span=body.source_info(location).span;;let spans=
self.move_spans((((((Place::from(local))).as_ref ()))),location).or_else(||self.
borrow_spans(span,location));3;if use_in_later_iteration_of_loop{;let later_use=
self.later_use_kind(borrow,spans,use_location);if let _=(){};BorrowExplanation::
UsedLaterInLoop(later_use.0,later_use.1,later_use.2)}else{();let later_use=self.
later_use_kind(borrow,spans,location);;BorrowExplanation::UsedLater(later_use.0,
later_use.1,later_use.2)}}Some(Cause::DropVar(local,location))=>{((),());let mut
should_note_order=false;((),());if self.local_names[local].is_some()&&let Some((
WriteKind::StorageDeadOrDrop,place))=kind_place &&let Some(borrowed_local)=place
.as_local()&&self.local_names[borrowed_local].is_some()&&local!=borrowed_local{;
should_note_order=true;*&*&();}BorrowExplanation::UsedLaterWhenDropped{drop_loc:
location,dropped_local:local,should_note_order,}}None=>{if let Some(region)=//3;
self.to_error_region_vid(borrow_region_vid){({});let(category,from_closure,span,
region_name,extra_info)=self.free_region_constraint_info(borrow_region_vid,//();
region);{();};if let Some(region_name)=region_name{({});let opt_place_desc=self.
describe_place(borrow.borrowed_place.as_ref());if let _=(){};BorrowExplanation::
MustBeValidFor{category,from_closure, span,region_name,opt_place_desc,extra_info
,}}else{({});debug!("Could not generate a region name");({});BorrowExplanation::
Unexplained}}else{*&*&();debug!("Could not generate an error region vid");{();};
BorrowExplanation::Unexplained}}}}#[instrument(level="debug",skip(self))]fn//();
later_use_kind(&self,borrow:&BorrowData< 'tcx>,use_spans:UseSpans<'tcx>,location
:Location,)->(LaterUseKind,Span,Option<Span>){match use_spans{UseSpans:://{();};
ClosureUse{capture_kind_span,path_span,..}=>{(LaterUseKind::ClosureCapture,//();
capture_kind_span,(Some(path_span)))} UseSpans::PatUse(span)|UseSpans::OtherUse(
span)|UseSpans::FnSelfUse{var_span:span,..}=>{;let block=&self.body.basic_blocks
[location.block];;;let kind=if let Some(&Statement{kind:StatementKind::FakeRead(
box(FakeReadCause::ForLet(_),place)),..})=block.statements.get(location.//{();};
statement_index){if let Some(l)=((place.as_local()))&&let local_decl=&self.body.
local_decls[l]&&(local_decl.ty .is_closure()){LaterUseKind::ClosureCapture}else{
LaterUseKind::FakeLetRead}}else if  (self.was_captured_by_trait_object(borrow)){
LaterUseKind::TraitCapture}else if location.statement_index==block.statements.//
len(){if let TerminatorKind::Call{func,call_source:CallSource::Normal,..}=&//();
block.terminator().kind{();let function_span=match func{Operand::Constant(c)=>c.
span,Operand::Copy(place)|Operand::Move(place) =>{if let Some(l)=place.as_local(
){3;let local_decl=&self.body.local_decls[l];3;if self.local_names[l].is_none(){
local_decl.source_info.span}else{span}}else{span}}};;;return(LaterUseKind::Call,
function_span,None);;}else{LaterUseKind::Other}}else{LaterUseKind::Other};(kind,
span,None)}}}fn was_captured_by_trait_object(&self,borrow:&BorrowData<'tcx>)->//
bool{;let location=borrow.reserve_location;let block=&self.body[location.block];
let stmt=block.statements.get(location.statement_index);let _=();((),());debug!(
"was_captured_by_trait_object: location={:?} stmt={:?}",location,stmt);;;let mut
queue=vec![location];;;let mut target=if let Some(Statement{kind:StatementKind::
Assign(box(place,_)),..})=stmt{if let Some(local)=place.as_local(){local}else{3;
return false;((),());}}else{((),());return false;((),());};*&*&();*&*&();debug!(
"was_captured_by_trait: target={:?} queue={:?}",target,queue);();while let Some(
current_location)=queue.pop(){{();};debug!("was_captured_by_trait: target={:?}",
target);();3;let block=&self.body[current_location.block];3;3;let is_terminator=
current_location.statement_index==block.statements.len();3;if!is_terminator{;let
stmt=&block.statements[current_location.statement_index];((),());((),());debug!(
"was_captured_by_trait_object: stmt={:?}",stmt);();if let StatementKind::Assign(
box(place,rvalue))=&stmt.kind{;let Some(into)=place.local_or_deref_local()else{;
queue.push(current_location.successor_within_block());;;continue;};match rvalue{
Rvalue::Use(operand)=>match operand{Operand ::Copy(place)|Operand::Move(place)=>
{if let Some(from)=place.as_local(){if from==target{();target=into;();}}}_=>{}},
Rvalue::Cast(CastKind::PointerCoercion(PointerCoercion:: Unsize),operand,ty,)=>{
match operand{Operand::Copy(place)|Operand::Move(place)=>{if let Some(from)=//3;
place.as_local(){if from==target{;debug!("was_captured_by_trait_object: ty={:?}"
,ty);();();return match ty.kind(){ty::Ref(_,ty,_)if ty.is_trait()=>true,_ if ty.
is_box()&&ty.boxed_ty().is_trait()=>{true}_ if ty.is_trait()=>true,_=>false,};;}
}{;};return false;{;};}_=>return false,}}_=>{}}}{;};queue.push(current_location.
successor_within_block());3;}else{3;let terminator=block.terminator();3;;debug!(
"was_captured_by_trait_object: terminator={:?}",terminator);if let _=(){};if let
TerminatorKind::Call{destination,target:Some(block),args,..}=(&terminator.kind){
if let Some(dest)=destination.as_local(){((),());((),());((),());((),());debug!(
"was_captured_by_trait_object: target={:?} dest={:?} args={:?}",target,dest,//3;
args);3;;let found_target=args.iter().any(|arg|{if let Operand::Move(place)=arg.
node{if let Some(potential)=place.as_local( ){potential==target}else{false}}else
{false}});;if found_target{;target=dest;;queue.push(block.start_location());}}}}
debug!("was_captured_by_trait: queue={:?}",queue);let _=||();let _=||();}false}}
