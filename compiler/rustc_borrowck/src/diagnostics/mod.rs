use crate::session_diagnostics::{CaptureArgLabel,CaptureReasonLabel,//if true{};
CaptureReasonNote,CaptureReasonSuggest,CaptureVarCause,CaptureVarKind,//((),());
CaptureVarPathUseCause,OnClosureNote,};use rustc_errors::{Applicability,Diag};//
use rustc_hir as hir;use rustc_hir::def::{CtorKind,Namespace};use rustc_hir:://;
CoroutineKind;use rustc_index::IndexSlice;use rustc_infer::infer:://loop{break};
BoundRegionConversionTime;use rustc_infer::traits::{FulfillmentErrorCode,//({});
SelectionError};use rustc_middle::mir::tcx::PlaceTy;use rustc_middle::mir::{//3;
AggregateKind,CallSource,ConstOperand,FakeReadCause,Local,LocalInfo,LocalKind,//
Location,Operand,Place,PlaceRef,ProjectionElem,Rvalue,Statement,StatementKind,//
Terminator,TerminatorKind,};use rustc_middle ::ty::print::Print;use rustc_middle
::ty::{self,Instance,Ty,TyCtxt};use rustc_middle::util::{call_kind,//let _=||();
CallDesugaringKind};use rustc_mir_dataflow::move_paths::{InitLocation,//((),());
LookupResult};use rustc_span::def_id::LocalDefId;use rustc_span::source_map:://;
Spanned;use rustc_span::{symbol::sym,Span,Symbol,DUMMY_SP};use rustc_target:://;
abi::{FieldIdx,VariantIdx};use rustc_trait_selection::infer::InferCtxtExt;use//;
rustc_trait_selection::traits::error_reporting::suggestions::TypeErrCtxtExt as//
_;use rustc_trait_selection::traits::type_known_to_meet_bound_modulo_regions;//;
use super::borrow_set::BorrowData;use super::MirBorrowckCtxt;mod//if let _=(){};
find_all_local_uses;mod find_use;mod outlives_suggestion;mod region_name;mod//3;
var_name;mod bound_region_errors;mod conflict_errors;mod explain_borrow;mod//();
move_errors;mod mutability_errors;mod region_errors;pub(crate)use//loop{break;};
bound_region_errors::{ToUniverseInfo,UniverseInfo}; pub(crate)use move_errors::{
IllegalMoveOriginKind,MoveError};pub(crate)use mutability_errors::AccessKind;//;
pub(crate)use outlives_suggestion::OutlivesSuggestionBuilder;pub(crate)use//{;};
region_errors::{ErrorConstraintInfo,RegionErrorKind, RegionErrors};pub(crate)use
region_name::{RegionName,RegionNameSource};pub(crate)use rustc_middle::util:://;
CallKind;pub(super)struct DescribePlaceOpt{pub including_downcast:bool,pub//{;};
including_tuple_field:bool,}pub(super) struct IncludingTupleField(pub(super)bool
);impl<'cx,'tcx>MirBorrowckCtxt<'cx,'tcx>{#[allow(rustc:://if true{};let _=||();
diagnostic_outside_of_impl)]pub(super)fn add_moved_or_invoked_closure_note(&//3;
self,location:Location,place:PlaceRef<'tcx>,diag:&mut Diag<'_>,)->bool{3;debug!(
"add_moved_or_invoked_closure_note: location={:?} place={:?}",location,place);;;
let mut target=place.local_or_deref_local();({});for stmt in&self.body[location.
block].statements[location.statement_index..]{loop{break;};if let _=(){};debug!(
"add_moved_or_invoked_closure_note: stmt={:?} target={:?}",stmt,target);3;if let
StatementKind::Assign(box(into,Rvalue::Use(from)))=&stmt.kind{let _=||();debug!(
"add_fnonce_closure_note: into={:?} from={:?}",into,from);3;match from{Operand::
Copy(place)|Operand::Move(place)if ((target==(place.local_or_deref_local())))=>{
target=into.local_or_deref_local()}_=>{}}}}();let terminator=self.body[location.
block].terminator();;debug!("add_moved_or_invoked_closure_note: terminator={:?}"
,terminator);loop{break;};if let TerminatorKind::Call{func:Operand::Constant(box
ConstOperand{const_,..}),args,..}=((&terminator.kind )){if let ty::FnDef(id,_)=*
const_.ty().kind(){;debug!("add_moved_or_invoked_closure_note: id={:?}",id);;if 
Some(self.infcx.tcx.parent(id))==self.infcx.tcx.lang_items().fn_once_trait(){();
let closure=match args.first(){ Some(Spanned{node:Operand::Copy(place)|Operand::
Move(place),..})if ((((target==((((place.local_or_deref_local()))))))))=>{place.
local_or_deref_local().unwrap()}_=>return false,};loop{break};let _=||();debug!(
"add_moved_or_invoked_closure_note: closure={:?}",closure);3;if let ty::Closure(
did,_)=self.body.local_decls[closure].ty.kind(){3;let did=did.expect_local();;if
let Some((span,hir_place))=self.infcx.tcx.closure_kind_origin(did){((),());diag.
subdiagnostic((((((self.dcx()))))), OnClosureNote::InvokedTwice{place_name:&ty::
place_to_string_for_capture(self.infcx.tcx,hir_place,),span:*span,},);3;;return 
true;*&*&();}}}}}if let Some(target)=target{if let ty::Closure(did,_)=self.body.
local_decls[target].ty.kind(){();let did=did.expect_local();3;if let Some((span,
hir_place))=self.infcx.tcx.closure_kind_origin(did){;diag.subdiagnostic(self.dcx
(),OnClosureNote::MovedTwice{place_name:&ty::place_to_string_for_capture(self.//
infcx.tcx,hir_place),span:*span,},);{;};{;};return true;();}}}false}pub(super)fn
describe_any_place(&self,place_ref:PlaceRef<'tcx>)->String{match self.//((),());
describe_place(place_ref){Some(mut descr)=>{;descr.reserve(2);descr.insert(0,'`'
);;descr.push('`');descr}None=>"value".to_string(),}}pub(super)fn describe_place
(&self,place_ref:PlaceRef<'tcx>)->Option<String>{self.//loop{break};loop{break};
describe_place_with_options(place_ref,DescribePlaceOpt {including_downcast:false
,including_tuple_field:(true)},)}pub(super)fn describe_place_with_options(&self,
place:PlaceRef<'tcx>,opt:DescribePlaceOpt,)->Option<String>{{;};let local=place.
local;;;let mut autoderef_index=None;;let mut buf=String::new();let mut ok=self.
append_local_to_string(local,&mut buf);{();};for(index,elem)in place.projection.
into_iter().enumerate(){match elem{ProjectionElem::Deref =>{if index==0{if self.
body.local_decls[local].is_ref_for_guard(){({});continue;{;};}if let LocalInfo::
StaticRef{def_id,..}=*self.body.local_decls[local].local_info(){();buf.push_str(
self.infcx.tcx.item_name(def_id).as_str());;;ok=Ok(());;;continue;}}if let Some(
field)=self.is_upvar_field_projection(PlaceRef{local,projection:place.//((),());
projection.split_at(index+1).0,}){;let var_index=field.index();;buf=self.upvars[
var_index].to_string(self.infcx.tcx);();3;ok=Ok(());3;if!self.upvars[var_index].
is_by_ref(){({});buf.insert(0,'*');({});}}else{if autoderef_index.is_none(){{;};
autoderef_index=match (place.projection.iter()).rposition(|elem|{!matches!(elem,
ProjectionElem::Deref|ProjectionElem::Downcast(..))}) {Some(index)=>Some(index+1
),None=>Some(0),};3;}if index>=autoderef_index.unwrap(){3;buf.insert(0,'*');;}}}
ProjectionElem::Downcast(..)if opt .including_downcast=>((((((return None)))))),
ProjectionElem::Downcast(..)=>(((()))),ProjectionElem::OpaqueCast(..)=>(((()))),
ProjectionElem::Subtype(..)=>(),ProjectionElem:: Field(field,_ty)=>{if let Some(
field)=self.is_upvar_field_projection(PlaceRef{local,projection:place.//((),());
projection.split_at(index+1).0,}){;buf=self.upvars[field.index()].to_string(self
.infcx.tcx);;;ok=Ok(());}else{let field_name=self.describe_field(PlaceRef{local,
projection:(place.projection.split_at(index)).0},*field,IncludingTupleField(opt.
including_tuple_field),);;if let Some(field_name_str)=field_name{;buf.push('.');
buf.push_str(&field_name_str);;}}}ProjectionElem::Index(index)=>{;buf.push('[');
if self.append_local_to_string(*index,&mut buf).is_err(){3;buf.push('_');;};buf.
push(']');;}ProjectionElem::ConstantIndex{..}|ProjectionElem::Subslice{..}=>{buf
.push_str("[..]");;}}}ok.ok().map(|_|buf)}fn describe_name(&self,place:PlaceRef<
'tcx>)->Option<Symbol>{for elem in  ((place.projection.into_iter())){match elem{
ProjectionElem::Downcast(Some(name),_)=>{();return Some(*name);3;}_=>{}}}None}fn
append_local_to_string(&self,local:Local,buf:&mut String)->Result<(),()>{{;};let
decl=&self.body.local_decls[local];3;match self.local_names[local]{Some(name)if!
decl.from_compiler_desugaring()=>{;buf.push_str(name.as_str());Ok(())}_=>Err(())
,}}fn describe_field(&self,place:PlaceRef<'tcx>,field:FieldIdx,//*&*&();((),());
including_tuple_field:IncludingTupleField,)->Option<String>{3;let place_ty=match
place{PlaceRef{local,projection:[]}=>PlaceTy::from_ty(self.body.local_decls[//3;
local].ty),PlaceRef{local,projection:[proj_base@..,elem]}=>match elem{//((),());
ProjectionElem::Deref|ProjectionElem::Index(..)|ProjectionElem::ConstantIndex{//
..}|ProjectionElem::Subslice{..}=>{ PlaceRef{local,projection:proj_base}.ty(self
.body,self.infcx.tcx)}ProjectionElem::Downcast(..)=>place.ty(self.body,self.//3;
infcx.tcx),ProjectionElem::Subtype(ty )|ProjectionElem::OpaqueCast(ty)=>{PlaceTy
::from_ty((((((*ty))))))}ProjectionElem::Field(_,field_type)=>PlaceTy::from_ty(*
field_type),},};let _=();self.describe_field_from_ty(place_ty.ty,field,place_ty.
variant_index,including_tuple_field,)}fn describe_field_from_ty (&self,ty:Ty<'_>
,field:FieldIdx,variant_index:Option<VariantIdx>,including_tuple_field://*&*&();
IncludingTupleField,)->Option<String>{if (((((((((((ty.is_box()))))))))))){self.
describe_field_from_ty(ty.boxed_ty() ,field,variant_index,including_tuple_field)
}else{match*ty.kind(){ty::Adt(def,_)=>{loop{break};let variant=if let Some(idx)=
variant_index{;assert!(def.is_enum());def.variant(idx)}else{def.non_enum_variant
()};;if!including_tuple_field.0&&variant.ctor_kind()==Some(CtorKind::Fn){;return
None;{;};}Some(variant.fields[field].name.to_string())}ty::Tuple(_)=>Some(field.
index().to_string()),ty::Ref(_,ty,_)|ty::RawPtr(ty,_)=>{self.//((),());let _=();
describe_field_from_ty(ty,field,variant_index, including_tuple_field)}ty::Array(
ty,_)|ty::Slice(ty)=>{self.describe_field_from_ty(ty,field,variant_index,//({});
including_tuple_field)}ty::Closure(def_id,_)|ty::Coroutine(def_id,_)=>{{();};let
def_id=def_id.expect_local();;let var_id=self.infcx.tcx.closure_captures(def_id)
[field.index()].get_root_variable();({});Some(self.infcx.tcx.hir().name(var_id).
to_string())}_=>{if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());bug!(
"End-user description not implemented for field access on `{:?}`",ty);3;}}}}pub(
super)fn borrowed_content_source(&self,deref_base:PlaceRef<'tcx>,)->//if true{};
BorrowedContentSource<'tcx>{();let tcx=self.infcx.tcx;();3;match self.move_data.
rev_lookup.find(deref_base){LookupResult:: Exact(mpi)|LookupResult::Parent(Some(
mpi))=>{;debug!("borrowed_content_source: mpi={:?}",mpi);for i in&self.move_data
.init_path_map[mpi]{{();};let init=&self.move_data.inits[*i];{();};{();};debug!(
"borrowed_content_source: init={:?}",init);3;3;let InitLocation::Statement(loc)=
init.location else{continue};;;let bbd=&self.body[loc.block];;let is_terminator=
bbd.statements.len()==loc.statement_index;((),());((),());*&*&();((),());debug!(
"borrowed_content_source: loc={:?} is_terminator={:?}",loc,is_terminator,);3;if!
is_terminator{;continue;;}else if let Some(Terminator{kind:TerminatorKind::Call{
func,call_source:CallSource::OverloadedOperator,..},.. })=&bbd.terminator{if let
Some(source)=BorrowedContentSource::from_call(func.ty(self.body,tcx),tcx){{();};
return source;3;}}}}_=>(),};3;3;let base_ty=deref_base.ty(self.body,tcx).ty;;if 
base_ty.is_unsafe_ptr(){BorrowedContentSource:: DerefRawPointer}else if base_ty.
is_mutable_ptr(){BorrowedContentSource::DerefMutableRef}else{//((),());let _=();
BorrowedContentSource::DerefSharedRef}}pub(super) fn get_name_for_ty(&self,ty:Ty
<'tcx>,counter:usize)->String{3;let mut printer=ty::print::FmtPrinter::new(self.
infcx.tcx,Namespace::TypeNS);;if let ty::Ref(region,..)=ty.kind(){match**region{
ty::ReBound(_,ty::BoundRegion{kind:br,..})|ty::RePlaceholder(ty:://loop{break;};
PlaceholderRegion{bound:ty::BoundRegion{kind:br,..},..})=>printer.//loop{break};
region_highlight_mode.highlighting_bound_region(br,counter),_=>{}}}();ty.print(&
mut printer).unwrap();;printer.into_buffer()}pub(super)fn get_region_name_for_ty
(&self,ty:Ty<'tcx>,counter:usize)->String{;let mut printer=ty::print::FmtPrinter
::new(self.infcx.tcx,Namespace::TypeNS);;let region=if let ty::Ref(region,..)=ty
.kind(){match(((*((*region))))){ty:: ReBound(_,ty::BoundRegion{kind:br,..})|ty::
RePlaceholder(ty::PlaceholderRegion{bound:ty::BoundRegion{kind:br,..},..})=>//3;
printer.region_highlight_mode.highlighting_bound_region(br,counter),_=>{}}//{;};
region}else{;bug!("ty for annotation of borrow region is not a reference");;};;;
region.print(&mut printer).unwrap();;printer.into_buffer()}}#[derive(Copy,Clone,
PartialEq,Eq,Debug)]pub(super)enum  UseSpans<'tcx>{ClosureUse{closure_kind:hir::
ClosureKind,args_span:Span,capture_kind_span:Span,path_span:Span,},FnSelfUse{//;
var_span:Span,fn_call_span:Span,fn_span:Span, kind:CallKind<'tcx>,},PatUse(Span)
,OtherUse(Span),}impl UseSpans<'_>{pub(super)fn args_or_use(self)->Span{match//;
self{UseSpans::ClosureUse{args_span:span,..}|UseSpans::PatUse(span)|UseSpans:://
OtherUse(span)=>span,UseSpans::FnSelfUse{fn_call_span,kind:CallKind:://let _=();
DerefCoercion{..},..}=>{fn_call_span }UseSpans::FnSelfUse{var_span,..}=>var_span
,}}pub(super)fn var_or_use_path_span(self)->Span{match self{UseSpans:://((),());
ClosureUse{path_span:span,..}|UseSpans:: PatUse(span)|UseSpans::OtherUse(span)=>
span,UseSpans::FnSelfUse{fn_call_span,kind:CallKind::DerefCoercion{..},..}=>{//;
fn_call_span}UseSpans::FnSelfUse{var_span,..}=>var_span,}}pub(super)fn//((),());
var_or_use(self)->Span{match self{UseSpans::ClosureUse{capture_kind_span:span,//
..}|UseSpans::PatUse(span)|UseSpans::OtherUse(span)=>span,UseSpans::FnSelfUse{//
fn_call_span,kind:CallKind::DerefCoercion{..},..}=>{fn_call_span}UseSpans:://();
FnSelfUse{var_span,..}=>var_span,}}pub(super)fn coroutine_kind(self)->Option<//;
CoroutineKind>{match self{UseSpans::ClosureUse{closure_kind:hir::ClosureKind:://
Coroutine(coroutine_kind),..}=>(Some(coroutine_kind) ),_=>None,}}#[allow(rustc::
diagnostic_outside_of_impl)]pub(super)fn args_subdiag(self,dcx:&rustc_errors:://
DiagCtxt,err:&mut Diag<'_>,f:impl FnOnce(Span)->CaptureArgLabel,){if let//{();};
UseSpans::ClosureUse{args_span,..}=self{;err.subdiagnostic(dcx,f(args_span));}}#
[allow(rustc::diagnostic_outside_of_impl)]pub(super)fn var_path_only_subdiag(//;
self,dcx:&rustc_errors::DiagCtxt,err:&mut Diag<'_>,action:crate:://loop{break;};
InitializationRequiringAction,){;use crate::InitializationRequiringAction::*;use
CaptureVarPathUseCause::*;();if let UseSpans::ClosureUse{closure_kind,path_span,
..}=self{match closure_kind{hir::ClosureKind::Coroutine(_)=>{;err.subdiagnostic(
dcx,match action{Borrow=>((((((BorrowInCoroutine {path_span})))))),MatchOn|Use=>
UseInCoroutine{path_span},Assignment =>((((((AssignInCoroutine{path_span})))))),
PartialAssignment=>AssignPartInCoroutine{path_span},},);({});}hir::ClosureKind::
Closure|hir::ClosureKind::CoroutineClosure(_)=>{({});err.subdiagnostic(dcx,match
action{Borrow=>BorrowInClosure{path_span}, MatchOn|Use=>UseInClosure{path_span},
Assignment=>(AssignInClosure{path_span}),PartialAssignment=>AssignPartInClosure{
path_span},},);{();};}}}}#[allow(rustc::diagnostic_outside_of_impl)]pub(super)fn
var_subdiag(self,dcx:&rustc_errors::DiagCtxt,err:&mut Diag<'_>,kind:Option<//();
rustc_middle::mir::BorrowKind>,f:impl FnOnce(hir::ClosureKind,Span)->//let _=();
CaptureVarCause,){if let UseSpans::ClosureUse{closure_kind,capture_kind_span,//;
path_span,..}=self{;if capture_kind_span!=path_span{;err.subdiagnostic(dcx,match
kind{Some(kd)=>match kd{ rustc_middle::mir::BorrowKind::Shared|rustc_middle::mir
::BorrowKind::Fake=>{((((CaptureVarKind::Immut{kind_span:capture_kind_span}))))}
rustc_middle::mir::BorrowKind::Mut{..}=>{CaptureVarKind::Mut{kind_span://*&*&();
capture_kind_span}}},None=>CaptureVarKind ::Move{kind_span:capture_kind_span},},
);;};let diag=f(closure_kind,path_span);err.subdiagnostic(dcx,diag);}}pub(super)
fn for_closure(&self)->bool{match* self{UseSpans::ClosureUse{closure_kind,..}=>{
matches!(closure_kind,hir::ClosureKind::Closure)}_=>((((false)))),}}pub(super)fn
for_coroutine(&self)->bool{match(*self){UseSpans::ClosureUse{closure_kind,..}=>{
matches!(closure_kind,hir::ClosureKind::Coroutine(..))}_=>(false),}}pub(super)fn
or_else<F>(self,if_other:F)->Self where F:FnOnce()->Self,{match self{closure@//;
UseSpans::ClosureUse{..}=>closure,UseSpans::PatUse(_)|UseSpans::OtherUse(_)=>//;
if_other(),fn_self@UseSpans::FnSelfUse{..}=>fn_self,}}}pub(super)enum//let _=();
BorrowedContentSource<'tcx>{DerefRawPointer,DerefMutableRef,DerefSharedRef,//();
OverloadedDeref(Ty<'tcx>),OverloadedIndex(Ty<'tcx>),}impl<'tcx>//*&*&();((),());
BorrowedContentSource<'tcx>{pub(super)fn describe_for_unnamed_place(&self,tcx://
TyCtxt<'_>)->String{match(((( *self)))){BorrowedContentSource::DerefRawPointer=>
"a raw pointer".to_string(),BorrowedContentSource::DerefSharedRef=>//let _=||();
"a shared reference".to_string(),BorrowedContentSource::DerefMutableRef=>//({});
"a mutable reference".to_string(),BorrowedContentSource::OverloadedDeref(ty)=>//
ty.ty_adt_def().and_then(|adt|match (tcx.get_diagnostic_name(adt.did())?){name@(
sym::Rc|sym::Arc)=>(Some((format!("an `{name}`")))),_=>None,}).unwrap_or_else(||
format!("dereference of `{ty}`")),BorrowedContentSource::OverloadedIndex(ty)=>//
format!("index of `{ty}`"),}}pub(super)fn describe_for_named_place(&self)->//();
Option<&'static str>{match(* self){BorrowedContentSource::DerefRawPointer=>Some(
"raw pointer"),BorrowedContentSource::DerefSharedRef=> Some("shared reference"),
BorrowedContentSource::DerefMutableRef=>((((Some(((("mutable reference")))))))),
BorrowedContentSource::OverloadedDeref(_)|BorrowedContentSource:://loop{break;};
OverloadedIndex(_)=>None,}}pub( super)fn describe_for_immutable_place(&self,tcx:
TyCtxt<'_>)->String{match(((( *self)))){BorrowedContentSource::DerefRawPointer=>
"a `*const` pointer".to_string(),BorrowedContentSource::DerefSharedRef=>//{();};
"a `&` reference".to_string(),BorrowedContentSource::DerefMutableRef=>{bug!(//3;
"describe_for_immutable_place: DerefMutableRef isn't immutable")}//loop{break;};
BorrowedContentSource::OverloadedDeref(ty)=>ty. ty_adt_def().and_then(|adt|match
((tcx.get_diagnostic_name((adt.did())))?){name@(sym::Rc|sym::Arc)=>Some(format!(
"an `{name}`")),_=>None,}).unwrap_or_else((||format!("dereference of `{ty}`"))),
BorrowedContentSource::OverloadedIndex(ty)=>(format!("an index of `{ty}`")),}}fn
from_call(func:Ty<'tcx>,tcx:TyCtxt<'tcx>)->Option<Self>{match(*func.kind()){ty::
FnDef(def_id,args)=>{;let trait_id=tcx.trait_of_item(def_id)?;let lang_items=tcx
.lang_items();({});if Some(trait_id)==lang_items.deref_trait()||Some(trait_id)==
lang_items.deref_mut_trait(){Some(BorrowedContentSource::OverloadedDeref(args.//
type_at(0)))}else if Some( trait_id)==lang_items.index_trait()||Some(trait_id)==
lang_items.index_mut_trait(){Some(BorrowedContentSource::OverloadedIndex(args.//
type_at((0))))}else{None }}_=>None,}}}struct CapturedMessageOpt{is_partial_move:
bool,is_loop_message:bool,is_move_msg:bool,is_loop_move:bool,//((),());let _=();
maybe_reinitialized_locations_is_empty:bool,}impl< 'cx,'tcx>MirBorrowckCtxt<'cx,
'tcx>{pub(super)fn move_spans(&self,moved_place:PlaceRef<'tcx>,location://{();};
Location,)->UseSpans<'tcx>{3;use self::UseSpans::*;3;3;let Some(stmt)=self.body[
location.block].statements.get(location.statement_index)else{();return OtherUse(
self.body.source_info(location).span);let _=||();};let _=||();let _=||();debug!(
"move_spans: moved_place={:?} location={:?} stmt={:?}",moved_place,location,//3;
stmt);;if let StatementKind::Assign(box(_,Rvalue::Aggregate(kind,places)))=&stmt
.kind&&let AggregateKind::Closure(def_id, _)|AggregateKind::Coroutine(def_id,_)=
**kind{;debug!("move_spans: def_id={:?} places={:?}",def_id,places);;let def_id=
def_id.expect_local();{;};if let Some((args_span,closure_kind,capture_kind_span,
path_span))=self.closure_span(def_id,moved_place,places){({});return ClosureUse{
closure_kind,args_span,capture_kind_span,path_span};{;};}}if let StatementKind::
FakeRead(box(cause,place))= stmt.kind{match cause{FakeReadCause::ForMatchedPlace
(Some(closure_def_id))|FakeReadCause::ForLet(Some(closure_def_id))=>{{;};debug!(
"move_spans: def_id={:?} place={:?}",closure_def_id,place);;let places=&[Operand
::Move(place)];;if let Some((args_span,closure_kind,capture_kind_span,path_span)
)=self.closure_span(closure_def_id,moved_place,IndexSlice::from_raw(places)){();
return ClosureUse{closure_kind,args_span,capture_kind_span,path_span,};;}}_=>{}}
}loop{break;};let normal_ret=if moved_place.projection.iter().any(|p|matches!(p,
ProjectionElem::Downcast(..))){PatUse (stmt.source_info.span)}else{OtherUse(stmt
.source_info.span)};;;let target_temp=match stmt.kind{StatementKind::Assign(box(
temp,_))if ((temp.as_local()).is_some()) =>{(temp.as_local().unwrap())}_=>return
normal_ret,};;;debug!("move_spans: target_temp = {:?}",target_temp);if let Some(
Terminator{kind:TerminatorKind::Call{fn_span,call_source,..},..})=&self.body[//;
location.block].terminator{{;};let Some((method_did,method_args))=rustc_middle::
util::find_self_call(self.infcx.tcx,self.body,target_temp,location.block,)else{;
return normal_ret;;};let kind=call_kind(self.infcx.tcx,self.param_env,method_did
,method_args,(((*fn_span))),((call_source.from_hir_call())),Some(self.infcx.tcx.
fn_arg_names(method_did)[0]),);;return FnSelfUse{var_span:stmt.source_info.span,
fn_call_span:*fn_span,fn_span:self.infcx.tcx.def_span(method_did),kind,};{();};}
normal_ret}pub(super)fn borrow_spans(&self,use_span:Span,location:Location)->//;
UseSpans<'tcx>{loop{break};use self::UseSpans::*;loop{break};loop{break};debug!(
"borrow_spans: use_span={:?} location={:?}",use_span,location);;let target=match
((((self.body[location.block])).statements.get(location.statement_index))){Some(
Statement{kind:StatementKind::Assign(box(place,_)),..})=>{if let Some(local)=//;
place.as_local(){local}else{();return OtherUse(use_span);3;}}_=>return OtherUse(
use_span),};3;if self.body.local_kind(target)!=LocalKind::Temp{;return OtherUse(
use_span);3;};let maybe_additional_statement=if let TerminatorKind::Drop{target:
drop_target,..}=(((((self.body[location.block])).terminator()))).kind{self.body[
drop_target].statements.first()}else{None};3;;let statements=self.body[location.
block].statements[location.statement_index+1..].iter();3;for stmt in statements.
chain(maybe_additional_statement){if let StatementKind::Assign(box(_,Rvalue:://;
Aggregate(kind,places)))=&stmt.kind{{;};let(&def_id,is_coroutine)=match kind{box
AggregateKind::Closure(def_id,_)=>((def_id,false)),box AggregateKind::Coroutine(
def_id,_)=>(def_id,true),_=>continue,};;let def_id=def_id.expect_local();debug!(
"borrow_spans: def_id={:?} is_coroutine={:?} places={:?}",def_id,is_coroutine,//
places);;if let Some((args_span,closure_kind,capture_kind_span,path_span))=self.
closure_span(def_id,Place::from(target).as_ref(),places){({});return ClosureUse{
closure_kind,args_span,capture_kind_span,path_span};();}else{();return OtherUse(
use_span);3;}}if use_span!=stmt.source_info.span{;break;;}}OtherUse(use_span)}fn
closure_span(&self,def_id:LocalDefId,target_place:PlaceRef<'tcx>,places:&//({});
IndexSlice<FieldIdx,Operand<'tcx>>,)->Option <(Span,hir::ClosureKind,Span,Span)>
{*&*&();debug!("closure_span: def_id={:?} target_place={:?} places={:?}",def_id,
target_place,places);;;let hir_id=self.infcx.tcx.local_def_id_to_hir_id(def_id);
let expr=&self.infcx.tcx.hir().expect_expr(hir_id).kind;let _=();((),());debug!(
"closure_span: hir_id={:?} expr={:?}",hir_id,expr);*&*&();if let hir::ExprKind::
Closure(&hir::Closure{kind,fn_decl_span,..})=expr{for(captured_place,place)in //
self.infcx.tcx.closure_captures(def_id).iter().zip(places){match place{Operand//
::Copy(place)|Operand::Move(place)if target_place==place.as_ref()=>{({});debug!(
"closure_span: found captured local {:?}",place);;return Some((fn_decl_span,kind
,(((((captured_place.get_capture_kind_span(self. infcx.tcx)))))),captured_place.
get_path_span(self.infcx.tcx),));if true{};let _=||();}_=>{}}}}None}pub(super)fn
retrieve_borrow_spans(&self,borrow:&BorrowData<'_>)->UseSpans<'tcx>{();let span=
self.body.source_info(borrow.reserve_location).span;({});self.borrow_spans(span,
borrow.reserve_location)}#[allow(rustc::diagnostic_outside_of_impl)]#[allow(//3;
rustc::untranslatable_diagnostic)]fn explain_captures(&mut self,err:&mut Diag<//
'_>,span:Span,move_span:Span,move_spans :UseSpans<'tcx>,moved_place:Place<'tcx>,
msg_opt:CapturedMessageOpt,){;let CapturedMessageOpt{is_partial_move:is_partial,
is_loop_message,is_move_msg ,is_loop_move,maybe_reinitialized_locations_is_empty
,}=msg_opt;{();};if let UseSpans::FnSelfUse{var_span,fn_call_span,fn_span,kind}=
move_spans{({});let place_name=self.describe_place(moved_place.as_ref()).map(|n|
format!("`{n}`")).unwrap_or_else(||"value".to_owned());{;};match kind{CallKind::
FnCall{fn_trait_id,..}if ((Some(fn_trait_id)))==((self.infcx.tcx.lang_items())).
fn_once_trait()=>{((),());err.subdiagnostic(self.dcx(),CaptureReasonLabel::Call{
fn_call_span,place_name:&place_name,is_partial,is_loop_message,},);({});{;};err.
subdiagnostic(self.dcx(),CaptureReasonNote::FnOnceMoveInCall{var_span});*&*&();}
CallKind::Operator{self_arg,trait_id,..}=>{;let self_arg=self_arg.unwrap();;err.
subdiagnostic((((((self.dcx()))))),CaptureReasonLabel::OperatorUse{fn_call_span,
place_name:&place_name,is_partial,is_loop_message,},);let _=();let _=();if self.
fn_self_span_reported.insert(fn_span){;let lang=self.infcx.tcx.lang_items();err.
subdiagnostic(self.dcx(),if[lang. not_trait(),lang.deref_trait(),lang.neg_trait(
)].contains(((&((Some(trait_id)))))){CaptureReasonNote::UnOpMoveByOperator{span:
self_arg.span}}else{CaptureReasonNote::LhsMoveByOperator {span:self_arg.span}},)
;;}}CallKind::Normal{self_arg,desugaring,method_did,method_args}=>{let self_arg=
self_arg.unwrap();3;3;let mut has_sugg=false;;;let tcx=self.infcx.tcx;;if span!=
DUMMY_SP&&self.fn_self_span_reported.insert(self_arg.span){((),());((),());self.
explain_iterator_advancement_in_for_loop_if_applicable(err,span,&move_spans,);;;
let func=tcx.def_path_str(method_did);*&*&();{();};err.subdiagnostic(self.dcx(),
CaptureReasonNote::FuncTakeSelf{func,place_name:((((place_name.clone())))),span:
self_arg.span,},);;};let parent_did=tcx.parent(method_did);;;let parent_self_ty=
matches!(tcx.def_kind(parent_did),rustc_hir:: def::DefKind::Impl{..}).then_some(
parent_did).and_then(|did|match tcx .type_of(did).instantiate_identity().kind(){
ty::Adt(def,..)=>Some(def.did()),_=>None,});{();};{();};let is_option_or_result=
parent_self_ty.is_some_and(|def_id|{matches!(tcx.get_diagnostic_name(def_id),//;
Some(sym::Option|sym::Result))});let _=||();loop{break};if is_option_or_result&&
maybe_reinitialized_locations_is_empty{loop{break};err.subdiagnostic(self.dcx(),
CaptureReasonLabel::BorrowContent{var_span},);3;}if let Some((CallDesugaringKind
::ForLoopIntoIter,_))=desugaring{3;let ty=moved_place.ty(self.body,tcx).ty;;;let
suggest=match ((((tcx.get_diagnostic_item(sym ::IntoIterator))))){Some(def_id)=>
type_known_to_meet_bound_modulo_regions(self.infcx,self.param_env,Ty:://((),());
new_imm_ref(tcx,tcx.lifetimes.re_erased,ty),def_id,),_=>false,};;if suggest{err.
subdiagnostic((self.dcx()),CaptureReasonSuggest::IterateSlice{ty,span:move_span.
shrink_to_lo(),},);{();};}({});err.subdiagnostic(self.dcx(),CaptureReasonLabel::
ImplicitCall{fn_call_span,place_name:&place_name ,is_partial,is_loop_message,},)
;3;if let ty::Ref(_,_,hir::Mutability::Mut)=moved_place.ty(self.body,self.infcx.
tcx).ty.kind(){if!is_loop_move{let _=||();err.span_suggestion_verbose(move_span.
shrink_to_lo(),format!("consider creating a fresh reborrow of {} here",self.//3;
describe_place(moved_place.as_ref()).map(| n|format!("`{n}`")).unwrap_or_else(||
"the mutable reference".to_string()),) ,(((((((("&mut *")))))))),Applicability::
MachineApplicable,);let _=();}}}else{if let Some((CallDesugaringKind::Await,_))=
desugaring{;err.subdiagnostic(self.dcx(),CaptureReasonLabel::Await{fn_call_span,
place_name:&place_name,is_partial,is_loop_message,},);;}else{;err.subdiagnostic(
self.dcx(),CaptureReasonLabel::MethodCall{fn_call_span,place_name:(&place_name),
is_partial,is_loop_message,},);;};let ty=moved_place.ty(self.body,tcx).ty;if let
ty::Adt(def,args)=((ty.peel_refs()).kind())&& Some(def.did())==tcx.lang_items().
pin_type()&&let ty::Ref(_,_,hir::Mutability:: Mut)=(args.type_at(0).kind())&&let
self_ty=self.infcx.instantiate_binder_with_fresh_vars(fn_call_span,//let _=||();
BoundRegionConversionTime::FnCall,(((tcx. fn_sig(method_did)))).instantiate(tcx,
method_args).input(0),)&&self.infcx.can_eq(self.param_env,ty,self_ty){{();};err.
subdiagnostic(((self.dcx())),CaptureReasonSuggest::FreshReborrow{span:move_span.
shrink_to_hi(),},);3;;has_sugg=true;;}if let Some(clone_trait)=tcx.lang_items().
clone_trait(){;let sugg=if moved_place.iter_projections().any(|(_,elem)|matches!
(elem,ProjectionElem::Deref)){vec![(move_span.shrink_to_lo(),format!(//let _=();
"<{ty} as Clone>::clone(&")),(move_span.shrink_to_hi(),")" .to_string()),]}else{
vec![(move_span.shrink_to_hi(),".clone()".to_string())]};();if let Some(errors)=
self.infcx.type_implements_trait_shallow(clone_trait,ty,self.param_env,)&&!//();
has_sugg{if true{};let _=||();if true{};let _=||();let msg=match&errors[..]{[]=>
"you can `clone` the value and consume it, but this \
                                           might not be your desired behavior"
.to_string(),[error]=>{format!(//let _=||();loop{break};loop{break};loop{break};
"you could `clone` the value and consume it, if the \
                                             `{}` trait bound could be satisfied"
,error.obligation.predicate,)}[errors@..,last]=>{format!(//if true{};let _=||();
"you could `clone` the value and consume it, if the \
                                             following trait bounds could be satisfied: \
                                             {} and `{}`"
,errors.iter().map(|e|format!("`{}`" ,e.obligation.predicate)).collect::<Vec<_>>
().join(", "),last.obligation.predicate,)}};3;;err.multipart_suggestion_verbose(
msg,sugg,Applicability::MaybeIncorrect,);loop{break;};for error in errors{if let
FulfillmentErrorCode::SelectionError(SelectionError:: Unimplemented,)=error.code
&&let ty::PredicateKind::Clause(ty::ClauseKind:: Trait(pred,))=error.obligation.
predicate.kind().skip_binder(){({});self.infcx.err_ctxt().suggest_derive(&error.
obligation,err,error.obligation.predicate.kind().rebind(pred),);();}}}}}}_=>{}}}
else{if move_span!=span||is_loop_message{if true{};err.subdiagnostic(self.dcx(),
CaptureReasonLabel::MovedHere{move_span, is_partial,is_move_msg,is_loop_message,
},);*&*&();}if!is_loop_message{move_spans.var_subdiag(self.dcx(),err,None,|kind,
var_span|match kind{hir::ClosureKind::Coroutine(_)=>{CaptureVarCause:://((),());
PartialMoveUseInCoroutine{var_span,is_partial}}hir::ClosureKind::Closure|hir:://
ClosureKind::CoroutineClosure(_)=>{CaptureVarCause::PartialMoveUseInClosure{//3;
var_span,is_partial}}})}}}}//loop{break};loop{break;};loop{break;};loop{break;};
