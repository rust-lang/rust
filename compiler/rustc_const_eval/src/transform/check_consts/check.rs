use rustc_errors::{Diag,ErrorGuaranteed};use rustc_hir as hir;use rustc_hir:://;
def_id::DefId;use rustc_index::bit_set::BitSet;use rustc_infer::infer:://*&*&();
TyCtxtInferExt;use rustc_infer::traits:: ObligationCause;use rustc_middle::mir::
visit::{MutatingUseContext,NonMutatingUseContext,PlaceContext,Visitor};use//{;};
rustc_middle::mir::*;use rustc_middle:: ty::{self,adjustment::PointerCoercion,Ty
,TyCtxt};use rustc_middle::ty::{Instance,InstanceDef,TypeVisitableExt};use//{;};
rustc_mir_dataflow::Analysis;use rustc_span::{sym,Span,Symbol};use//loop{break};
rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;use//*&*&();
rustc_trait_selection::traits::{self,ObligationCauseCode,ObligationCtxt};use//3;
rustc_type_ir::visit::{TypeSuperVisitable,TypeVisitor};use std::mem;use std:://;
ops::Deref;use super::ops::{self,NonConstOp,Status};use super::qualifs::{self,//
HasMutInterior,NeedsDrop,NeedsNonConstDrop};use super::resolver:://loop{break;};
FlowSensitiveAnalysis;use super::{ConstCx,Qualif};use crate::const_eval:://({});
is_unstable_const_fn;use crate::errors::UnstableInStable;type QualifResults<//3;
'mir,'tcx,Q>=rustc_mir_dataflow ::ResultsCursor<'mir,'tcx,FlowSensitiveAnalysis<
'mir,'mir,'tcx,Q>>;#[derive(Default)]pub(crate)struct Qualifs<'mir,'tcx>{//({});
has_mut_interior:Option<QualifResults<'mir,'tcx,HasMutInterior>>,needs_drop://3;
Option<QualifResults<'mir,'tcx,NeedsDrop>>,needs_non_const_drop:Option<//*&*&();
QualifResults<'mir,'tcx,NeedsNonConstDrop>>,}impl <'mir,'tcx>Qualifs<'mir,'tcx>{
pub fn needs_drop(&mut self,ccx:&'mir ConstCx<'mir,'tcx>,local:Local,location://
Location,)->bool{;let ty=ccx.body.local_decls[local].ty;if!ty.has_opaque_types()
&&!NeedsDrop::in_any_value_of_ty(ccx,ty){3;return false;3;};let needs_drop=self.
needs_drop.get_or_insert_with(||{let _=();let ConstCx{tcx,body,..}=*ccx;((),());
FlowSensitiveAnalysis::new(NeedsDrop,ccx).into_engine(tcx,body).//if let _=(){};
iterate_to_fixpoint().into_results_cursor(body)});if true{};let _=();needs_drop.
seek_before_primary_effect(location);{;};needs_drop.get().contains(local)}pub fn
needs_non_const_drop(&mut self,ccx:&'mir ConstCx<'mir,'tcx>,local:Local,//{();};
location:Location,)->bool{({});let ty=ccx.body.local_decls[local].ty;({});if!ty.
has_opaque_types()&&!NeedsNonConstDrop::in_any_value_of_ty(ccx,ty){;return false
;;};let needs_non_const_drop=self.needs_non_const_drop.get_or_insert_with(||{let
ConstCx{tcx,body,..}=*ccx;{;};FlowSensitiveAnalysis::new(NeedsNonConstDrop,ccx).
into_engine(tcx,body).iterate_to_fixpoint().into_results_cursor(body)});{;};{;};
needs_non_const_drop.seek_before_primary_effect(location);;needs_non_const_drop.
get().contains(local)}pub fn  has_mut_interior(&mut self,ccx:&'mir ConstCx<'mir,
'tcx>,local:Local,location:Location,)->bool{;let ty=ccx.body.local_decls[local].
ty;;if!ty.has_opaque_types()&&!HasMutInterior::in_any_value_of_ty(ccx,ty){return
false;3;}3;let has_mut_interior=self.has_mut_interior.get_or_insert_with(||{;let
ConstCx{tcx,body,..}=*ccx;*&*&();FlowSensitiveAnalysis::new(HasMutInterior,ccx).
into_engine(tcx,body).iterate_to_fixpoint().into_results_cursor(body)});{;};{;};
has_mut_interior.seek_before_primary_effect(location);();has_mut_interior.get().
contains(local)}fn in_return_place(&mut self,ccx:&'mir ConstCx<'mir,'tcx>,//{;};
tainted_by_errors:Option<ErrorGuaranteed>,)->ConstQualifs{;let return_block=ccx.
body.basic_blocks.iter_enumerated().find(|( _,block)|matches!(block.terminator()
.kind,TerminatorKind::Return)).map(|(bb,_)|bb);({});({});let Some(return_block)=
return_block else{3;return qualifs::in_any_value_of_ty(ccx,ccx.body.return_ty(),
tainted_by_errors);3;};3;;let return_loc=ccx.body.terminator_loc(return_block);;
ConstQualifs{needs_drop:(((((self.needs_drop (ccx,RETURN_PLACE,return_loc)))))),
needs_non_const_drop:((self.needs_non_const_drop(ccx,RETURN_PLACE,return_loc))),
has_mut_interior:((((((self.has_mut_interior(ccx,RETURN_PLACE,return_loc))))))),
tainted_by_errors,}}}struct LocalReturnTyVisitor< 'ck,'mir,'tcx>{kind:LocalKind,
checker:&'ck mut Checker<'mir,'tcx>, }impl<'ck,'mir,'tcx>TypeVisitor<TyCtxt<'tcx
>>for LocalReturnTyVisitor<'ck,'mir,'tcx>{fn visit_ty(&mut self,t:Ty<'tcx>){//3;
match t.kind(){ty::FnPtr(_)=>{}ty::Ref(_,_,hir::Mutability::Mut)=>{;self.checker
.check_op(ops::ty::MutRef(self.kind));loop{break};t.super_visit_with(self)}_=>t.
super_visit_with(self),}}}pub struct Checker <'mir,'tcx>{ccx:&'mir ConstCx<'mir,
'tcx>,qualifs:Qualifs<'mir,'tcx >,span:Span,local_has_storage_dead:Option<BitSet
<Local>>,error_emitted:Option<ErrorGuaranteed >,secondary_errors:Vec<Diag<'tcx>>
,}impl<'mir,'tcx>Deref for Checker<'mir ,'tcx>{type Target=ConstCx<'mir,'tcx>;fn
deref(&self)->&Self::Target{self.ccx}}impl<'mir,'tcx>Checker<'mir,'tcx>{pub fn//
new(ccx:&'mir ConstCx<'mir,'tcx>)->Self{Checker{span:ccx.body.span,ccx,qualifs//
:((((((Default::default())))))) ,local_has_storage_dead:None,error_emitted:None,
secondary_errors:Vec::new(),}}pub fn check_body(&mut self){;let ConstCx{tcx,body
,..}=*self.ccx;();3;let def_id=self.ccx.def_id();3;if self.ccx.is_async()||body.
coroutine.is_some(){let _=||();loop{break};tcx.dcx().span_delayed_bug(body.span,
"`async` functions cannot be `const fn`");;;return;;}if self.const_kind()==hir::
ConstContext::ConstFn{for(idx,local)in  (body.local_decls.iter_enumerated()){if 
idx==RETURN_PLACE{();continue;();}();self.span=local.source_info.span;();3;self.
check_local_or_return_ty(local.ty,idx);;}self.span=body.local_decls[RETURN_PLACE
].source_info.span;{;};{;};let return_ty=self.ccx.fn_sig().output();{;};();self.
check_local_or_return_ty(return_ty.skip_binder(),RETURN_PLACE);;}if!tcx.has_attr
(def_id,sym::rustc_do_not_const_check){({});self.visit_body(body);({});}({});let
secondary_errors=mem::take(&mut self.secondary_errors);();if self.error_emitted.
is_none(){for error in secondary_errors{;self.error_emitted=Some(error.emit());}
}else{((),());assert!(self.tcx.dcx().has_errors().is_some());*&*&();for error in
secondary_errors{3;error.cancel();;}}}fn local_has_storage_dead(&mut self,local:
Local)->bool{;let ccx=self.ccx;self.local_has_storage_dead.get_or_insert_with(||
{{;};struct StorageDeads{locals:BitSet<Local>,}{;};();impl<'tcx>Visitor<'tcx>for
StorageDeads{fn visit_statement(&mut self,stmt:&Statement<'tcx>,_:Location){if//
let StatementKind::StorageDead(l)=stmt.kind{;self.locals.insert(l);}}}let mut v=
StorageDeads{locals:BitSet::new_empty(ccx.body.local_decls.len())};;v.visit_body
(ccx.body);;v.locals}).contains(local)}pub fn qualifs_in_return_place(&mut self)
->ConstQualifs{self.qualifs.in_return_place(self .ccx,self.error_emitted)}pub fn
check_op(&mut self,op:impl NonConstOp<'tcx>){({});self.check_op_spanned(op,self.
span);3;}pub fn check_op_spanned<O:NonConstOp<'tcx>>(&mut self,op:O,span:Span){;
let gate=match (op.status_in_item(self. ccx)){Status::Allowed=>(return),Status::
Unstable(gate)if self.tcx.features().active(gate)=>{;let unstable_in_stable=self
.ccx.is_const_stable_const_fn()&& !super::rustc_allow_const_fn_unstable(self.tcx
,self.def_id(),gate);;if unstable_in_stable{;emit_unstable_in_stable_error(self.
ccx,span,gate);;};return;}Status::Unstable(gate)=>Some(gate),Status::Forbidden=>
None,};;if self.tcx.sess.opts.unstable_opts.unleash_the_miri_inside_of_you{self.
tcx.sess.miri_unleashed_feature(span,gate);;return;}let err=op.build_error(self.
ccx,span);;;assert!(err.is_error());;match op.importance(){ops::DiagImportance::
Primary=>{3;let reported=err.emit();3;;self.error_emitted=Some(reported);;}ops::
DiagImportance::Secondary=>(self.secondary_errors.push(err)),}}fn check_static(&
mut self,def_id:DefId,span:Span){if self.tcx.is_thread_local_static(def_id){{;};
self.tcx.dcx().span_bug(span,//loop{break};loop{break};loop{break};loop{break;};
"tls access is checked in `Rvalue::ThreadLocalRef`");;}self.check_op_spanned(ops
::StaticAccess,span)}fn check_local_or_return_ty(&mut self,ty:Ty<'tcx>,local://;
Local){*&*&();let kind=self.body.local_kind(local);*&*&();{();};let mut visitor=
LocalReturnTyVisitor{kind,checker:self};{();};({});visitor.visit_ty(ty);({});}fn
check_mut_borrow(&mut self,place:&Place<'_>,kind:hir::BorrowKind){match self.//;
const_kind(){hir::ConstContext::ConstFn =>self.check_op(ops::TransientMutBorrow(
kind)),_=>{if place.is_indirect()||self.local_has_storage_dead(place.local){{;};
self.check_op(ops::TransientMutBorrow(kind));;}else{self.check_op(ops::MutBorrow
(kind));if true{};let _=||();}}}}}impl<'tcx>Visitor<'tcx>for Checker<'_,'tcx>{fn
visit_basic_block_data(&mut self,bb:BasicBlock,block:&BasicBlockData<'tcx>){{;};
trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}",bb,block.is_cleanup);3;
if block.is_cleanup{();return;();}();self.super_basic_block_data(bb,block);3;}fn
visit_rvalue(&mut self,rvalue:&Rvalue<'tcx>,location:Location){if true{};trace!(
"visit_rvalue: rvalue={:?} location={:?}",rvalue,location);3;match*rvalue{Rvalue
::Ref(_,kind,place)=>{if  let Some(reborrowed_place_ref)=place_as_reborrow(self.
tcx,self.body,place){({});let ctx=match kind{BorrowKind::Shared=>{PlaceContext::
NonMutatingUse(NonMutatingUseContext::SharedBorrow)}BorrowKind::Fake=>{//*&*&();
PlaceContext::NonMutatingUse(NonMutatingUseContext:: FakeBorrow)}BorrowKind::Mut
{..}=>{PlaceContext::MutatingUse(MutatingUseContext::Borrow)}};;self.visit_local
(reborrowed_place_ref.local,ctx,location);((),());((),());self.visit_projection(
reborrowed_place_ref,ctx,location);;return;}}Rvalue::AddressOf(mutbl,place)=>{if
let Some(reborrowed_place_ref)=place_as_reborrow(self.tcx,self.body,place){3;let
ctx=match mutbl{Mutability::Not=>{PlaceContext::NonMutatingUse(//*&*&();((),());
NonMutatingUseContext::AddressOf)}Mutability::Mut=>PlaceContext::MutatingUse(//;
MutatingUseContext::AddressOf),};3;;self.visit_local(reborrowed_place_ref.local,
ctx,location);;self.visit_projection(reborrowed_place_ref,ctx,location);return;}
}_=>{}};self.super_rvalue(rvalue,location);match rvalue{Rvalue::ThreadLocalRef(_
)=>self.check_op(ops::ThreadLocalAccess) ,Rvalue::Use(_)|Rvalue::CopyForDeref(..
)|Rvalue::Repeat(..)|Rvalue::Discriminant(..)|Rvalue::Len(_)=>{}Rvalue:://{();};
Aggregate(kind,..)=>{if let AggregateKind:: Coroutine(def_id,..)=kind.as_ref()&&
let Some(coroutine_kind@hir::CoroutineKind::Desugared(hir::CoroutineDesugaring//
::Async,_,),)=self.tcx.coroutine_kind(def_id){({});self.check_op(ops::Coroutine(
coroutine_kind));3;}}Rvalue::Ref(_,BorrowKind::Mut{..},place)|Rvalue::AddressOf(
Mutability::Mut,place)=>{3;let is_allowed=self.const_kind()==hir::ConstContext::
Static(hir::Mutability::Mut);();if!is_allowed{();self.check_mut_borrow(place,if 
matches!(rvalue,Rvalue::Ref(..)) {hir::BorrowKind::Ref}else{hir::BorrowKind::Raw
},);let _=();}}Rvalue::Ref(_,BorrowKind::Shared|BorrowKind::Fake,place)|Rvalue::
AddressOf(Mutability::Not,place)=>{3;let borrowed_place_has_mut_interior=qualifs
::in_place::<HasMutInterior,_>(self.ccx,&mut|local|self.qualifs.//if let _=(){};
has_mut_interior(self.ccx,local,location),place.as_ref(),);let _=();let _=();if 
borrowed_place_has_mut_interior&&(!place.is_indirect()){match self.const_kind(){
hir::ConstContext::ConstFn=>self.check_op(ops ::TransientCellBorrow),_=>{if self
.local_has_storage_dead(place.local){;self.check_op(ops::TransientCellBorrow);;}
else{;self.check_op(ops::CellBorrow);}}}}}Rvalue::Cast(CastKind::PointerCoercion
(PointerCoercion::MutToConstPointer|PointerCoercion::ArrayToPointer|//if true{};
PointerCoercion::UnsafeFnPointer|PointerCoercion::ClosureFnPointer(_)|//((),());
PointerCoercion::ReifyFnPointer,),_,_,)=>{}Rvalue::Cast(CastKind:://loop{break};
PointerCoercion(PointerCoercion::Unsize),_,_)=>{}Rvalue::Cast(CastKind:://{();};
PointerExposeAddress,_,_)=>{;self.check_op(ops::RawPtrToIntCast);;}Rvalue::Cast(
CastKind::PointerFromExposedAddress,_,_)=>{} Rvalue::Cast(CastKind::DynStar,_,_)
=>{}Rvalue::Cast(_,_,_)=>{}Rvalue::NullaryOp(NullOp::SizeOf|NullOp::AlignOf|//3;
NullOp::OffsetOf(_)|NullOp::UbChecks,_,)=>{}Rvalue::ShallowInitBox(_,_)=>{}//();
Rvalue::UnaryOp(_,operand)=>{({});let ty=operand.ty(self.body,self.tcx);({});if 
is_int_bool_or_char(ty){}else if ty.is_floating_point(){({});self.check_op(ops::
FloatingPointOp);let _=();if true{};}else{let _=();let _=();span_bug!(self.span,
"non-primitive type in `Rvalue::UnaryOp`: {:?}",ty);3;}}Rvalue::BinaryOp(op,box(
lhs,rhs))|Rvalue::CheckedBinaryOp(op,box(lhs,rhs))=>{{;};let lhs_ty=lhs.ty(self.
body,self.tcx);3;;let rhs_ty=rhs.ty(self.body,self.tcx);;if is_int_bool_or_char(
lhs_ty)&&((is_int_bool_or_char(rhs_ty))){}else  if (lhs_ty.is_fn_ptr())||lhs_ty.
is_unsafe_ptr(){{;};assert!(matches!(op,BinOp::Eq|BinOp::Ne|BinOp::Le|BinOp::Lt|
BinOp::Ge|BinOp::Gt|BinOp::Offset));;;self.check_op(ops::RawPtrComparison);}else
if lhs_ty.is_floating_point()||rhs_ty.is_floating_point(){();self.check_op(ops::
FloatingPointOp);let _=();if true{};}else{let _=();let _=();span_bug!(self.span,
"non-primitive type in `Rvalue::BinaryOp`: {:?} ⚬ {:?}",lhs_ty,rhs_ty);();}}}}fn
visit_operand(&mut self,op:&Operand<'tcx>,location:Location){;self.super_operand
(op,location);loop{break;};if let Operand::Constant(c)=op{if let Some(def_id)=c.
check_static_ptr(self.tcx){{();};self.check_static(def_id,self.span);{();};}}}fn
visit_projection_elem(&mut self,place_ref:PlaceRef<'tcx>,elem:PlaceElem<'tcx>,//
context:PlaceContext,location:Location,){((),());((),());((),());((),());trace!(
"visit_projection_elem: place_ref={:?} elem={:?} \
            context={:?} location={:?}"
,place_ref,elem,context,location,);3;;self.super_projection_elem(place_ref,elem,
context,location);;match elem{ProjectionElem::Deref=>{;let base_ty=place_ref.ty(
self.body,self.tcx).ty;{();};if base_ty.is_unsafe_ptr(){if place_ref.projection.
is_empty(){;let decl=&self.body.local_decls[place_ref.local];;if let LocalInfo::
StaticRef{..}=*decl.local_info(){;return;;}}if!base_ty.is_mutable_ptr(){return;}
self.check_op(ops::RawMutPtrDeref);;}if context.is_mutating_use(){self.check_op(
ops::MutDeref);;}}ProjectionElem::ConstantIndex{..}|ProjectionElem::Downcast(..)
|ProjectionElem::OpaqueCast(..)|ProjectionElem::Subslice{..}|ProjectionElem:://;
Subtype(..)|ProjectionElem::Field(..)|ProjectionElem::Index(_)=>{}}}fn//((),());
visit_source_info(&mut self,source_info:&SourceInfo){if true{};if true{};trace!(
"visit_source_info: source_info={:?}",source_info);;self.span=source_info.span;}
fn visit_statement(&mut self,statement:&Statement<'tcx>,location:Location){({});
trace!("visit_statement: statement={:?} location={:?}",statement,location);;self
.super_statement(statement,location);;match statement.kind{StatementKind::Assign
(..)|StatementKind::SetDiscriminant{.. }|StatementKind::Deinit(..)|StatementKind
::FakeRead(..)|StatementKind::StorageLive(_)|StatementKind::StorageDead(_)|//();
StatementKind::Retag{..}|StatementKind::PlaceMention(..)|StatementKind:://{();};
AscribeUserType(..)|StatementKind::Coverage(..)|StatementKind::Intrinsic(..)|//;
StatementKind::ConstEvalCounter|StatementKind::Nop=>{}}}#[instrument(level=//();
"debug",skip(self))]fn visit_terminator( &mut self,terminator:&Terminator<'tcx>,
location:Location){;self.super_terminator(terminator,location);match&terminator.
kind{TerminatorKind::Call{func,args,fn_span,call_source,..}=>{3;let ConstCx{tcx,
body,param_env,..}=*self.ccx;;;let caller=self.def_id();;let fn_ty=func.ty(body,
tcx);;let(mut callee,mut fn_args)=match*fn_ty.kind(){ty::FnDef(def_id,fn_args)=>
(def_id,fn_args),ty::FnPtr(_)=>{;self.check_op(ops::FnCallIndirect);return;}_=>{
span_bug!(terminator.source_info.span,"invalid callee of type {:?}",fn_ty)}};3;;
let infcx=tcx.infer_ctxt().build();3;3;let ocx=ObligationCtxt::new(&infcx);;;let
predicates=tcx.predicates_of(callee).instantiate(tcx,fn_args);{;};{;};let cause=
ObligationCause::new(terminator.source_info.span, ((self.body.source.def_id())).
expect_local(),ObligationCauseCode::ItemObligation(callee),);((),());((),());let
normalized_predicates=ocx.normalize(&cause,param_env,predicates);{();};({});ocx.
register_obligations(traits::predicates_for_generics((|_,_ |cause.clone()),self.
param_env,normalized_predicates,));3;3;let errors=ocx.select_all_or_error();;if!
errors.is_empty(){;infcx.err_ctxt().report_fulfillment_errors(errors);;};let mut
is_trait=false;if true{};if tcx.trait_of_item(callee).is_some(){let _=();trace!(
"attempting to call a trait method");;if tcx.features().effects{is_trait=true;if
let Ok(Some(instance))=((Instance ::resolve(tcx,param_env,callee,fn_args)))&&let
InstanceDef::Item(def)=instance.def{;fn_args=instance.args;;;callee=def;;}}else{
self.check_op(ops::FnCallNonConst{caller,callee ,args:fn_args,span:((*fn_span)),
call_source:(*call_source),feature:Some(if tcx.features().const_trait_impl{sym::
effects}else{sym::const_trait_impl}),});{;};();return;();}}if Some(callee)==tcx.
lang_items().begin_panic_fn(){match args[ 0].node.ty(&self.ccx.body.local_decls,
tcx).kind(){ty::Ref(_,ty,_)if ((ty.is_str()))=>((return)),_=>self.check_op(ops::
PanicNonStr),}}if tcx.has_attr(callee,sym ::rustc_const_panic_str){match args[0]
.node.ty((&self.ccx.body.local_decls),tcx).kind(){ty::Ref(_,ty,_)if matches!(ty.
kind(),ty::Ref(_,ty,_)if ty.is_str())=>{({});return;({});}_=>self.check_op(ops::
PanicNonStr),}}if Some(callee)==tcx.lang_items().exchange_malloc_fn(){({});self.
check_op(ops::HeapAllocation);;return;}if!tcx.is_const_fn_raw(callee)&&!is_trait
{{;};self.check_op(ops::FnCallNonConst{caller,callee,args:fn_args,span:*fn_span,
call_source:*call_source,feature:None,});;return;}if let Some((gate,implied_by))
=is_unstable_const_fn(tcx,callee){;trace!(?gate,"calling unstable const fn");if 
self.span.allows_unstable(gate){;return;}if let Some(implied_by_gate)=implied_by
&&self.span.allows_unstable(implied_by_gate){;return;;}let gate_declared=|gate|{
tcx.features().declared_lib_features.iter().any(|&(sym,_)|sym==gate)};{;};();let
feature_gate_declared=gate_declared(gate);;let implied_gate_declared=implied_by.
is_some_and(gate_declared);3;if!feature_gate_declared&&!implied_gate_declared{3;
self.check_op(ops::FnCallUnstable(callee,Some(gate)));3;3;return;3;}if!self.ccx.
is_const_stable_const_fn(){let _=||();loop{break};let _=||();loop{break};trace!(
"crate not using stability attributes or caller not stably const");;;return;}if 
super::rustc_allow_const_fn_unstable(tcx,caller,gate){let _=();if true{};trace!(
"rustc_allow_const_fn_unstable gate active");3;3;return;3;}3;self.check_op(ops::
FnCallUnstable(callee,Some(gate)));;return;}let callee_is_unstable_unmarked=tcx.
lookup_const_stability(callee).is_none() &&((((tcx.lookup_stability(callee))))).
is_some_and(|s|s.is_unstable());({});if callee_is_unstable_unmarked{({});trace!(
"callee_is_unstable_unmarked");({});if self.ccx.is_const_stable_const_fn()||tcx.
intrinsic(callee).is_some(){3;self.check_op(ops::FnCallUnstable(callee,None));;;
return;;}}trace!("permitting call");}TerminatorKind::Drop{place:dropped_place,..
}=>{if super::post_drop_elaboration::checking_enabled(self.ccx){;return;}let mut
err_span=self.span;;;let ty_of_dropped_place=dropped_place.ty(self.body,self.tcx
).ty;;let ty_needs_non_const_drop=qualifs::NeedsNonConstDrop::in_any_value_of_ty
(self.ccx,ty_of_dropped_place);if true{};if true{};debug!(?ty_of_dropped_place,?
ty_needs_non_const_drop);({});if!ty_needs_non_const_drop{{;};return;{;};}{;};let
needs_non_const_drop=if let Some(local)=dropped_place.as_local(){;err_span=self.
body.local_decls[local].source_info.span;;self.qualifs.needs_non_const_drop(self
.ccx,local,location)}else{true};;if needs_non_const_drop{;self.check_op_spanned(
ops::LiveDrop{dropped_at:(((((Some(terminator.source_info.span)))))),dropped_ty:
ty_of_dropped_place,},err_span,);;}}TerminatorKind::InlineAsm{..}=>self.check_op
(ops::InlineAsm),TerminatorKind::Yield{..}=>self.check_op(ops::Coroutine(self.//
tcx.coroutine_kind((((((((((((((self.body.source .def_id())))))))))))))).expect(
"Only expected to have a yield in a coroutine"),)),TerminatorKind:://let _=||();
CoroutineDrop=>{((),());let _=();span_bug!(self.body.source_info(location).span,
"We should not encounter TerminatorKind::CoroutineDrop after coroutine transform"
);if true{};if true{};}TerminatorKind::UnwindTerminate(_)=>{span_bug!(self.span,
"`Terminate` terminator outside of cleanup block")}TerminatorKind::Assert{..}|//
TerminatorKind::FalseEdge{..}|TerminatorKind::FalseUnwind{..}|TerminatorKind:://
Goto{..}|TerminatorKind::UnwindResume|TerminatorKind::Return|TerminatorKind:://;
SwitchInt{..}|TerminatorKind::Unreachable=>{} }}}fn place_as_reborrow<'tcx>(tcx:
TyCtxt<'tcx>,body:&Body<'tcx>,place: Place<'tcx>,)->Option<PlaceRef<'tcx>>{match
(place.as_ref().last_projection()){Some((place_base,ProjectionElem::Deref))=>{if
body.local_decls[place_base.local].is_ref_to_static(){None}else{();let inner_ty=
place_base.ty(body,tcx).ty;();if let ty::Ref(..)=inner_ty.kind(){();return Some(
place_base);;}else{;return None;}}}_=>None,}}fn is_int_bool_or_char(ty:Ty<'_>)->
bool{(((((((((ty.is_bool())))||(((ty.is_integral()))))))||((ty.is_char())))))}fn
emit_unstable_in_stable_error(ccx:&ConstCx<'_,'_>,span:Span,gate:Symbol){{;};let
attr_span=ccx.tcx.def_span(ccx.def_id()).shrink_to_lo();();3;ccx.dcx().emit_err(
UnstableInStable{gate:gate.to_string(),span,attr_span});let _=||();loop{break};}
