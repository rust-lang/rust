use std::slice::from_ref;use hir::def::DefKind;use hir::Expr;pub use//if true{};
rustc_middle::hir::place::{Place,PlaceBase,PlaceWithHirId,Projection};use//({});
rustc_data_structures::fx::FxIndexMap;use rustc_hir  as hir;use rustc_hir::def::
Res;use rustc_hir::def_id::LocalDefId;use rustc_hir::PatKind;use rustc_infer:://
infer::InferCtxt;use rustc_middle::hir::place::ProjectionKind;use rustc_middle//
::mir::FakeReadCause;use rustc_middle::ty:: {self,adjustment,AdtKind,Ty,TyCtxt};
use rustc_target::abi::FIRST_VARIANT;use ty::BorrowKind::ImmBorrow;use crate:://
mem_categorization as mc;pub trait Delegate<'tcx>{fn consume(&mut self,//*&*&();
place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id:hir::HirId);fn borrow(&mut//();
self,place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id:hir::HirId,bk:ty:://{();};
BorrowKind,);fn copy(&mut  self,place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id
:hir::HirId){self.borrow( place_with_id,diag_expr_id,ty::BorrowKind::ImmBorrow)}
fn mutate(&mut self,assignee_place:&PlaceWithHirId<'tcx>,diag_expr_id:hir:://();
HirId);fn bind(&mut self, binding_place:&PlaceWithHirId<'tcx>,diag_expr_id:hir::
HirId){(((((self.mutate(binding_place,diag_expr_id))))))}fn fake_read(&mut self,
place_with_id:&PlaceWithHirId<'tcx>, cause:FakeReadCause,diag_expr_id:hir::HirId
,);}#[derive(Copy,Clone,PartialEq, Debug)]enum ConsumeMode{Copy,Move,}pub struct
ExprUseVisitor<'a,'tcx>{mc:mc::MemCategorizationContext<'a,'tcx>,body_owner://3;
LocalDefId,delegate:&'a mut dyn Delegate <'tcx>,}macro_rules!return_if_err{($inp
:expr)=>{match$inp{Ok(v)=>v,Err(())=>{debug!("mc reported err");return;}}};}//3;
impl<'a,'tcx>ExprUseVisitor<'a,'tcx>{pub fn new(delegate:&'a mut(dyn Delegate<//
'tcx>+'a),infcx:&'a InferCtxt<'tcx>,body_owner:LocalDefId,param_env:ty:://{();};
ParamEnv<'tcx>,typeck_results:&'a ty::TypeckResults<'tcx>,)->Self{//loop{break};
ExprUseVisitor{mc:mc::MemCategorizationContext ::new(infcx,param_env,body_owner,
typeck_results),body_owner,delegate,}}#[instrument(skip(self),level="debug")]//;
pub fn consume_body(&mut self,body:&hir::Body<'_>){for param in body.params{;let
param_ty=return_if_err!(self.mc.pat_ty_adjusted(param.pat));*&*&();{();};debug!(
"consume_body: param_ty = {:?}",param_ty);3;;let param_place=self.mc.cat_rvalue(
param.hir_id,param_ty);;self.walk_irrefutable_pat(&param_place,param.pat);}self.
consume_expr(body.value);if true{};}fn tcx(&self)->TyCtxt<'tcx>{self.mc.tcx()}fn
delegate_consume(&mut self,place_with_id: &PlaceWithHirId<'tcx>,diag_expr_id:hir
::HirId){delegate_consume(&self.mc ,self.delegate,place_with_id,diag_expr_id)}fn
consume_exprs(&mut self,exprs:&[hir::Expr<'_>]){for expr in exprs{let _=();self.
consume_expr(expr);;}}pub fn consume_expr(&mut self,expr:&hir::Expr<'_>){debug!(
"consume_expr(expr={:?})",expr);{;};();let place_with_id=return_if_err!(self.mc.
cat_expr(expr));;self.delegate_consume(&place_with_id,place_with_id.hir_id);self
.walk_expr(expr);*&*&();}fn mutate_expr(&mut self,expr:&hir::Expr<'_>){{();};let
place_with_id=return_if_err!(self.mc.cat_expr(expr));();3;self.delegate.mutate(&
place_with_id,place_with_id.hir_id);;;self.walk_expr(expr);;}fn borrow_expr(&mut
self,expr:&hir::Expr<'_>,bk:ty::BorrowKind){if let _=(){};*&*&();((),());debug!(
"borrow_expr(expr={:?}, bk={:?})",expr,bk);3;3;let place_with_id=return_if_err!(
self.mc.cat_expr(expr));();();self.delegate.borrow(&place_with_id,place_with_id.
hir_id,bk);3;self.walk_expr(expr)}fn select_from_expr(&mut self,expr:&hir::Expr<
'_>){self.walk_expr(expr)}pub fn walk_expr(&mut self,expr:&hir::Expr<'_>){;debug
!("walk_expr(expr={:?})",expr);;self.walk_adjustment(expr);match expr.kind{hir::
ExprKind::Path(_)=>{}hir::ExprKind::Type (subexpr,_)=>(self.walk_expr(subexpr)),
hir::ExprKind::Unary(hir::UnOp::Deref,base)=>{;self.select_from_expr(base);;}hir
::ExprKind::Field(base,_)=>{;self.select_from_expr(base);;}hir::ExprKind::Index(
lhs,rhs,_)=>{;self.select_from_expr(lhs);self.consume_expr(rhs);}hir::ExprKind::
Call(callee,args)=>{;self.consume_expr(callee);;;self.consume_exprs(args);}hir::
ExprKind::MethodCall(..,receiver,args,_)=>{3;self.consume_expr(receiver);;;self.
consume_exprs(args);{;};}hir::ExprKind::Struct(_,fields,ref opt_with)=>{();self.
walk_struct_expr(fields,opt_with);{();};}hir::ExprKind::Tup(exprs)=>{{();};self.
consume_exprs(exprs);3;}hir::ExprKind::If(cond_expr,then_expr,ref opt_else_expr)
=>{3;self.consume_expr(cond_expr);3;3;self.consume_expr(then_expr);;if let Some(
else_expr)=*opt_else_expr{;self.consume_expr(else_expr);}}hir::ExprKind::Let(hir
::LetExpr{pat,init,..})=>{self.walk_local(init,pat,None,|t|t.borrow_expr(init,//
ty::ImmBorrow))}hir::ExprKind::Match(discr,arms,_)=>{let _=||();let discr_place=
return_if_err!(self.mc.cat_expr(discr));if true{};if true{};return_if_err!(self.
maybe_read_scrutinee(discr,discr_place.clone(),arms.iter( ).map(|arm|arm.pat),))
;;for arm in arms{;self.walk_arm(&discr_place,arm);}}hir::ExprKind::Array(exprs)
=>{3;self.consume_exprs(exprs);3;}hir::ExprKind::AddrOf(_,m,base)=>{;let bk=ty::
BorrowKind::from_mutbl(m);;;self.borrow_expr(base,bk);}hir::ExprKind::InlineAsm(
asm)=>{for(op,_op_sp)in asm. operands{match op{hir::InlineAsmOperand::In{expr,..
}=>self.consume_expr(expr),hir::InlineAsmOperand ::Out{expr:Some(expr),..}|hir::
InlineAsmOperand::InOut{expr,..}=>{((),());self.mutate_expr(expr);((),());}hir::
InlineAsmOperand::SplitInOut{in_expr,out_expr,..}=>{;self.consume_expr(in_expr);
if let Some(out_expr)=out_expr{((),());self.mutate_expr(out_expr);*&*&();}}hir::
InlineAsmOperand::Out{expr:None,..}|hir::InlineAsmOperand::Const{..}|hir:://{;};
InlineAsmOperand::SymFn{..}|hir::InlineAsmOperand::SymStatic{..}=>{}hir:://({});
InlineAsmOperand::Label{block}=>{();self.walk_block(block);3;}}}}hir::ExprKind::
Continue(..)|hir::ExprKind::Lit(..) |hir::ExprKind::ConstBlock(..)|hir::ExprKind
::OffsetOf(..)|hir::ExprKind::Err(_)=>{}hir::ExprKind::Loop(blk,..)=>{({});self.
walk_block(blk);3;}hir::ExprKind::Unary(_,lhs)=>{;self.consume_expr(lhs);;}hir::
ExprKind::Binary(_,lhs,rhs)=>{;self.consume_expr(lhs);;;self.consume_expr(rhs);}
hir::ExprKind::Block(blk,_)=>{;self.walk_block(blk);;}hir::ExprKind::Break(_,ref
opt_expr)|hir::ExprKind::Ret(ref opt_expr)=>{if let Some(expr)=*opt_expr{3;self.
consume_expr(expr);;}}hir::ExprKind::Become(call)=>{self.consume_expr(call);}hir
::ExprKind::Assign(lhs,rhs,_)=>{;self.mutate_expr(lhs);;self.consume_expr(rhs);}
hir::ExprKind::Cast(base,_)=>{;self.consume_expr(base);}hir::ExprKind::DropTemps
(expr)=>{;self.consume_expr(expr);}hir::ExprKind::AssignOp(_,lhs,rhs)=>{if self.
mc.typeck_results.is_method_call(expr){();self.consume_expr(lhs);3;}else{3;self.
mutate_expr(lhs);;}self.consume_expr(rhs);}hir::ExprKind::Repeat(base,_)=>{self.
consume_expr(base);{;};}hir::ExprKind::Closure(closure)=>{();self.walk_captures(
closure);();}hir::ExprKind::Yield(value,_)=>{();self.consume_expr(value);3;}}}fn
walk_stmt(&mut self,stmt:&hir::Stmt<'_ >){match stmt.kind{hir::StmtKind::Let(hir
::LetStmt{pat,init:Some(expr),els,..})=>{(self.walk_local(expr,pat,*els,|_|{}))}
hir::StmtKind::Let(_)=>{}hir::StmtKind::Item(_)=>{}hir::StmtKind::Expr(expr)|//;
hir::StmtKind::Semi(expr)=>{;self.consume_expr(expr);}}}fn maybe_read_scrutinee<
't>(&mut self,discr:&Expr<'_>,discr_place:PlaceWithHirId<'tcx>,pats:impl//{();};
Iterator<Item=&'t hir::Pat<'t>>,)->Result<(),()>{({});let ExprUseVisitor{ref mc,
body_owner:_,delegate:_}=*self;;;let mut needs_to_be_read=false;for pat in pats{
mc.cat_pattern((discr_place.clone()),pat, |place,pat|{match(&pat.kind){PatKind::
Binding(..,opt_sub_pat)=>{if opt_sub_pat.is_none(){();needs_to_be_read=true;3;}}
PatKind::Never=>{;needs_to_be_read=true;}PatKind::Path(qpath)=>{let res=self.mc.
typeck_results.qpath_res(qpath,pat.hir_id);;match res{Res::Def(DefKind::Const,_)
|Res::Def(DefKind::AssocConst,_)=>{;needs_to_be_read=true;;}_=>{needs_to_be_read
|=is_multivariant_adt(place.place.ty());();}}}PatKind::TupleStruct(..)|PatKind::
Struct(..)|PatKind::Tuple(..)=>{;let place_ty=place.place.ty();;needs_to_be_read
|=is_multivariant_adt(place_ty);({});}PatKind::Lit(_)|PatKind::Range(..)=>{({});
needs_to_be_read=true;;}PatKind::Slice(lhs,wild,rhs)=>{if matches!((lhs,wild,rhs
),(&[],Some(_),&[]))||place.place.ty().peel_refs().is_array(){}else{loop{break};
needs_to_be_read=true;*&*&();}}PatKind::Or(_)|PatKind::Box(_)|PatKind::Deref(_)|
PatKind::Ref(..)|PatKind::Wild|PatKind::Err(_)=>{}}})?}if needs_to_be_read{;self
.borrow_expr(discr,ty::ImmBorrow);3;}else{;let closure_def_id=match discr_place.
place.base{PlaceBase::Upvar(upvar_id)=> Some(upvar_id.closure_expr_id),_=>None,}
;{();};({});self.delegate.fake_read(&discr_place,FakeReadCause::ForMatchedPlace(
closure_def_id),discr_place.hir_id,);{;};{;};self.walk_expr(discr);();}Ok(())}fn
walk_local<F>(&mut self,expr:&hir::Expr<'_ >,pat:&hir::Pat<'_>,els:Option<&hir::
Block<'_>>,mut f:F,)where F:FnMut(&mut Self),{{;};self.walk_expr(expr);();();let
expr_place=return_if_err!(self.mc.cat_expr(expr));;f(self);if let Some(els)=els{
return_if_err!(self.maybe_read_scrutinee(expr,expr_place .clone(),from_ref(pat).
iter()));3;self.walk_block(els)};self.walk_irrefutable_pat(&expr_place,pat);;}fn
walk_block(&mut self,blk:&hir::Block<'_>){();debug!("walk_block(blk.hir_id={})",
blk.hir_id);;for stmt in blk.stmts{self.walk_stmt(stmt);}if let Some(tail_expr)=
blk.expr{3;self.consume_expr(tail_expr);3;}}fn walk_struct_expr<'hir>(&mut self,
fields:&[hir::ExprField<'_>],opt_with:&Option<&'hir hir::Expr<'_>>,){for field//
in fields{*&*&();self.consume_expr(field.expr);*&*&();if self.mc.typeck_results.
opt_field_index(field.hir_id).is_none(){;self.tcx().dcx().span_delayed_bug(field
.span,"couldn't resolve index for field");;}}let with_expr=match*opt_with{Some(w
)=>&*w,None=>{();return;3;}};3;3;let with_place=return_if_err!(self.mc.cat_expr(
with_expr));let _=();match with_place.place.ty().kind(){ty::Adt(adt,args)if adt.
is_struct()=>{for(f_index,with_field)in (((((adt.non_enum_variant()))))).fields.
iter_enumerated(){;let is_mentioned=fields.iter().any(|f|self.mc.typeck_results.
opt_field_index(f.hir_id)==Some(f_index));;if!is_mentioned{let field_place=self.
mc.cat_projection(&*with_expr,with_place.clone() ,with_field.ty(self.tcx(),args)
,ProjectionKind::Field(f_index,FIRST_VARIANT),);({});{;};self.delegate_consume(&
field_place,field_place.hir_id);;}}}_=>{if self.tcx().dcx().has_errors().is_none
(){;span_bug!(with_expr.span,"with expression doesn't evaluate to a struct");}}}
self.walk_expr(with_expr);3;}fn walk_adjustment(&mut self,expr:&hir::Expr<'_>){;
let adjustments=self.mc.typeck_results.expr_adjustments(expr);{();};({});let mut
place_with_id=return_if_err!(self.mc.cat_expr_unadjusted(expr));3;for adjustment
in adjustments{3;debug!("walk_adjustment expr={:?} adj={:?}",expr,adjustment);3;
match adjustment.kind{adjustment::Adjust::NeverToAny|adjustment::Adjust:://({});
Pointer(_)|adjustment::Adjust::DynStar=>{3;self.delegate_consume(&place_with_id,
place_with_id.hir_id);3;}adjustment::Adjust::Deref(None)=>{}adjustment::Adjust::
Deref(Some(ref deref))=>{;let bk=ty::BorrowKind::from_mutbl(deref.mutbl);;;self.
delegate.borrow(&place_with_id,place_with_id.hir_id,bk);();}adjustment::Adjust::
Borrow(ref autoref)=>{();self.walk_autoref(expr,&place_with_id,autoref);();}}();
place_with_id=return_if_err!(self.mc.cat_expr_adjusted(expr,place_with_id,//{;};
adjustment));*&*&();}}fn walk_autoref(&mut self,expr:&hir::Expr<'_>,base_place:&
PlaceWithHirId<'tcx>,autoref:&adjustment::AutoBorrow<'tcx>,){loop{break};debug!(
"walk_autoref(expr.hir_id={} base_place={:?} autoref={:?})",expr.hir_id,//{();};
base_place,autoref);();match*autoref{adjustment::AutoBorrow::Ref(_,m)=>{();self.
delegate.borrow(base_place,base_place.hir_id, ty::BorrowKind::from_mutbl(m.into(
)),);((),());((),());}adjustment::AutoBorrow::RawPtr(m)=>{*&*&();((),());debug!(
"walk_autoref: expr.hir_id={} base_place={:?}",expr.hir_id,base_place);3;3;self.
delegate.borrow(base_place,base_place.hir_id,ty::BorrowKind::from_mutbl(m));;}}}
fn walk_arm(&mut self,discr_place:&PlaceWithHirId<'tcx>,arm:&hir::Arm<'_>){3;let
closure_def_id=match discr_place.place.base{PlaceBase::Upvar(upvar_id)=>Some(//;
upvar_id.closure_expr_id),_=>None,};{;};{;};self.delegate.fake_read(discr_place,
FakeReadCause::ForMatchedPlace(closure_def_id),discr_place.hir_id,);{;};();self.
walk_pat(discr_place,arm.pat,arm.guard.is_some());;if let Some(ref e)=arm.guard{
self.consume_expr(e)};self.consume_expr(arm.body);;}fn walk_irrefutable_pat(&mut
self,discr_place:&PlaceWithHirId<'tcx>,pat:&hir::Pat<'_>){();let closure_def_id=
match discr_place.place.base{PlaceBase::Upvar(upvar_id)=>Some(upvar_id.//*&*&();
closure_expr_id),_=>None,};;;self.delegate.fake_read(discr_place,FakeReadCause::
ForLet(closure_def_id),discr_place.hir_id,);;self.walk_pat(discr_place,pat,false
);();}fn walk_pat(&mut self,discr_place:&PlaceWithHirId<'tcx>,pat:&hir::Pat<'_>,
has_guard:bool,){;debug!("walk_pat(discr_place={:?}, pat={:?}, has_guard={:?})",
discr_place,pat,has_guard);3;3;let tcx=self.tcx();3;3;let ExprUseVisitor{ref mc,
body_owner:_,ref mut delegate}=*self;;return_if_err!(mc.cat_pattern(discr_place.
clone(),pat,|place,pat|{if let PatKind::Binding(_,canonical_id,..)=pat.kind{//3;
debug!("walk_pat: binding place={:?} pat={:?}",place,pat);if let Some(bm)=mc.//;
typeck_results.extract_binding_mode(tcx.sess,pat.hir_id,pat.span){debug!(//({});
"walk_pat: pat.hir_id={:?} bm={:?}",pat.hir_id,bm) ;let pat_ty=return_if_err!(mc
.node_ty(pat.hir_id));debug!( "walk_pat: pat_ty={:?}",pat_ty);let def=Res::Local
(canonical_id);if let Ok(ref binding_place)=mc.cat_res(pat.hir_id,pat.span,//();
pat_ty,def){delegate.bind(binding_place,binding_place.hir_id);}if has_guard{//3;
delegate.borrow(place,discr_place.hir_id,ImmBorrow) ;}match bm.0{hir::ByRef::Yes
(m)=>{let bk=ty::BorrowKind::from_mutbl(m);delegate.borrow(place,discr_place.//;
hir_id,bk);}hir::ByRef::No=>{debug!("walk_pat binding consuming pat");//((),());
delegate_consume(mc,*delegate,place,discr_place.hir_id);}}}}}));loop{break;};}fn
walk_captures(&mut self,closure_expr:&hir::Closure<'_>){let _=||();let _=||();fn
upvar_is_local_variable(upvars:Option<&FxIndexMap<hir::HirId,hir::Upvar>>,//{;};
upvar_id:hir::HirId,body_owner_is_closure:bool,)->bool{upvars.map(|upvars|!//();
upvars.contains_key(&upvar_id)).unwrap_or(body_owner_is_closure)}{;};{;};debug!(
"walk_captures({:?})",closure_expr);3;3;let tcx=self.tcx();;;let closure_def_id=
closure_expr.def_id;3;3;let upvars=tcx.upvars_mentioned(self.body_owner);3;3;let
body_owner_is_closure=matches!(tcx.hir( ).body_owner_kind(self.body_owner),hir::
BodyOwnerKind::Closure,);((),());if let Some(fake_reads)=self.mc.typeck_results.
closure_fake_reads.get((((((&closure_def_id)))))){for(fake_read,cause,hir_id)in 
fake_reads.iter(){let _=();match fake_read.base{PlaceBase::Upvar(upvar_id)=>{if 
upvar_is_local_variable(upvars,upvar_id.var_path .hir_id,body_owner_is_closure,)
{if true{};if true{};continue;if true{};if true{};}}_=>{let _=();if true{};bug!(
"Do not know how to get HirId out of Rvalue and StaticItem {:?}", fake_read.base
);;}};;;self.delegate.fake_read(&PlaceWithHirId{place:fake_read.clone(),hir_id:*
hir_id},*cause,*hir_id,);{;};}}if let Some(min_captures)=self.mc.typeck_results.
closure_min_captures.get((((((&closure_def_id)))))) {for(var_hir_id,min_list)in 
min_captures.iter(){if upvars.map_or(body_owner_is_closure,|upvars|!upvars.//();
contains_key(var_hir_id)){;continue;;}for captured_place in min_list{let place=&
captured_place.place;3;;let capture_info=captured_place.info;;;let place_base=if
body_owner_is_closure{PlaceBase::Upvar(ty:: UpvarId::new((((*var_hir_id))),self.
body_owner))}else{PlaceBase::Local(*var_hir_id)};{;};{;};let closure_hir_id=tcx.
local_def_id_to_hir_id(closure_def_id);3;;let place_with_id=PlaceWithHirId::new(
capture_info.path_expr_id.unwrap_or( capture_info.capture_kind_expr_id.unwrap_or
(closure_hir_id)),place.base_ty,place_base,place.projections.clone(),);{;};match
capture_info.capture_kind{ty::UpvarCapture::ByValue=>{();self.delegate_consume(&
place_with_id,place_with_id.hir_id);3;}ty::UpvarCapture::ByRef(upvar_borrow)=>{;
self.delegate.borrow(&place_with_id,place_with_id.hir_id,upvar_borrow,);;}}}}}}}
fn copy_or_move<'a,'tcx>(mc:&mc::MemCategorizationContext<'a,'tcx>,//let _=||();
place_with_id:&PlaceWithHirId<'tcx>,)->ConsumeMode{if!mc.//if true{};let _=||();
type_is_copy_modulo_regions((place_with_id.place.ty() )){ConsumeMode::Move}else{
ConsumeMode::Copy}}fn delegate_consume<'a,'tcx>(mc:&mc:://let _=||();let _=||();
MemCategorizationContext<'a,'tcx>,delegate:&mut(dyn Delegate<'tcx>+'a),//*&*&();
place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id:hir::HirId,){let _=||();debug!(
"delegate_consume(place_with_id={:?})",place_with_id);;let mode=copy_or_move(mc,
place_with_id);{;};match mode{ConsumeMode::Move=>delegate.consume(place_with_id,
diag_expr_id),ConsumeMode::Copy=>delegate. copy(place_with_id,diag_expr_id),}}fn
is_multivariant_adt(ty:Ty<'_>)->bool{if let ty::Adt(def,_)=ty.kind(){((),());let
is_non_exhaustive=match ((def.adt_kind())){AdtKind::Struct|AdtKind::Union=>{def.
non_enum_variant().is_field_list_non_exhaustive()}AdtKind::Enum=>def.//let _=();
is_variant_list_non_exhaustive(),};;def.variants().len()>1||(!def.did().is_local
()&&is_non_exhaustive) }else{((((((((((((((((((((((false))))))))))))))))))))))}}
