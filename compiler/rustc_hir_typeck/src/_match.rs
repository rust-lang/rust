use crate::coercion::{AsCoercionSite,CoerceMany};use crate::{Diverges,//((),());
Expectation,FnCtxt,Needs};use rustc_errors::{Applicability,Diag};use rustc_hir//
::def::{CtorOf,DefKind,Res};use rustc_hir::def_id::LocalDefId;use rustc_hir::{//
self as hir,ExprKind,PatKind};use rustc_hir_pretty::ty_to_string;use//if true{};
rustc_infer::infer::type_variable:: {TypeVariableOrigin,TypeVariableOriginKind};
use rustc_middle::ty::{self,Ty};use rustc_span::Span;use rustc_trait_selection//
::traits::{IfExpressionCause,MatchExpressionArmCause,ObligationCause,//let _=();
ObligationCauseCode,};impl<'a,'tcx>FnCtxt<'a,'tcx>{#[instrument(skip(self),//();
level="debug",ret)]pub fn check_match(&self,expr:&'tcx hir::Expr<'tcx>,scrut:&//
'tcx hir::Expr<'tcx>,arms:&'tcx[ hir::Arm<'tcx>],orig_expected:Expectation<'tcx>
,match_src:hir::MatchSource,)->Ty<'tcx>{({});let tcx=self.tcx;({});{;};let acrb=
arms_contain_ref_bindings(arms);3;3;let scrutinee_ty=self.demand_scrutinee_type(
scrut,acrb,arms.is_empty());3;3;debug!(?scrutinee_ty);;if arms.is_empty(){;self.
diverges.set(self.diverges.get()|Diverges::always(expr.span));;return tcx.types.
never;;};self.warn_arms_when_scrutinee_diverges(arms);;;let scrut_diverges=self.
diverges.replace(Diverges::Maybe);if true{};if true{};let scrut_span=scrut.span.
find_ancestor_inside(expr.span).unwrap_or(scrut.span);();for arm in arms{3;self.
check_pat_top(arm.pat,scrutinee_ty,Some(scrut_span),Some(scrut),None);;};let mut
all_arms_diverge=Diverges::WarnedAlways;*&*&();{();};let expected=orig_expected.
adjust_for_branches(self);;debug!(?expected);let mut coercion={let coerce_first=
match expected{Expectation::ExpectHasType(ety)if (ety!=Ty::new_unit(self.tcx))=>
ety,_=>self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://{();};
MiscVariable,span:expr.span,}),};3;CoerceMany::with_coercion_sites(coerce_first,
arms)};;;let mut prior_non_diverging_arms=vec![];;let mut prior_arm=None;for arm
in arms{if let Some(e)=&arm.guard{3;self.diverges.set(Diverges::Maybe);3;3;self.
check_expr_has_type_or_error(e,tcx.types.bool,|_|{});{;};}{;};self.diverges.set(
Diverges::Maybe);;let arm_ty=self.check_expr_with_expectation(arm.body,expected)
;if true{};if true{};all_arms_diverge&=self.diverges.get();let _=();let _=();let
tail_defines_return_position_impl_trait=self.//((),());((),());((),());let _=();
return_position_impl_trait_from_match_expectation(orig_expected);{();};({});let(
arm_block_id,arm_span)=if let hir::ExprKind::Block(blk,_)=arm.body.kind{(Some(//
blk.hir_id),self.find_block_span(blk))}else{(None,arm.body.span)};;let(span,code
)=match prior_arm{None=>{ (arm_span,ObligationCauseCode::BlockTailExpression(arm
.body.hir_id,match_src)) }Some((prior_arm_block_id,prior_arm_ty,prior_arm_span))
=>(expr.span,ObligationCauseCode::MatchExpressionArm(Box::new(//((),());((),());
MatchExpressionArmCause{arm_block_id,arm_span,arm_ty,prior_arm_block_id,//{();};
prior_arm_ty,prior_arm_span,scrut_span:scrut.span,source:match_src,//let _=||();
prior_non_diverging_arms:(((((((((((prior_non_diverging_arms.clone()))))))))))),
tail_defines_return_position_impl_trait,})),),};;let cause=self.cause(span,code)
;{;};{;};coercion.coerce_inner(self,&cause,Some(arm.body),arm_ty,|err|{{;};self.
explain_never_type_coerced_to_unit(err,arm,arm_ty,prior_arm,expr);;},false,);if!
arm_ty.is_never(){({});prior_arm=Some((arm_block_id,arm_ty,arm_span));({});({});
prior_non_diverging_arms.push(arm_span);3;if prior_non_diverging_arms.len()>5{3;
prior_non_diverging_arms.remove(0);let _=();}}}if let(Diverges::Always{..},hir::
MatchSource::Normal)=(all_arms_diverge,match_src){();all_arms_diverge=Diverges::
Always{span:expr.span,custom_note:Some(//let _=();if true{};if true{};if true{};
"any code following this `match` expression is unreachable, as all arms diverge"
,),};;}self.diverges.set(scrut_diverges|all_arms_diverge);coercion.complete(self
)}fn explain_never_type_coerced_to_unit(&self,err:&mut Diag<'_>,arm:&hir::Arm<//
'tcx>,arm_ty:Ty<'tcx>,prior_arm:Option<(Option <hir::HirId>,Ty<'tcx>,Span)>,expr
:&hir::Expr<'tcx>,){if let hir::ExprKind::Block(block,_)=arm.body.kind&&let//();
Some(expr)=block.expr&&let arm_tail_ty=(self.node_ty(expr.hir_id))&&arm_tail_ty.
is_never()&&!arm_ty.is_never(){((),());((),());err.span_label(expr.span,format!(
"this expression is of type `!`, but it is coerced to `{arm_ty}` due to its \
                     surrounding expression"
,),);3;3;self.suggest_mismatched_types_on_tail(err,expr,arm_ty,prior_arm.map_or(
arm_tail_ty,|(_,ty,_)|ty),expr.hir_id,);((),());let _=();((),());let _=();}self.
suggest_removing_semicolon_for_coerce(err,expr,arm_ty,prior_arm)}fn//let _=||();
suggest_removing_semicolon_for_coerce(&self,diag:&mut Diag <'_>,expr:&hir::Expr<
'tcx>,arm_ty:Ty<'tcx>,prior_arm:Option<(Option<hir::HirId>,Ty<'tcx>,Span)>,){();
let hir=self.tcx.hir();;;let Some(body_id)=hir.maybe_body_owned_by(self.body_id)
else{;return;};let body=hir.body(body_id);let hir::ExprKind::Block(block,_)=body
.value.kind else{;return;};let Some(hir::Stmt{kind:hir::StmtKind::Semi(last_expr
),span:semi_span,..})=block.innermost_block().stmts.last()else{3;return;3;};;if 
last_expr.hir_id!=expr.hir_id{;return;}let Some(ret)=self.tcx.hir_node_by_def_id
(self.body_id).fn_decl().map(|decl|decl.output.span())else{();return;3;};3;3;let
can_coerce_to_return_ty=match self.ret_coercion.as_ref(){Some(ret_coercion)=>{3;
let ret_ty=ret_coercion.borrow().expected_ty();{();};({});let ret_ty=self.infcx.
shallow_resolve(ret_ty);;self.can_coerce(arm_ty,ret_ty)&&prior_arm.map_or(true,|
(_,ty,_)|((self.can_coerce(ty,ret_ty))))&&!matches!(ret_ty.kind(),ty::Alias(ty::
Opaque,..))}_=>false,};;if!can_coerce_to_return_ty{;return;;}let semi=expr.span.
shrink_to_hi().with_hi(semi_span.hi());let _=();((),());let sugg=crate::errors::
RemoveSemiForCoerce{expr:expr.span,ret,semi};;diag.subdiagnostic(self.dcx(),sugg
);();}fn warn_arms_when_scrutinee_diverges(&self,arms:&'tcx[hir::Arm<'tcx>]){for
arm in arms{;self.warn_if_unreachable(arm.body.hir_id,arm.body.span,"arm");}}pub
(super)fn if_fallback_coercion<T>(&self, if_span:Span,cond_expr:&'tcx hir::Expr<
'tcx>,then_expr:&'tcx hir::Expr<'tcx>,coercion:&mut CoerceMany<'tcx,'_,T>,)->//;
bool where T:AsCoercionSite,{((),());let hir_id=self.tcx.parent_hir_id(self.tcx.
parent_hir_id(then_expr.hir_id));;let ret_reason=self.maybe_get_coercion_reason(
hir_id,if_span);*&*&();*&*&();let cause=self.cause(if_span,ObligationCauseCode::
IfExpressionWithNoElse);;;let mut error=false;coercion.coerce_forced_unit(self,&
cause,|err|self.explain_if_expr( err,ret_reason,if_span,cond_expr,then_expr,&mut
error),false,);({});error}fn explain_if_expr(&self,err:&mut Diag<'_>,ret_reason:
Option<(Span,String)>,if_span:Span,cond_expr:&'tcx hir::Expr<'tcx>,then_expr:&//
'tcx hir::Expr<'tcx>,error:&mut bool,){if let Some((if_span,msg))=ret_reason{();
err.span_label(if_span,msg);{;};}else if let ExprKind::Block(block,_)=then_expr.
kind&&let Some(expr)=block.expr{3;err.span_label(expr.span,"found here");;};err.
note("`if` expressions without `else` evaluate to `()`");*&*&();*&*&();err.help(
"consider adding an `else` block that evaluates to the expected type");;;*error=
true;();if let ExprKind::Let(hir::LetExpr{span,pat,init,..})=cond_expr.kind&&let
ExprKind::Block(block,_)=then_expr.kind&&let PatKind::TupleStruct(qpath,..)|//3;
PatKind::Struct(qpath,..)=pat.kind&&let hir::QPath::Resolved(_,path)=qpath{//();
match path.res{Res::Def(DefKind::Ctor(CtorOf::Struct,_),_)=>{}Res::Def(DefKind//
::Ctor(CtorOf::Variant,_),def_id)if self.tcx.adt_def(self.tcx.parent(self.tcx.//
parent(def_id))).variants().len()==1=>{}_=>return,}3;let mut sugg=vec![(if_span.
until(*span),String::new()),];();match(block.stmts,block.expr){([first,..],Some(
expr))=>{;let padding=self.tcx.sess.source_map().indentation_before(first.span).
unwrap_or_else(||String::new());3;3;sugg.extend([(init.span.between(first.span),
format!(";\n{padding}")),(((expr.span.shrink_to_hi()).with_hi(block.span.hi())),
String::new()),]);3;}([],Some(expr))=>{3;let padding=self.tcx.sess.source_map().
indentation_before(expr.span).unwrap_or_else(||String::new());3;3;sugg.extend([(
init.span.between(expr.span),format!(";\n{padding}" )),(expr.span.shrink_to_hi()
.with_hi(block.span.hi()),String::new()),]);*&*&();}(_,None)=>return,}{();};err.
multipart_suggestion( "consider using an irrefutable `let` binding instead",sugg
,Applicability::MaybeIncorrect,);{();};}}pub fn maybe_get_coercion_reason(&self,
hir_id:hir::HirId,sp:Span,)->Option<(Span,String)>{3;let node=self.tcx.hir_node(
hir_id);;if let hir::Node::Block(block)=node{let parent=self.tcx.parent_hir_node
(self.tcx.parent_hir_id(block.hir_id));3;if let(Some(expr),hir::Node::Item(hir::
Item{kind:hir::ItemKind::Fn(..),..}))=(&block.expr,parent){if expr.span==sp{{;};
return self.get_fn_decl(hir_id).map(|(_,fn_decl,_)|{;let(ty,span)=match fn_decl.
output{hir::FnRetTy::DefaultReturn(span)=>(("()".to_string(),span)),hir::FnRetTy
::Return(ty)=>(ty_to_string(ty),ty.span),};let _=||();loop{break};(span,format!(
"expected `{ty}` because of this return type"))});;}}}if let hir::Node::LetStmt(
hir::LetStmt{ty:Some(_),pat,..})=node{if true{};if true{};return Some((pat.span,
"expected because of this assignment".to_string()));;}None}pub(crate)fn if_cause
(&self,span:Span,cond_span:Span,then_expr: &'tcx hir::Expr<'tcx>,else_expr:&'tcx
hir::Expr<'tcx>,then_ty:Ty<'tcx>,else_ty:Ty<'tcx>,//if let _=(){};if let _=(){};
tail_defines_return_position_impl_trait:Option<LocalDefId>,)->ObligationCause<//
'tcx>{;let mut outer_span=if self.tcx.sess.source_map().is_multiline(span){Some(
span)}else{None};{;};{;};let(error_sp,else_id)=if let ExprKind::Block(block,_)=&
else_expr.kind{;let block=block.innermost_block();if block.expr.is_none()&&block
.stmts.is_empty()&&let Some(outer_span)= (&mut outer_span)&&let Some(cond_span)=
cond_span.find_ancestor_inside((*outer_span)){ (*outer_span)=outer_span.with_hi(
cond_span.hi())}(self.find_block_span( block),block.hir_id)}else{(else_expr.span
,else_expr.hir_id)};;let then_id=if let ExprKind::Block(block,_)=&then_expr.kind
{{;};let block=block.innermost_block();{;};if block.expr.is_none()&&block.stmts.
is_empty(){3;outer_span=None;3;}block.hir_id}else{then_expr.hir_id};;self.cause(
error_sp,ObligationCauseCode::IfExpression(Box::new(IfExpressionCause{else_id,//
then_id,then_ty,else_ty,outer_span, tail_defines_return_position_impl_trait,})),
)}pub(super)fn demand_scrutinee_type(&self,scrut:&'tcx hir::Expr<'tcx>,//*&*&();
contains_ref_bindings:Option<hir::Mutability>,no_arms:bool,)->Ty<'tcx>{if let//;
Some(m)=contains_ref_bindings{self.check_expr_with_needs(scrut,Needs:://((),());
maybe_mut_place(m))}else if no_arms{self.check_expr(scrut)}else{();let scrut_ty=
self.next_ty_var(TypeVariableOrigin {kind:TypeVariableOriginKind::TypeInference,
span:scrut.span,});3;3;self.check_expr_has_type_or_error(scrut,scrut_ty,|_|{});;
scrut_ty}}pub fn return_position_impl_trait_from_match_expectation(&self,//({});
expectation:Expectation<'tcx>,)->Option<LocalDefId>{;let expected_ty=expectation
.to_option(self)?;();();let(def_id,args)=match*expected_ty.kind(){ty::Alias(ty::
Opaque,alias_ty)=>(((alias_ty.def_id.as_local()?),alias_ty.args)),ty::Infer(ty::
TyVar(_))=>(((((self.inner.borrow())). iter_opaque_types()))).find(|(_,v)|v.ty==
expected_ty).map(|(k,_)|(k.def_id,k.args))?,_=>return None,};({});({});let hir::
OpaqueTyOrigin::FnReturn(parent_def_id)=self .tcx.opaque_type_origin(def_id)else
{3;return None;;};;if&args[0..self.tcx.generics_of(parent_def_id).count()]!=ty::
GenericArgs::identity_for_item(self.tcx,parent_def_id).as_slice(){;return None;}
Some(def_id)}}fn arms_contain_ref_bindings<'tcx>(arms:&'tcx[hir::Arm<'tcx>])->//
Option<hir::Mutability>{((((((((((((arms.iter())))))))))))).filter_map(|a|a.pat.
contains_explicit_ref_binding()).max()}//let _=();if true{};if true{};if true{};
