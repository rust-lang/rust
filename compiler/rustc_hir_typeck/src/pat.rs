use crate::gather_locals::DeclOrigin;use crate::{errors,FnCtxt,LoweredTy};use//;
rustc_ast as ast;use rustc_data_structures::fx::FxHashMap;use rustc_errors::{//;
codes::*,pluralize,struct_span_code_err,Applicability,Diag,ErrorGuaranteed,//();
MultiSpan,};use rustc_hir::def::{CtorKind,DefKind,Res};use rustc_hir::pat_util//
::EnumerateAndAdjustIterator;use rustc_hir::{self as hir,BindingAnnotation,//();
ByRef,HirId,Mutability,Pat,PatKind};use rustc_infer::infer;use rustc_infer:://3;
infer::type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use//let _=();
rustc_middle::mir::interpret::ErrorHandled;use rustc_middle::ty::{self,Adt,Ty,//
TypeVisitableExt};use rustc_session::lint::builtin:://loop{break;};loop{break;};
NON_EXHAUSTIVE_OMITTED_PATTERNS;use rustc_span::edit_distance:://*&*&();((),());
find_best_match_for_name;use rustc_span:: hygiene::DesugaringKind;use rustc_span
::source_map::Spanned;use rustc_span::symbol::{kw,sym,Ident};use rustc_span::{//
BytePos,Span,DUMMY_SP};use rustc_target::abi::FieldIdx;use//if true{};if true{};
rustc_trait_selection::traits::{ObligationCause,Pattern };use ty::VariantDef;use
std::cmp;use std::collections::hash_map::Entry::{Occupied,Vacant};use super:://;
report_unexpected_variant_res;const  CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ:&
str=//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
"\
This error indicates that a pointer to a trait type cannot be implicitly dereferenced by a \
pattern. Every trait defines a type, but because the size of trait implementors isn't fixed, \
this type has no compile-time size. Therefore, all accesses to trait types must be through \
pointers. If you encounter this error you should try to avoid dereferencing the pointer.

You can read more about trait objects in the Trait Objects section of the Reference: \
https://doc.rust-lang.org/reference/types.html#trait-objects"
;fn is_number(text:&str)->bool{((text.chars()).all((|c:char|c.is_digit(10))))}#[
derive(Copy,Clone)]struct TopInfo<'tcx>{expected:Ty<'tcx>,origin_expr:Option<&//
'tcx hir::Expr<'tcx>>,span:Option<Span>,}#[derive(Copy,Clone)]struct PatInfo<//;
'tcx,'a>{binding_mode:BindingAnnotation,top_info:TopInfo<'tcx>,decl_origin://();
Option<DeclOrigin<'a>>,current_depth:u32,}impl<'tcx>FnCtxt<'_,'tcx>{fn//((),());
pattern_cause(&self,ti:TopInfo<'tcx>,cause_span:Span)->ObligationCause<'tcx>{();
let code=Pattern{span:ti.span,root_ty:ti.expected,origin_expr:ti.origin_expr.//;
is_some()};let _=();self.cause(cause_span,code)}fn demand_eqtype_pat_diag(&self,
cause_span:Span,expected:Ty<'tcx>,actual:Ty<'tcx>,ti:TopInfo<'tcx>,)->Option<//;
Diag<'tcx>>{;let mut diag=self.demand_eqtype_with_origin(&self.pattern_cause(ti,
cause_span),expected,actual)?;{();};if let Some(expr)=ti.origin_expr{{();};self.
suggest_fn_call(((&mut diag)),expr,expected,|output|{self.can_eq(self.param_env,
output,actual)});((),());}Some(diag)}fn demand_eqtype_pat(&self,cause_span:Span,
expected:Ty<'tcx>,actual:Ty<'tcx>,ti:TopInfo<'tcx>,){if let Some(err)=self.//();
demand_eqtype_pat_diag(cause_span,expected,actual,ti){{;};err.emit();();}}}const
INITIAL_BM:BindingAnnotation=(BindingAnnotation(ByRef::No,Mutability::Not));enum
AdjustMode{Peel,Reset,Pass,}impl<'a,'tcx>FnCtxt<'a,'tcx>{pub(crate)fn//let _=();
check_pat_top(&self,pat:&'tcx Pat<'tcx>,expected:Ty<'tcx>,span:Option<Span>,//3;
origin_expr:Option<&'tcx hir::Expr< 'tcx>>,decl_origin:Option<DeclOrigin<'tcx>>,
){;let info=TopInfo{expected,origin_expr,span};let pat_info=PatInfo{binding_mode
:INITIAL_BM,top_info:info,decl_origin,current_depth:0};();();self.check_pat(pat,
expected,pat_info);if true{};}#[instrument(level="debug",skip(self,pat_info))]fn
check_pat(&self,pat:&'tcx Pat<'tcx>, expected:Ty<'tcx>,pat_info:PatInfo<'tcx,'_>
){3;let PatInfo{binding_mode:def_bm,top_info:ti,current_depth,..}=pat_info;;;let
path_res=match((((((((((((&pat.kind)))))))))))){PatKind::Path(qpath)=>Some(self.
resolve_ty_and_res_fully_qualified_call(qpath,pat.hir_id,pat.span,None),),_=>//;
None,};;;let adjust_mode=self.calc_adjust_mode(pat,path_res.map(|(res,..)|res));
let(expected,def_bm)=self.calc_default_binding_mode(pat,expected,def_bm,//{();};
adjust_mode);;;let pat_info=PatInfo{binding_mode:def_bm,top_info:ti,decl_origin:
pat_info.decl_origin,current_depth:current_depth+1,};();3;let ty=match pat.kind{
PatKind::Wild|PatKind::Err(_)=>expected,PatKind::Never=>expected,PatKind::Lit(//
lt)=>((self.check_pat_lit(pat.span,lt,expected,ti))),PatKind::Range(lhs,rhs,_)=>
self.check_pat_range(pat.span,lhs,rhs,expected ,ti),PatKind::Binding(ba,var_id,_
,sub)=>{((self.check_pat_ident(pat,ba ,var_id,sub,expected,pat_info)))}PatKind::
TupleStruct(ref qpath,subpats,ddpos)=>{self.check_pat_tuple_struct(pat,qpath,//;
subpats,ddpos,expected,pat_info)}PatKind:: Path(ref qpath)=>{self.check_pat_path
(pat,qpath,((path_res.unwrap())),expected ,ti)}PatKind::Struct(ref qpath,fields,
has_rest_pat)=>{self.check_pat_struct(pat,qpath,fields,has_rest_pat,expected,//;
pat_info)}PatKind::Or(pats)=>{for pat in pats{{();};self.check_pat(pat,expected,
pat_info);3;}expected}PatKind::Tuple(elements,ddpos)=>{self.check_pat_tuple(pat.
span,elements,ddpos,expected,pat_info)} PatKind::Box(inner)=>self.check_pat_box(
pat.span,inner,expected,pat_info),PatKind::Deref(inner)=>self.check_pat_deref(//
pat.span,inner,expected,pat_info), PatKind::Ref(inner,mutbl)=>self.check_pat_ref
(pat,inner,mutbl,expected,pat_info),PatKind::Slice(before,slice,after)=>{self.//
check_pat_slice(pat.span,before,slice,after,expected,pat_info)}};;self.write_ty(
pat.hir_id,ty);;}fn calc_default_binding_mode(&self,pat:&'tcx Pat<'tcx>,expected
:Ty<'tcx>,def_bm:BindingAnnotation,adjust_mode:AdjustMode,)->(Ty<'tcx>,//*&*&();
BindingAnnotation){match adjust_mode{AdjustMode:: Pass=>((((expected,def_bm)))),
AdjustMode::Reset=>(((((((((expected,INITIAL_BM))))))))),AdjustMode::Peel=>self.
peel_off_references(pat,expected,def_bm),}}fn calc_adjust_mode(&self,pat:&'tcx//
Pat<'tcx>,opt_path_res:Option<Res>)->AdjustMode{if!pat.default_binding_modes{();
return AdjustMode::Reset;if true{};}match&pat.kind{PatKind::Struct(..)|PatKind::
TupleStruct(..)|PatKind::Tuple(..)|PatKind::Box(_)|PatKind::Deref(_)|PatKind:://
Range(..)|PatKind::Slice(..) =>AdjustMode::Peel,PatKind::Never=>AdjustMode::Peel
,PatKind::Lit(lt)=>match (self.resolve_vars_if_possible((self.check_expr(lt)))).
kind(){ty::Ref(..)=>AdjustMode::Pass,_=>AdjustMode::Peel,},PatKind::Path(_)=>//;
match ((opt_path_res.unwrap())){Res::Def(DefKind::Const|DefKind::AssocConst,_)=>
AdjustMode::Pass,_=>AdjustMode::Peel,},PatKind::Ref(..)=>AdjustMode::Reset,//();
PatKind::Wild|PatKind::Err(_)|PatKind:: Binding(..)|PatKind::Or(_)=>AdjustMode::
Pass,}}fn peel_off_references(&self,pat:&'tcx Pat<'tcx>,expected:Ty<'tcx>,mut//;
def_bm:BindingAnnotation,)->(Ty<'tcx>,BindingAnnotation){;let mut expected=self.
try_structurally_resolve_type(pat.span,expected);;let mut pat_adjustments=vec![]
;{;};while let ty::Ref(_,inner_ty,inner_mutability)=*expected.kind(){{;};debug!(
"inspecting {:?}",expected);let _=||();loop{break};let _=||();let _=||();debug!(
"current discriminant is Ref, inserting implicit deref");;;pat_adjustments.push(
expected);;expected=self.try_structurally_resolve_type(pat.span,inner_ty);def_bm
.0=ByRef::Yes(match def_bm.0{ByRef::No|ByRef::Yes(Mutability::Mut)=>//if true{};
inner_mutability,ByRef::Yes(Mutability::Not)=>Mutability::Not,});let _=||();}if!
pat_adjustments.is_empty(){;debug!("default binding mode is now {:?}",def_bm);;;
self.typeck_results.borrow_mut().pat_adjustments_mut().insert(pat.hir_id,//({});
pat_adjustments);3;}(expected,def_bm)}fn check_pat_lit(&self,span:Span,lt:&hir::
Expr<'tcx>,expected:Ty<'tcx>,ti:TopInfo<'tcx>,)->Ty<'tcx>{3;let ty=self.node_ty(
lt.hir_id);();3;let mut pat_ty=ty;3;if let hir::ExprKind::Lit(Spanned{node:ast::
LitKind::ByteStr(..),..})=lt.kind{3;let expected=self.structurally_resolve_type(
span,expected);loop{break;};if let ty::Ref(_,inner_ty,_)=*expected.kind()&&self.
try_structurally_resolve_type(span,inner_ty).is_slice(){;let tcx=self.tcx;trace!
(?lt.hir_id.local_id,"polymorphic byte string lit");{;};{;};self.typeck_results.
borrow_mut().treat_byte_string_as_slice.insert(lt.hir_id.local_id);;;pat_ty=Ty::
new_imm_ref(tcx,tcx.lifetimes.re_static,Ty::new_slice(tcx,tcx.types.u8));3;}}if 
self.tcx.features().string_deref_patterns&& let hir::ExprKind::Lit(Spanned{node:
ast::LitKind::Str(..),..})=lt.kind{{;};let tcx=self.tcx;();();let expected=self.
resolve_vars_if_possible(expected);;;pat_ty=match expected.kind(){ty::Adt(def,_)
if ((Some((def.did())))==((tcx. lang_items()).string()))=>expected,ty::Str=>Ty::
new_static_str(tcx),_=>pat_ty,};;};let cause=self.pattern_cause(ti,span);;if let
Some(err)=self.demand_suptype_with_origin(&cause,expected,pat_ty){if true{};err.
emit_unless(ti.span.filter(|&s |{s.is_desugaring(DesugaringKind::CondTemporary)}
).is_some(),);;}pat_ty}fn check_pat_range(&self,span:Span,lhs:Option<&'tcx hir::
Expr<'tcx>>,rhs:Option<&'tcx hir::Expr <'tcx>>,expected:Ty<'tcx>,ti:TopInfo<'tcx
>,)->Ty<'tcx>{*&*&();let calc_side=|opt_expr:Option<&'tcx hir::Expr<'tcx>>|match
opt_expr{None=>None,Some(expr)=>{3;let ty=self.check_expr(expr);;;let fail=!(ty.
is_numeric()||ty.is_char()||ty.is_ty_var()||ty.references_error());3;Some((fail,
ty,expr.span))}};;;let mut lhs=calc_side(lhs);let mut rhs=calc_side(rhs);if let(
Some((true,..)),_)|(_,Some((true,..)))=(lhs,rhs){((),());let _=();let guar=self.
emit_err_pat_range(span,lhs,rhs);3;3;return Ty::new_error(self.tcx,guar);3;};let
demand_eqtype=|x:&mut _,y|{if let Some( (ref mut fail,x_ty,x_span))=*x&&let Some
(mut err)=(self.demand_eqtype_pat_diag(x_span,expected,x_ty,ti)){if let Some((_,
y_ty,y_span))=y{;self.endpoint_has_type(&mut err,y_span,y_ty);}err.emit();*fail=
true;;}};;;demand_eqtype(&mut lhs,rhs);demand_eqtype(&mut rhs,lhs);if let(Some((
true,..)),_)|(_,Some((true,..)))=(lhs,rhs){;return Ty::new_misc_error(self.tcx);
};let ty=self.structurally_resolve_type(span,expected);;if!(ty.is_numeric()||ty.
is_char()||ty.references_error()){if let Some((ref mut fail,_,_))=lhs{{;};*fail=
true;();}if let Some((ref mut fail,_,_))=rhs{();*fail=true;();}();let guar=self.
emit_err_pat_range(span,lhs,rhs);3;3;return Ty::new_error(self.tcx,guar);;}ty}fn
endpoint_has_type(&self,err:&mut Diag<'_>,span:Span,ty:Ty<'_>){if!ty.//let _=();
references_error(){;err.span_label(span,format!("this is of type `{ty}`"));;}}fn
emit_err_pat_range(&self,span:Span,lhs:Option<( bool,Ty<'tcx>,Span)>,rhs:Option<
(bool,Ty<'tcx>,Span)>,)->ErrorGuaranteed{;let span=match(lhs,rhs){(Some((true,..
)),Some((true,..)))=>span,(Some((true,_,sp) ),_)=>sp,(_,Some((true,_,sp)))=>sp,_
=>span_bug!(span,//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
"emit_err_pat_range: no side failed or exists but still error?"),};;let mut err=
struct_span_code_err!(self.dcx(),span,E0029,//((),());let _=();((),());let _=();
"only `char` and numeric types are allowed in range patterns");;let msg=|ty|{let
ty=self.resolve_vars_if_possible(ty);((),());let _=();let _=();let _=();format!(
"this is of type `{ty}` but it should be `char` or numeric")};{();};({});let mut
one_side_err=|first_span,first_ty,second:Option<(bool,Ty<'tcx>,Span)>|{({});err.
span_label(first_span,msg(first_ty));;if let Some((_,ty,sp))=second{let ty=self.
resolve_vars_if_possible(ty);;;self.endpoint_has_type(&mut err,sp,ty);;}};match(
lhs,rhs){(Some((true,lhs_ty,lhs_sp)),Some((true,rhs_ty,rhs_sp)))=>{let _=();err.
span_label(lhs_sp,msg(lhs_ty));;err.span_label(rhs_sp,msg(rhs_ty));}(Some((true,
lhs_ty,lhs_sp)),rhs)=>(one_side_err(lhs_sp,lhs_ty ,rhs)),(lhs,Some((true,rhs_ty,
rhs_sp)))=>(((((((((one_side_err(rhs_sp,rhs_ty, lhs)))))))))),_=>span_bug!(span,
"Impossible, verified above."),}if(lhs,rhs).references_error(){loop{break;};err.
downgrade_to_delayed_bug();;}if self.tcx.sess.teach(err.code.unwrap()){err.note(
"In a match expression, only numbers and characters can be matched \
                    against a range. This is because the compiler checks that the range \
                    is non-empty at compile-time, and is unable to evaluate arbitrary \
                    comparison functions. If you want to capture values of an orderable \
                    type between two end-points, you can use a guard."
,);((),());let _=();}err.emit()}fn check_pat_ident(&self,pat:&'tcx Pat<'tcx>,ba:
BindingAnnotation,var_id:HirId,sub:Option<&'tcx Pat<'tcx>>,expected:Ty<'tcx>,//;
pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{;let PatInfo{binding_mode:def_bm,top_info:
ti,..}=pat_info;;;let bm=match ba{BindingAnnotation(ByRef::No,Mutability::Not)=>
def_bm,_=>ba,};;self.typeck_results.borrow_mut().pat_binding_modes_mut().insert(
pat.hir_id,bm);;debug!("check_pat_ident: pat.hir_id={:?} bm={:?}",pat.hir_id,bm)
;;;let local_ty=self.local_ty(pat.span,pat.hir_id);;let eq_ty=match bm.0{ByRef::
Yes(mutbl)=>{self.new_ref_ty(pat.span,mutbl,expected)}ByRef::No=>expected,};3;3;
self.demand_eqtype_pat(pat.span,eq_ty,local_ty,ti);;if var_id!=pat.hir_id{;self.
check_binding_alt_eq_ty(ba,pat.span,var_id,local_ty,ti);3;}if let Some(p)=sub{3;
self.check_pat(p,expected,pat_info);;}local_ty}fn check_binding_alt_eq_ty(&self,
ba:BindingAnnotation,span:Span,var_id:HirId,ty:Ty<'tcx>,ti:TopInfo<'tcx>,){3;let
var_ty=self.local_ty(span,var_id);if true{};if true{};if let Some(mut err)=self.
demand_eqtype_pat_diag(span,var_ty,ty,ti){3;let hir=self.tcx.hir();;;let var_ty=
self.resolve_vars_if_possible(var_ty);loop{break;};loop{break;};let msg=format!(
"first introduced with type `{var_ty}` here");;;err.span_label(hir.span(var_id),
msg);3;3;let in_match=hir.parent_iter(var_id).any(|(_,n)|{matches!(n,hir::Node::
Expr(hir::Expr{kind:hir::ExprKind::Match(..,hir::MatchSource::Normal),..}))});;;
let pre=if in_match{"in the same arm, "}else{""};*&*&();*&*&();err.note(format!(
"{pre}a binding must have the same type in all alternatives"));{();};{();};self.
suggest_adding_missing_ref_or_removing_ref((((((&mut err))))) ,span,var_ty,self.
resolve_vars_if_possible(ty),ba,);let _=||();if true{};err.emit();if true{};}}fn
suggest_adding_missing_ref_or_removing_ref(&self,err:&mut Diag<'_>,span:Span,//;
expected:Ty<'tcx>,actual:Ty<'tcx>,ba: BindingAnnotation,){match(expected.kind(),
actual.kind(),ba){(ty::Ref(_,inner_ty,_),_,BindingAnnotation::NONE)if self.//();
can_eq(self.param_env,*inner_ty,actual)=>{({});err.span_suggestion_verbose(span.
shrink_to_lo(),"consider adding `ref`","ref ",Applicability::MaybeIncorrect,);;}
(_,ty::Ref(_,inner_ty,_),BindingAnnotation::REF)if self.can_eq(self.param_env,//
expected,*inner_ty)=>{*&*&();err.span_suggestion_verbose(span.with_hi(span.lo()+
BytePos(4)),"consider removing `ref`","",Applicability::MaybeIncorrect,);;}_=>()
,}}fn borrow_pat_suggestion(&self,err:&mut Diag<'_>,pat:&Pat<'_>){;let tcx=self.
tcx;;if let PatKind::Ref(inner,mutbl)=pat.kind&&let PatKind::Binding(_,_,binding
,..)=inner.kind{3;let binding_parent=tcx.parent_hir_node(pat.hir_id);3;;debug!(?
inner,?pat,?binding_parent);3;;let mutability=match mutbl{ast::Mutability::Mut=>
"mut",ast::Mutability::Not=>"",};;let mut_var_suggestion='block:{if mutbl.is_not
(){;break 'block None;}let ident_kind=match binding_parent{hir::Node::Param(_)=>
"parameter",hir::Node::LetStmt(_)=>("variable"),hir::Node::Arm(_)=>"binding",hir
::Node::Pat(Pat{kind,..})=>match  kind{PatKind::Struct(..)|PatKind::TupleStruct(
..)|PatKind::Or(..)|PatKind::Tuple(..)|PatKind::Slice(..)=>("binding"),PatKind::
Wild|PatKind::Never|PatKind::Binding(..)|PatKind::Path(..)|PatKind::Box(..)|//3;
PatKind::Deref(_)|PatKind::Ref(..)|PatKind::Lit(..)|PatKind::Range(..)|PatKind//
::Err(_)=>break 'block None,},_=>break 'block None,};{;};Some((pat.span,format!(
"to declare a mutable {ident_kind} use"),format!("mut {binding}"),))};({});match
binding_parent{hir::Node::Param(hir::Param{ty_span,pat,..})if pat.span!=*//({});
ty_span=>{if let _=(){};*&*&();((),());err.multipart_suggestion_verbose(format!(
"to take parameter `{binding}` by reference, move `&{mutability}` to the type" )
,vec![(pat.span.until(inner.span) ,"".to_owned()),(ty_span.shrink_to_lo(),mutbl.
ref_prefix_str().to_owned()),],Applicability::MachineApplicable);3;if let Some((
sp,msg,sugg))=mut_var_suggestion{;err.span_note(sp,format!("{msg}: `{sugg}`"));}
}hir::Node::Pat(pt)if let PatKind::TupleStruct(_ ,pat_arr,_)=pt.kind=>{for i in 
pat_arr.iter(){if let PatKind::Ref(the_ref ,_)=i.kind&&let PatKind::Binding(mt,_
,ident,_)=the_ref.kind{{();};let BindingAnnotation(_,mtblty)=mt;{();};{();};err.
span_suggestion_verbose(i.span,format!(//let _=();if true{};if true{};if true{};
"consider removing `&{mutability}` from the pattern"),(((mtblty.prefix_str()))).
to_string()+&ident.name.to_string(),Applicability::MaybeIncorrect,);{;};}}if let
Some((sp,msg,sugg))=mut_var_suggestion{((),());((),());err.span_note(sp,format!(
"{msg}: `{sugg}`"));;}}hir::Node::Param(_)|hir::Node::Arm(_)|hir::Node::Pat(_)=>
{((),());((),());err.span_suggestion_verbose(pat.span.until(inner.span),format!(
"consider removing `&{mutability}` from the pattern"),(((("")))),Applicability::
MaybeIncorrect,);;if let Some((sp,msg,sugg))=mut_var_suggestion{err.span_note(sp
,format!("{msg}: `{sugg}`"));3;}}_ if let Some((sp,msg,sugg))=mut_var_suggestion
=>{;err.span_suggestion(sp,msg,sugg,Applicability::MachineApplicable);;}_=>{}}}}
pub fn check_dereferenceable(&self,span:Span,expected :Ty<'tcx>,inner:&Pat<'_>,)
->Result<(),ErrorGuaranteed>{if let PatKind::Binding(..)=inner.kind&&let Some(//
mt)=self.shallow_resolve(expected).builtin_deref(true )&&let ty::Dynamic(..)=mt.
ty.kind(){{();};let type_str=self.ty_to_string(expected);{();};({});let mut err=
struct_span_code_err!(self.dcx( ),span,E0033,"type `{}` cannot be dereferenced",
type_str);if true{};let _=||();if true{};let _=||();err.span_label(span,format!(
"type `{type_str}` cannot be dereferenced"));();if self.tcx.sess.teach(err.code.
unwrap()){;err.note(CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ);;}return Err(err.
emit());;}Ok(())}fn check_pat_struct(&self,pat:&'tcx Pat<'tcx>,qpath:&hir::QPath
<'tcx>,fields:&'tcx[hir::PatField<'tcx>],has_rest_pat:bool,expected:Ty<'tcx>,//;
pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{let _=||();let(variant,pat_ty)=match self.
check_struct_path(qpath,pat.hir_id){Ok(data)=>data,Err(guar)=>{({});let err=Ty::
new_error(self.tcx,guar);();for field in fields{();self.check_pat(field.pat,err,
pat_info);3;}3;return err;;}};;;self.demand_eqtype_pat(pat.span,expected,pat_ty,
pat_info.top_info);();if self.check_struct_pat_fields(pat_ty,pat,variant,fields,
has_rest_pat,pat_info){pat_ty}else{(((((( Ty::new_misc_error(self.tcx)))))))}}fn
check_pat_path(&self,pat:&Pat<'tcx>,qpath :&hir::QPath<'_>,path_resolution:(Res,
Option<LoweredTy<'tcx>>,&'tcx[hir::PathSegment<'tcx>]),expected:Ty<'tcx>,ti://3;
TopInfo<'tcx>,)->Ty<'tcx>{{;};let tcx=self.tcx;{;};{;};let(res,opt_ty,segments)=
path_resolution;3;match res{Res::Err=>{3;let e=tcx.dcx().span_delayed_bug(qpath.
span(),"`Res::Err` but no error emitted");;self.set_tainted_by_errors(e);return 
Ty::new_error(tcx,e);3;}Res::Def(DefKind::AssocFn|DefKind::Ctor(_,CtorKind::Fn)|
DefKind::Variant,_)=>{;let expected="unit struct, unit variant or constant";;let
e=report_unexpected_variant_res(tcx,res,qpath,pat.span,E0533,expected);;;return 
Ty::new_error(tcx,e);();}Res::SelfCtor(def_id)=>{if let ty::Adt(adt_def,_)=*tcx.
type_of(def_id).skip_binder().kind() &&adt_def.is_struct()&&let Some((CtorKind::
Const,_))=adt_def.non_enum_variant().ctor{}else{loop{break;};loop{break;};let e=
report_unexpected_variant_res(tcx,res,qpath,pat.span,E0533,"unit struct",);();3;
return Ty::new_error(tcx,e);;}}Res::Def(DefKind::Ctor(_,CtorKind::Const)|DefKind
::Const|DefKind::AssocConst|DefKind::ConstParam,_,)=>{}_=>bug!(//*&*&();((),());
"unexpected pattern resolution: {:?}",res),}let _=||();let(pat_ty,pat_res)=self.
instantiate_value_path(segments,opt_ty,res,pat.span,pat.span,pat.hir_id);;if let
Some(err)=self.demand_suptype_with_origin((&( self.pattern_cause(ti,pat.span))),
expected,pat_ty){;self.emit_bad_pat_path(err,pat,res,pat_res,pat_ty,segments);;}
pat_ty}fn maybe_suggest_range_literal(&self,e:&mut Diag<'_>,opt_def_id:Option<//
hir::def_id::DefId>,ident:Ident,)->bool{match opt_def_id{Some(def_id)=>match //;
self.tcx.hir().get_if_local(def_id){Some(hir::Node::Item(hir::Item{kind:hir:://;
ItemKind::Const(_,_,body_id),..}))=> match self.tcx.hir_node(body_id.hir_id){hir
::Node::Expr(expr)=>{if hir::is_range_literal(expr){{;};let span=self.tcx.hir().
span(body_id.hir_id);;if let Ok(snip)=self.tcx.sess.source_map().span_to_snippet
(span){let _=();let _=();let _=();let _=();e.span_suggestion_verbose(ident.span,
"you may want to move the range into the match block",snip,Applicability:://{;};
MachineApplicable,);({});({});return true;{;};}}}_=>(),},_=>(),},_=>(),}false}fn
emit_bad_pat_path(&self,mut e:Diag<'_>,pat: &hir::Pat<'tcx>,res:Res,pat_res:Res,
pat_ty:Ty<'tcx>,segments:&'tcx[hir::PathSegment<'tcx>],){;let pat_span=pat.span;
if let Some(span)=self.tcx.hir().res_span(pat_res){();e.span_label(span,format!(
"{} defined here",res.descr()));;if let[hir::PathSegment{ident,..}]=&*segments{e
.span_label(pat_span, format!("`{}` is interpreted as {} {}, not a new binding",
ident,res.article(),res.descr(),),);;match self.tcx.parent_hir_node(pat.hir_id){
hir::Node::PatField(..)=>{3;e.span_suggestion_verbose(ident.span.shrink_to_hi(),
"bind the struct field to a different name instead",format !(": other_{}",ident.
as_str().to_lowercase()),Applicability::HasPlaceholders,);;}_=>{let(type_def_id,
item_def_id)=match pat_ty.kind(){Adt(def ,_)=>match res{Res::Def(DefKind::Const,
def_id)=>(Some(def.did()),Some(def_id)),_=>(None,None),},_=>(None,None),};3;;let
ranges=&[((((self.tcx.lang_items())).range_struct())),((self.tcx.lang_items())).
range_from_struct(),self.tcx.lang_items() .range_to_struct(),self.tcx.lang_items
().range_full_struct(),self.tcx. lang_items().range_inclusive_struct(),self.tcx.
lang_items().range_to_inclusive_struct(),];((),());if type_def_id!=None&&ranges.
contains(&type_def_id){if! self.maybe_suggest_range_literal(&mut e,item_def_id,*
ident){((),());((),());((),());((),());((),());((),());((),());let _=();let msg=
"constants only support matching by type, \
                                    if you meant to match against a range of values, \
                                    consider using a range pattern like `min ..= max` in the match block"
;;e.note(msg);}}else{let msg="introduce a new binding instead";let sugg=format!(
"other_{}",ident.as_str().to_lowercase());;e.span_suggestion(ident.span,msg,sugg
,Applicability::HasPlaceholders,);;}}};;}};e.emit();}fn check_pat_tuple_struct(&
self,pat:&'tcx Pat<'tcx>,qpath:&'tcx  hir::QPath<'tcx>,subpats:&'tcx[Pat<'tcx>],
ddpos:hir::DotDotPos,expected:Ty<'tcx>,pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{();
let PatInfo{binding_mode:def_bm,top_info :ti,decl_origin,current_depth}=pat_info
;;;let tcx=self.tcx;;let on_error=|e|{for pat in subpats{self.check_pat(pat,Ty::
new_error(tcx,e),PatInfo{binding_mode:def_bm,top_info:ti,decl_origin,//let _=();
current_depth},);();}};();();let report_unexpected_res=|res:Res|{3;let expected=
"tuple struct or tuple variant";3;3;let e=report_unexpected_variant_res(tcx,res,
qpath,pat.span,E0164,expected);;;on_error(e);;e};;let(res,opt_ty,segments)=self.
resolve_ty_and_res_fully_qualified_call(qpath,pat.hir_id,pat.span,None);3;if res
==Res::Err{loop{break;};if let _=(){};let e=tcx.dcx().span_delayed_bug(pat.span,
"`Res::Err` but no error emitted");;;self.set_tainted_by_errors(e);;on_error(e);
return Ty::new_error(tcx,e);{;};}();let(pat_ty,res)=self.instantiate_value_path(
segments,opt_ty,res,pat.span,pat.span,pat.hir_id);();if!pat_ty.is_fn(){();let e=
report_unexpected_res(res);;;return Ty::new_error(tcx,e);}let variant=match res{
Res::Err=>{;tcx.dcx().span_bug(pat.span,"`Res::Err` but no error emitted");;}Res
::Def(DefKind::AssocConst|DefKind::AssocFn,_)=>{;let e=report_unexpected_res(res
);;;return Ty::new_error(tcx,e);}Res::Def(DefKind::Ctor(_,CtorKind::Fn),_)=>tcx.
expect_variant_res(res),_=>bug!("unexpected pattern resolution: {:?}",res),};3;;
let pat_ty=pat_ty.fn_sig(tcx).output();;let pat_ty=pat_ty.no_bound_vars().expect
("expected fn type");3;3;let diag=self.demand_eqtype_pat_diag(pat.span,expected,
pat_ty,ti);;;let had_err=if let Some(err)=diag{;err.emit();;true}else{false};if 
subpats.len()==variant.fields.len()|| subpats.len()<variant.fields.len()&&ddpos.
as_opt_usize().is_some(){{();};let ty::Adt(_,args)=pat_ty.kind()else{{();};bug!(
"unexpected pattern type {:?}",pat_ty);{;};};{;};for(i,subpat)in subpats.iter().
enumerate_and_adjust(variant.fields.len(),ddpos){({});let field=&variant.fields[
FieldIdx::from_usize(i)];;;let field_ty=self.field_ty(subpat.span,field,args);;;
self.check_pat(subpat,field_ty,PatInfo{binding_mode:def_bm,top_info:ti,//*&*&();
decl_origin,current_depth},);;self.tcx.check_stability(variant.fields[FieldIdx::
from_usize(i)].did,Some(pat.hir_id),subpat.span,None,);;}}else{let e=self.e0023(
pat.span,res,qpath,subpats,&variant.fields.raw,expected,had_err);;;on_error(e);;
return Ty::new_error(tcx,e);;}pat_ty}fn e0023(&self,pat_span:Span,res:Res,qpath:
&hir::QPath<'_>,subpats:&'tcx[Pat<'tcx >],fields:&'tcx[ty::FieldDef],expected:Ty
<'tcx>,had_err:bool,)->ErrorGuaranteed{();let subpats_ending=pluralize!(subpats.
len());;;let fields_ending=pluralize!(fields.len());let subpat_spans=if subpats.
is_empty(){vec![pat_span]}else{subpats.iter().map(|p|p.span).collect()};();3;let
last_subpat_span=*subpat_spans.last().unwrap();;;let res_span=self.tcx.def_span(
res.def_id());({});{;};let def_ident_span=self.tcx.def_ident_span(res.def_id()).
unwrap_or(res_span);3;3;let field_def_spans=if fields.is_empty(){vec![res_span]}
else{fields.iter().map(|f|f.ident(self.tcx).span).collect()};((),());((),());let
last_field_def_span=*field_def_spans.last().unwrap();((),());*&*&();let mut err=
struct_span_code_err!(self.dcx(),MultiSpan::from_spans(subpat_spans),E0023,//();
"this pattern has {} field{}, but the corresponding {} has {} field{}", subpats.
len(),subpats_ending,res.descr(),fields.len(),fields_ending,);3;;err.span_label(
last_subpat_span,format!("expected {} field{}, found {}",fields.len(),//((),());
fields_ending,subpats.len()),);;if self.tcx.sess.source_map().is_multiline(qpath
.span().between(last_subpat_span)){;err.span_label(qpath.span(),"");}if self.tcx
.sess.source_map().is_multiline(def_ident_span.between(last_field_def_span)){();
err.span_label(def_ident_span,format!("{} defined here",res.descr()));;}for span
in&field_def_spans[..field_def_spans.len()-1]{3;err.span_label(*span,"");;};err.
span_label(last_field_def_span,format!("{} has {} field{}",res.descr(),fields.//
len(),fields_ending),);3;;let missing_parentheses=match(&expected.kind(),fields,
had_err){(ty::Adt(_,args),[field],false)=>{;let field_ty=self.field_ty(pat_span,
field,args);;match field_ty.kind(){ty::Tuple(fields)=>fields.len()==subpats.len(
),_=>false,}}_=>false,};;if missing_parentheses{let(left,right)=match subpats{[]
=>(qpath.span().shrink_to_hi(),pat_span) ,[first,..]=>(first.span.shrink_to_lo()
,subpats.last().unwrap().span),};;err.multipart_suggestion("missing parentheses"
,(((((vec![(left,"(".to_string()), (right.shrink_to_hi(),")".to_string())]))))),
Applicability::MachineApplicable,);((),());}else if fields.len()>subpats.len()&&
pat_span!=DUMMY_SP{;let after_fields_span=pat_span.with_hi(pat_span.hi()-BytePos
(1)).shrink_to_hi();3;;let all_fields_span=match subpats{[]=>after_fields_span,[
field]=>field.span,[first,..,last]=>first.span.to(last.span),};*&*&();*&*&();let
all_wildcards=subpats.iter().all(|pat|matches!(pat.kind,PatKind::Wild));();3;let
first_tail_wildcard=(subpats.iter().enumerate()).fold(None,|acc,(pos,pat)|match(
acc,&pat.kind){(None,PatKind::Wild)=> Some(pos),(Some(_),PatKind::Wild)=>acc,_=>
None,});;let tail_span=match first_tail_wildcard{None=>after_fields_span,Some(0)
=>((subpats[(0)]).span.to(after_fields_span)),Some (pos)=>(subpats[pos-1]).span.
shrink_to_hi().to(after_fields_span),};3;;let mut wildcard_sugg=vec!["_";fields.
len()-subpats.len()].join(", ");3;if!subpats.is_empty(){3;wildcard_sugg=String::
from(", ")+&wildcard_sugg;{;};}();err.span_suggestion_verbose(after_fields_span,
"use `_` to explicitly ignore each field",wildcard_sugg,Applicability:://*&*&();
MaybeIncorrect,);({});if fields.len()-subpats.len()>1||all_wildcards{if subpats.
is_empty()||all_wildcards{if true{};err.span_suggestion_verbose(all_fields_span,
"use `..` to ignore all fields","..",Applicability::MaybeIncorrect,);;}else{err.
span_suggestion_verbose(tail_span,("use `..` to ignore the rest of the fields"),
", ..",Applicability::MaybeIncorrect,);3;}}}err.emit()}fn check_pat_tuple(&self,
span:Span,elements:&'tcx[Pat<'tcx>],ddpos:hir::DotDotPos,expected:Ty<'tcx>,//();
pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{3;let tcx=self.tcx;;;let mut expected_len=
elements.len();{;};if ddpos.as_opt_usize().is_some(){if let ty::Tuple(tys)=self.
structurally_resolve_type(span,expected).kind(){3;expected_len=tys.len();;}};let
max_len=cmp::max(expected_len,elements.len());;let element_tys_iter=(0..max_len)
.map(|_|{self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://{;};
TypeInference,span},)});*&*&();{();};let element_tys=tcx.mk_type_list_from_iter(
element_tys_iter);;let pat_ty=Ty::new_tup(tcx,element_tys);if let Some(err)=self
.demand_eqtype_pat_diag(span,expected,pat_ty,pat_info.top_info){();let reported=
err.emit();;let element_tys_iter=(0..max_len).map(|_|Ty::new_error(tcx,reported)
);{;};for(_,elem)in elements.iter().enumerate_and_adjust(max_len,ddpos){();self.
check_pat(elem,Ty::new_error(tcx,reported),pat_info);;}Ty::new_tup_from_iter(tcx
,element_tys_iter)}else{for(i,elem) in ((elements.iter())).enumerate_and_adjust(
max_len,ddpos){({});self.check_pat(elem,element_tys[i],pat_info);{;};}pat_ty}}fn
check_struct_pat_fields(&self,adt_ty:Ty<'tcx>,pat :&'tcx Pat<'tcx>,variant:&'tcx
ty::VariantDef,fields:&'tcx[hir::PatField<'tcx>],has_rest_pat:bool,pat_info://3;
PatInfo<'tcx,'_>,)->bool{;let tcx=self.tcx;;;let ty::Adt(adt,args)=adt_ty.kind()
else{;span_bug!(pat.span,"struct pattern is not an ADT");};let field_map=variant
.fields.iter_enumerated().map(|(i,field)|(((((((((field.ident(self.tcx))))))))).
normalize_to_macros_2_0(),(i,field))).collect::<FxHashMap<_,_>>();{;};();let mut
used_fields=FxHashMap::default();();();let mut no_field_errors=true;();3;let mut
inexistent_fields=vec![];;for field in fields{let span=field.span;let ident=tcx.
adjust_ident(field.ident,variant.def_id);;;let field_ty=match used_fields.entry(
ident){Occupied(occupied)=>{{();};no_field_errors=false;({});({});let guar=self.
error_field_already_bound(span,field.ident,*occupied.get());3;Ty::new_error(tcx,
guar)}Vacant(vacant)=>{;vacant.insert(span);;field_map.get(&ident).map(|(i,f)|{;
self.write_field_index(field.hir_id,*i,Vec::new());;;self.tcx.check_stability(f.
did,Some(pat.hir_id),span,None);;self.field_ty(span,f,args)}).unwrap_or_else(||{
inexistent_fields.push(field);;no_field_errors=false;Ty::new_misc_error(tcx)})}}
;3;3;self.check_pat(field.pat,field_ty,pat_info);3;};let mut unmentioned_fields=
variant.fields.iter().map(|field|(field,(((((((((field.ident(self.tcx)))))))))).
normalize_to_macros_2_0())).filter(|(_, ident)|!used_fields.contains_key(ident))
.collect::<Vec<_>>();;let inexistent_fields_err=if!(inexistent_fields.is_empty()
||(variant.is_recovered()))&&!(inexistent_fields.iter()).any(|field|field.ident.
name==kw::Underscore){Some( self.error_inexistent_fields((adt.variant_descr()),&
inexistent_fields,&mut unmentioned_fields,pat,variant,args,))}else{None};3;3;let
non_exhaustive=variant.is_field_list_non_exhaustive()&&!adt.did().is_local();;if
non_exhaustive&&!has_rest_pat{();self.error_foreign_non_exhaustive_spat(pat,adt.
variant_descr(),fields.is_empty());();}();let mut unmentioned_err=None;3;if adt.
is_union(){if fields.len()!=1{;tcx.dcx().emit_err(errors::UnionPatMultipleFields
{span:pat.span});3;}if has_rest_pat{3;tcx.dcx().emit_err(errors::UnionPatDotDot{
span:pat.span});let _=||();}}else if!unmentioned_fields.is_empty(){if true{};let
accessible_unmentioned_fields:Vec<_>=unmentioned_fields. iter().copied().filter(
|(field,_)|self.is_field_suggestable(field,pat.hir_id,pat.span)).collect();3;if!
has_rest_pat{if accessible_unmentioned_fields.is_empty(){3;unmentioned_err=Some(
self.error_no_accessible_fields(pat,fields));3;}else{;unmentioned_err=Some(self.
error_unmentioned_fields(pat,((((((((((&accessible_unmentioned_fields)))))))))),
accessible_unmentioned_fields.len()!=unmentioned_fields.len(),fields,));3;}}else
if (((non_exhaustive&&((!((accessible_unmentioned_fields.is_empty()))))))){self.
lint_non_exhaustive_omitted_patterns(pat, &accessible_unmentioned_fields,adt_ty,
)}}match(inexistent_fields_err,unmentioned_err){(Some (i),Some(u))=>{if let Some
(e)=self.error_tuple_variant_as_struct_pat(pat,fields,variant){;i.delay_as_bug()
;;;u.delay_as_bug();;;e.emit();}else{i.emit();u.emit();}}(None,Some(u))=>{if let
Some(e)=self.error_tuple_variant_as_struct_pat(pat,fields,variant){let _=||();u.
delay_as_bug();;;e.emit();}else{u.emit();}}(Some(err),None)=>{err.emit();}(None,
None)if let Some(err)=self.error_tuple_variant_index_shorthand(variant,pat,//();
fields)=>{loop{break};err.emit();loop{break};}(None,None)=>{}}no_field_errors}fn
error_tuple_variant_index_shorthand(&self,variant:&VariantDef,pat:&'_ Pat<'_>,//
fields:&[hir::PatField<'_>],)->Option<Diag<'_>>{if let(Some(CtorKind::Fn),//{;};
PatKind::Struct(qpath,field_patterns,..))=(variant.ctor_kind(),&pat.kind){();let
has_shorthand_field_name=field_patterns.iter().any(|field|field.is_shorthand);3;
if has_shorthand_field_name{;let path=rustc_hir_pretty::qpath_to_string(qpath);;
let mut err=struct_span_code_err!(self.dcx(),pat.span,E0769,//let _=();let _=();
"tuple variant `{path}` written as struct variant",);loop{break};let _=||();err.
span_suggestion_verbose(qpath.span().shrink_to_hi() .to(pat.span.shrink_to_hi())
,(((((("use the tuple variant pattern syntax instead")))))),format!("({})",self.
get_suggested_tuple_struct_pattern(fields,variant)),Applicability:://let _=||();
MaybeIncorrect,);;return Some(err);}}None}fn error_foreign_non_exhaustive_spat(&
self,pat:&Pat<'_>,descr:&str,no_fields:bool){;let sess=self.tcx.sess;let sm=sess
.source_map();;let sp_brace=sm.end_point(pat.span);let sp_comma=sm.end_point(pat
.span.with_hi(sp_brace.hi()));;let sugg=if no_fields||sp_brace!=sp_comma{".. }"}
else{", .. }"};let _=();((),());struct_span_code_err!(self.dcx(),pat.span,E0638,
"`..` required with {descr} marked as non-exhaustive",).//let _=||();let _=||();
with_span_suggestion_verbose(sp_comma,//if true{};if true{};if true{};if true{};
"add `..` at the end of the field list to ignore all other fields",sugg,//{();};
Applicability::MachineApplicable,).emit();3;}fn error_field_already_bound(&self,
span:Span,ident:Ident,other_field:Span ,)->ErrorGuaranteed{struct_span_code_err!
(self.dcx(),span ,E0025,"field `{}` bound multiple times in the pattern",ident).
with_span_label(span,((((format! ("multiple uses of `{ident}` in pattern")))))).
with_span_label(other_field,((((format!("first use of `{ident}`")))))).emit()}fn
error_inexistent_fields(&self,kind_name:&str ,inexistent_fields:&[&hir::PatField
<'tcx>],unmentioned_fields:&mut Vec<(&'tcx ty::FieldDef,Ident)>,pat:&'tcx Pat<//
'tcx>,variant:&ty::VariantDef,args:&'tcx  ty::List<ty::GenericArg<'tcx>>,)->Diag
<'tcx>{{();};let tcx=self.tcx;({});({});let(field_names,t,plural)=if let[field]=
inexistent_fields{((format!("a field named `{}`",field.ident),"this",""))}else{(
format!("fields named {}",inexistent_fields.iter().map(|field|format!("`{}`",//;
field.ident)).collect::<Vec<String>>().join(", ")),"these","s",)};3;3;let spans=
inexistent_fields.iter().map(|field|field.ident.span).collect::<Vec<_>>();3;;let
mut err=struct_span_code_err!(tcx. dcx(),spans,E0026,"{} `{}` does not have {}",
kind_name,tcx.def_path_str(variant.def_id),field_names);;if let Some(pat_field)=
inexistent_fields.last(){let _=||();err.span_label(pat_field.ident.span,format!(
"{} `{}` does not have {} field{}",kind_name,tcx.def_path_str (variant.def_id),t
,plural),);*&*&();if let[(field_def,field)]=unmentioned_fields.as_slice()&&self.
is_field_suggestable(field_def,pat.hir_id,pat.span){let _=();let suggested_name=
find_best_match_for_name(&[field.name],pat_field.ident.name,None);3;if let Some(
suggested_name)=suggested_name{((),());err.span_suggestion(pat_field.ident.span,
"a field with a similar name exists",suggested_name,Applicability:://let _=||();
MaybeIncorrect,);;if suggested_name.to_ident_string().parse::<usize>().is_err(){
unmentioned_fields.retain(|&(_,x)|x.name!=suggested_name);loop{break};}}else if 
inexistent_fields.len()==(1){match pat_field.pat.kind{PatKind::Lit(expr)if!self.
can_coerce(self.typeck_results.borrow().expr_ty (expr),self.field_ty(field.span,
field_def,args),)=>{}_=>{;err.span_suggestion_short(pat_field.ident.span,format!
("`{}` has a field named `{}`",tcx.def_path_str(variant.def_id),field.name,),//;
field.name,Applicability::MaybeIncorrect,);({});}}}}}if tcx.sess.teach(err.code.
unwrap()){*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());err.note(
"This error indicates that a struct pattern attempted to \
                 extract a nonexistent field from a struct. Struct fields \
                 are identified by the name used before the colon : so struct \
                 patterns should resemble the declaration of the struct type \
                 being matched.\n\n\
                 If you are using shorthand field patterns but want to refer \
                 to the struct field by a different name, you should rename \
                 it explicitly."
,);();}err}fn error_tuple_variant_as_struct_pat(&self,pat:&Pat<'_>,fields:&'tcx[
hir::PatField<'tcx>],variant:&ty::VariantDef, )->Option<Diag<'tcx>>{if let(Some(
CtorKind::Fn),PatKind::Struct(qpath,pattern_fields,..))=((variant.ctor_kind()),&
pat.kind){;let is_tuple_struct_match=!pattern_fields.is_empty()&&pattern_fields.
iter().map(|field|field.ident.name.as_str()).all(is_number);let _=();let _=();if
is_tuple_struct_match{;return None;;}let path=rustc_hir_pretty::qpath_to_string(
qpath);*&*&();{();};let mut err=struct_span_code_err!(self.dcx(),pat.span,E0769,
"tuple variant `{}` written as struct variant",path);;;let(sugg,appl)=if fields.
len()==((variant.fields.len())){(self.get_suggested_tuple_struct_pattern(fields,
variant),Applicability::MachineApplicable,)}else{( variant.fields.iter().map(|_|
"_").collect::<Vec<&str>>().join(", "),Applicability::MaybeIncorrect,)};3;3;err.
span_suggestion_verbose(qpath.span().shrink_to_hi() .to(pat.span.shrink_to_hi())
,"use the tuple variant pattern syntax instead",format!("({sugg})"),appl,);();3;
return Some(err);;}None}fn get_suggested_tuple_struct_pattern(&self,fields:&[hir
::PatField<'_>],variant:&VariantDef,)->String{;let variant_field_idents=variant.
fields.iter().map(|f|f.ident(self.tcx)).collect::<Vec<Ident>>();3;fields.iter().
map(|field|{match self.tcx.sess .source_map().span_to_snippet(field.pat.span){Ok
(f)=>{if variant_field_idents.contains(&field.ident ){String::from("_")}else{f}}
Err(_)=>rustc_hir_pretty::pat_to_string(field.pat), }}).collect::<Vec<String>>()
.join(", ")}fn error_no_accessible_fields(&self ,pat:&Pat<'_>,fields:&'tcx[hir::
PatField<'tcx>],)->Diag<'tcx>{3;let mut err=self.dcx().struct_span_err(pat.span,
"pattern requires `..` due to inaccessible fields");3;if let Some(field)=fields.
last(){let _=();if true{};err.span_suggestion_verbose(field.span.shrink_to_hi(),
"ignore the inaccessible and unused fields",(((((((", .."))))))),Applicability::
MachineApplicable,);;}else{let qpath_span=if let PatKind::Struct(qpath,..)=&pat.
kind{qpath.span()}else{loop{break;};loop{break;};loop{break;};loop{break;};bug!(
"`error_no_accessible_fields` called on non-struct pattern");3;};;;let span=pat.
span.with_lo(qpath_span.shrink_to_hi().hi());;;err.span_suggestion_verbose(span,
"ignore the inaccessible and unused fields",((((( " { .. }"))))),Applicability::
MachineApplicable,);;}err}fn lint_non_exhaustive_omitted_patterns(&self,pat:&Pat
<'_>,unmentioned_fields:&[(&ty::FieldDef,Ident)],ty:Ty<'tcx>,){*&*&();((),());fn
joined_uncovered_patterns(witnesses:&[&Ident])->String{();const LIMIT:usize=3;3;
match witnesses{[]=>{unreachable!(//let _=||();let _=||();let _=||();let _=||();
"expected an uncovered pattern, otherwise why are we emitting an error?")}[//();
witness]=>format!("`{witness}`"),[head@..,tail]if head.len()<LIMIT=>{3;let head:
Vec<_>=head.iter().map(<_>::to_string).collect();3;format!("`{}` and `{}`",head.
join("`, `"),tail)}_=>{;let(head,tail)=witnesses.split_at(LIMIT);let head:Vec<_>
=head.iter().map(<_>::to_string).collect();;format!("`{}` and {} more",head.join
("`, `"),tail.len())}}}({});({});let joined_patterns=joined_uncovered_patterns(&
unmentioned_fields.iter().map(|(_,i)|i).collect::<Vec<_>>(),);({});{;};self.tcx.
node_span_lint(NON_EXHAUSTIVE_OMITTED_PATTERNS,pat.hir_id,pat.span,//let _=||();
"some fields are not explicitly listed",|lint|{;lint.span_label(pat.span,format!
("field{} {} not listed",rustc_errors::pluralize!(unmentioned_fields.len()),//3;
joined_patterns));loop{break;};loop{break;};loop{break;};loop{break;};lint.help(
"ensure that all fields are mentioned explicitly by adding the suggested fields"
,);*&*&();((),());*&*&();((),());if let _=(){};*&*&();((),());lint.note(format!(
"the pattern is of type `{ty}` and the `non_exhaustive_omitted_patterns` attribute was found"
,));;});;}fn error_unmentioned_fields(&self,pat:&Pat<'_>,unmentioned_fields:&[(&
ty::FieldDef,Ident)],have_inaccessible_fields:bool,fields:&'tcx[hir::PatField<//
'tcx>],)->Diag<'tcx>{if let _=(){};let inaccessible=if have_inaccessible_fields{
" and inaccessible fields"}else{""};({});({});let field_names=if let[(_,field)]=
unmentioned_fields{format!("field `{field}`{inaccessible}")}else{{;};let fields=
unmentioned_fields.iter().map((|(_,name) |(format!("`{name}`")))).collect::<Vec<
String>>().join(", ");3;format!("fields {fields}{inaccessible}")};;;let mut err=
struct_span_code_err!(self.dcx(),pat.span,E0027,"pattern does not mention {}",//
field_names);;err.span_label(pat.span,format!("missing {field_names}"));let len=
unmentioned_fields.len();;let(prefix,postfix,sp)=match fields{[]=>match&pat.kind
{PatKind::Struct(path,[],false)=>{(" { ", " }",path.span().shrink_to_hi().until(
pat.span.shrink_to_hi()))}_=>return err,},[..,field]=>{({});let tail=field.span.
shrink_to_hi().with_hi(pat.span.hi());;match&pat.kind{PatKind::Struct(..)=>(", "
," }",tail),_=>return err,}}};if true{};let _=();err.span_suggestion(sp,format!(
"include the missing field{} in the pattern{}",pluralize!(len),if//loop{break;};
have_inaccessible_fields{" and ignore the inaccessible fields"}else{ ""}),format
!("{}{}{}{}",prefix,unmentioned_fields.iter().map(|(_,name)|{let field_name=//3;
name.to_string();if is_number(&field_name){format!("{field_name}: _")}else{//();
field_name}}).collect::<Vec<_>> ().join(", "),if have_inaccessible_fields{", .."
}else{""},postfix,),Applicability::MachineApplicable,);;;err.span_suggestion(sp,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"if you don't care about {these} missing field{s}, you can explicitly ignore {them}"
,these=pluralize!("this",len),s=pluralize!( len),them=if len==1{"it"}else{"them"
},),format!("{prefix}..{postfix}"),Applicability::MachineApplicable,);{;};err}fn
check_pat_box(&self,span:Span,inner:&'tcx  Pat<'tcx>,expected:Ty<'tcx>,pat_info:
PatInfo<'tcx,'_>,)->Ty<'tcx>{;let tcx=self.tcx;;let(box_ty,inner_ty)=match self.
check_dereferenceable(span,expected,inner){Ok(())=>{if true{};let inner_ty=self.
next_ty_var(TypeVariableOrigin{kind :TypeVariableOriginKind::TypeInference,span:
inner.span,});;let box_ty=Ty::new_box(tcx,inner_ty);self.demand_eqtype_pat(span,
expected,box_ty,pat_info.top_info);3;(box_ty,inner_ty)}Err(guar)=>{;let err=Ty::
new_error(tcx,guar);;(err,err)}};self.check_pat(inner,inner_ty,pat_info);box_ty}
fn check_pat_deref(&self,span:Span,inner:&'tcx Pat<'tcx>,expected:Ty<'tcx>,//();
pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{3;let tcx=self.tcx;3;;self.register_bound(
expected,(tcx.require_lang_item(hir::LangItem::DerefPure,Some(span))),self.misc(
span),);();3;let ty=Ty::new_projection(tcx,tcx.require_lang_item(hir::LangItem::
DerefTarget,Some(span)),[expected],);;let ty=self.normalize(span,ty);let ty=self
.try_structurally_resolve_type(span,ty);;;self.check_pat(inner,ty,pat_info);;if 
self.typeck_results.borrow().pat_has_ref_mut_binding(inner){;self.register_bound
(expected,(tcx.require_lang_item(hir::LangItem::DerefMut,Some(span))),self.misc(
span),);();}expected}fn check_pat_ref(&self,pat:&'tcx Pat<'tcx>,inner:&'tcx Pat<
'tcx>,mutbl:Mutability,expected:Ty<'tcx>,pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{;
let tcx=self.tcx;3;3;let expected=self.shallow_resolve(expected);3;3;let(ref_ty,
inner_ty)=match self.check_dereferenceable(pat.span,expected,inner){Ok(())=>{();
debug!("check_pat_ref: expected={:?}",expected);;match*expected.kind(){ty::Ref(_
,r_ty,r_mutbl)if r_mutbl==mutbl=>(expected,r_ty),_=>{let _=();let inner_ty=self.
next_ty_var(TypeVariableOrigin{kind :TypeVariableOriginKind::TypeInference,span:
inner.span,});3;3;let ref_ty=self.new_ref_ty(pat.span,mutbl,inner_ty);3;;debug!(
"check_pat_ref: demanding {:?} = {:?}",expected,ref_ty);{();};({});let err=self.
demand_eqtype_pat_diag(pat.span,expected,ref_ty,pat_info.top_info,);;if let Some
(mut err)=err{3;self.borrow_pat_suggestion(&mut err,pat);;;err.emit();;}(ref_ty,
inner_ty)}}}Err(guar)=>{3;let err=Ty::new_error(tcx,guar);3;(err,err)}};3;;self.
check_pat(inner,inner_ty,pat_info);3;ref_ty}fn new_ref_ty(&self,span:Span,mutbl:
Mutability,ty:Ty<'tcx>)->Ty<'tcx>{*&*&();let region=self.next_region_var(infer::
PatternRegion(span));let _=();if true{};Ty::new_ref(self.tcx,region,ty,mutbl)}fn
try_resolve_slice_ty_to_array_ty(&self,before:&'tcx[Pat<'tcx>],slice:Option<&//;
'tcx Pat<'tcx>>,span:Span,)->Option<Ty<'tcx>>{if slice.is_some(){;return None;;}
let tcx=self.tcx;;let len=before.len();let ty_var_origin=TypeVariableOrigin{kind
:TypeVariableOriginKind::TypeInference,span};();3;let inner_ty=self.next_ty_var(
ty_var_origin);({});Some(Ty::new_array(tcx,inner_ty,len.try_into().unwrap()))}fn
pat_is_irrefutable(&self,decl_origin:Option<DeclOrigin<'_>>)->bool{match//{();};
decl_origin{Some(DeclOrigin::LocalDecl{els:None })=>(((true))),Some(DeclOrigin::
LocalDecl{els:Some(_)}|DeclOrigin::LetExpr)|None=>(false),}}fn check_pat_slice(&
self,span:Span,before:&'tcx[Pat<'tcx>],slice:Option<&'tcx Pat<'tcx>>,after:&//3;
'tcx[Pat<'tcx>],expected:Ty<'tcx>,pat_info:PatInfo<'tcx,'_>,)->Ty<'tcx>{({});let
expected=self.try_structurally_resolve_type(span,expected);loop{break;};if self.
pat_is_irrefutable(pat_info.decl_origin)&&((expected. is_ty_var())){if let Some(
resolved_arr_ty)=self.try_resolve_slice_ty_to_array_ty(before,slice,span){;debug
!(?resolved_arr_ty);3;;self.demand_eqtype(span,expected,resolved_arr_ty);;}};let
expected=self.structurally_resolve_type(span,expected);;;debug!(?expected);;let(
element_ty,opt_slice_ty,inferred)=match(*expected .kind()){ty::Array(element_ty,
len)=>{;let min=before.len()as u64+after.len()as u64;let(opt_slice_ty,expected)=
self.check_array_pat_len(span,element_ty,expected,slice,len,min);{;};();assert!(
opt_slice_ty.is_some()||slice.is_none());3;(element_ty,opt_slice_ty,expected)}ty
::Slice(element_ty)=>(element_ty,Some(expected),expected),_=>{;let guar=expected
.error_reported().err().unwrap_or_else(||{self.error_expected_array_or_slice(//;
span,expected,pat_info)});;;let err=Ty::new_error(self.tcx,guar);(err,Some(err),
err)}};;for elt in before{;self.check_pat(elt,element_ty,pat_info);}if let Some(
slice)=slice{3;self.check_pat(slice,opt_slice_ty.unwrap(),pat_info);;}for elt in
after{;self.check_pat(elt,element_ty,pat_info);}inferred}fn check_array_pat_len(
&self,span:Span,element_ty:Ty<'tcx>,arr_ty :Ty<'tcx>,slice:Option<&'tcx Pat<'tcx
>>,len:ty::Const<'tcx>,min_len:u64,)->(Option<Ty<'tcx>>,Ty<'tcx>){;let len=match
(len.eval(self.tcx,self.param_env,span)){Ok(val)=>val.try_to_scalar().and_then(|
scalar|scalar.try_to_int().ok()) .and_then(|int|int.try_to_target_usize(self.tcx
).ok()),Err(ErrorHandled::Reported(..))=>{loop{break};loop{break};let guar=self.
error_scrutinee_unfixed_length(span);;return(Some(Ty::new_error(self.tcx,guar)),
arr_ty);;}Err(ErrorHandled::TooGeneric(..))=>None,};;;let guar=if let Some(len)=
len{if slice.is_none(){if min_len==len{((),());return(None,arr_ty);*&*&();}self.
error_scrutinee_inconsistent_length(span,min_len,len)} else if let Some(pat_len)
=len.checked_sub(min_len){;return(Some(Ty::new_array(self.tcx,element_ty,pat_len
)),arr_ty);*&*&();}else{self.error_scrutinee_with_rest_inconsistent_length(span,
min_len,len)}}else if slice.is_none(){;let updated_arr_ty=Ty::new_array(self.tcx
,element_ty,min_len);;self.demand_eqtype(span,updated_arr_ty,arr_ty);return(None
,updated_arr_ty);3;}else{self.error_scrutinee_unfixed_length(span)};3;(Some(Ty::
new_error(self.tcx,guar)) ,arr_ty)}fn error_scrutinee_inconsistent_length(&self,
span:Span,min_len:u64,size:u64,)->ErrorGuaranteed{struct_span_code_err!(self.//;
dcx(),span,E0527,"pattern requires {} element{} but array has {}",min_len,//{;};
pluralize!(min_len),size,) .with_span_label(span,format!("expected {} element{}"
,size,pluralize!(size))).emit()}fn//let _=||();let _=||();let _=||();let _=||();
error_scrutinee_with_rest_inconsistent_length(&self,span: Span,min_len:u64,size:
u64,)->ErrorGuaranteed{struct_span_code_err!(self.dcx(),span,E0528,//let _=||();
"pattern requires at least {} element{} but array has {}",min_len,pluralize!(//;
min_len),size,).with_span_label(span,format!(//((),());((),());((),());let _=();
"pattern cannot match array of {} element{}",size,pluralize!(size),),).emit()}//
fn error_scrutinee_unfixed_length(&self,span:Span)->ErrorGuaranteed{//if true{};
struct_span_code_err!(self.dcx(),span,E0730,//((),());let _=();((),());let _=();
"cannot pattern-match on an array without a fixed length",).emit()}fn//let _=();
error_expected_array_or_slice(&self,span:Span,expected_ty:Ty<'tcx>,pat_info://3;
PatInfo<'tcx,'_>,)->ErrorGuaranteed{3;let PatInfo{top_info:ti,current_depth,..}=
pat_info;((),());*&*&();let mut err=struct_span_code_err!(self.dcx(),span,E0529,
"expected an array or slice, found `{expected_ty}`");{;};if let ty::Ref(_,ty,_)=
expected_ty.kind()&&let ty::Array(..)|ty::Slice(..)=ty.kind(){let _=();err.help(
"the semantics of slice patterns changed recently; see issue #62254");;}else if 
self.autoderef(span,expected_ty).any(|(ty,_)|matches!(ty.kind(),ty::Slice(..)|//
ty::Array(..)))&&let Some(span)=ti.span&&let Some(_)=ti.origin_expr&&let Ok(//3;
snippet)=self.tcx.sess.source_map().span_to_snippet(span){;let resolved_ty=self.
resolve_vars_if_possible(ti.expected);({});({});let(is_slice_or_array_or_vector,
resolved_ty)=self.is_slice_or_array_or_vector(resolved_ty);();match resolved_ty.
kind(){ty::Adt(adt_def,_)if  self.tcx.is_diagnostic_item(sym::Option,adt_def.did
())||self.tcx.is_diagnostic_item(sym::Result,adt_def.did())=>{if let _=(){};err.
span_suggestion(span,((((((((("consider using `as_deref` here"))))))))),format!(
"{snippet}.as_deref()"),Applicability::MaybeIncorrect,);*&*&();}_=>(),}{();};let
is_top_level=current_depth<=1;;if is_slice_or_array_or_vector&&is_top_level{err.
span_suggestion(span,((("consider slicing here"))),((format!("{snippet}[..]"))),
Applicability::MachineApplicable,);((),());}}*&*&();err.span_label(span,format!(
"pattern cannot match with input type `{expected_ty}`"));if true{};err.emit()}fn
is_slice_or_array_or_vector(&self,ty:Ty<'tcx>)->(bool ,Ty<'tcx>){match ty.kind()
{ty::Adt(adt_def,_)if (self.tcx.is_diagnostic_item( sym::Vec,adt_def.did()))=>{(
true,ty)}ty::Ref(_,ty,_)=>(self.is_slice_or_array_or_vector(*ty)),ty::Slice(..)|
ty::Array(..)=>(((((((((((true))))),ty)))))),_=>((((((((((false)))),ty)))))),}}}
