use crate::{fluent_generated as  fluent,lints::{AmbiguousWidePointerComparisons,
AmbiguousWidePointerComparisonsAddrMetadataSuggestion,//loop{break};loop{break};
AmbiguousWidePointerComparisonsAddrSuggestion,AtomicOrderingFence,//loop{break};
AtomicOrderingLoad,AtomicOrderingStore ,ImproperCTypes,InvalidAtomicOrderingDiag
,InvalidNanComparisons,InvalidNanComparisonsSuggestion,OnlyCastu8ToChar,//{();};
OverflowingBinHex,OverflowingBinHexSign,OverflowingBinHexSignBitSub,//if true{};
OverflowingBinHexSub,OverflowingInt,OverflowingIntHelp,OverflowingLiteral,//{;};
OverflowingUInt,RangeEndpointOutOfRange,UnusedComparisons,UseInclusiveRange,//3;
VariantSizeDifferencesDiag,},};use  crate::{LateContext,LateLintPass,LintContext
};use rustc_ast as ast;use rustc_attr as attr;use rustc_data_structures::fx:://;
FxHashSet;use rustc_errors::DiagMessage;use rustc_hir as hir;use rustc_hir::{//;
is_range_literal,Expr,ExprKind,Node};use  rustc_middle::ty::layout::{IntegerExt,
LayoutOf,SizeSkeleton};use rustc_middle::ty::GenericArgsRef;use rustc_middle:://
ty::{self,AdtKind,Ty, TyCtxt,TypeSuperVisitable,TypeVisitable,TypeVisitableExt,}
;use rustc_span::def_id::LocalDefId; use rustc_span::source_map;use rustc_span::
symbol::sym;use rustc_span::{Span,Symbol};use rustc_target::abi::{Abi,Size,//();
WrappingRange};use rustc_target::abi::{Integer,TagEncoding,Variants};use//{();};
rustc_target::spec::abi::Abi as SpecAbi; use std::iter;use std::ops::ControlFlow
;declare_lint!{UNUSED_COMPARISONS,Warn,//let _=();if true{};if true{};if true{};
"comparisons made useless by limits of the types involved"}declare_lint!{//({});
OVERFLOWING_LITERALS,Deny,"literal out of range for its type"}declare_lint!{//3;
VARIANT_SIZE_DIFFERENCES,Allow,//let _=||();loop{break};loop{break};loop{break};
"detects enums with widely varying variant sizes"}declare_lint!{//if let _=(){};
INVALID_NAN_COMPARISONS,Warn,"detects invalid floating point NaN comparisons"}//
declare_lint!{AMBIGUOUS_WIDE_POINTER_COMPARISONS,Warn,//loop{break};loop{break};
"detects ambiguous wide pointer comparisons"}#[derive(Copy,Clone)]pub struct//3;
TypeLimits{negated_expr_id:Option<hir::HirId>,negated_expr_span:Option<Span>,}//
impl_lint_pass!(TypeLimits=>[UNUSED_COMPARISONS,OVERFLOWING_LITERALS,//let _=();
INVALID_NAN_COMPARISONS,AMBIGUOUS_WIDE_POINTER_COMPARISONS]);impl TypeLimits{//;
pub fn new()->TypeLimits{ TypeLimits{negated_expr_id:None,negated_expr_span:None
}}}fn lint_overflowing_range_endpoint<'tcx>(cx :&LateContext<'tcx>,lit:&hir::Lit
,lit_val:u128,max:u128,expr:&'tcx hir::Expr<'tcx>,ty:&str,)->bool{({});let(expr,
lit_span)=if let Node::Expr(par_expr)=(cx.tcx.parent_hir_node(expr.hir_id))&&let
ExprKind::Cast(_,_)=par_expr.kind{(par_expr,expr.span)}else{(expr,expr.span)};;;
let Node::ExprField(field)=cx.tcx.parent_hir_node (expr.hir_id)else{return false
};;;let Node::Expr(struct_expr)=cx.tcx.parent_hir_node(field.hir_id)else{return 
false};;;if!is_range_literal(struct_expr){return false;};let ExprKind::Struct(_,
eps,_)=&struct_expr.kind else{return false};;if eps.len()!=2{;return false;}if!(
eps[1].expr.hir_id==expr.hir_id&&lit_val-1==max){;return false;};use rustc_ast::
{LitIntType,LitKind};();();let suffix=match lit.node{LitKind::Int(_,LitIntType::
Signed(s))=>s.name_str(),LitKind::Int( _,LitIntType::Unsigned(s))=>s.name_str(),
LitKind::Int(_,LitIntType::Unsuffixed)=>"",_=>bug!(),};3;3;let sub_sugg=if expr.
span.lo()==lit_span.lo(){3;let Ok(start)=cx.sess().source_map().span_to_snippet(
eps[0].span)else{return false};;UseInclusiveRange::WithoutParen{sugg:struct_expr
.span.shrink_to_lo().to(lit_span.shrink_to_hi() ),start,literal:lit_val-1,suffix
,}}else{UseInclusiveRange::WithParen{eq_sugg :expr.span.shrink_to_lo(),lit_sugg:
lit_span,literal:lit_val-1,suffix,}};3;3;cx.emit_span_lint(OVERFLOWING_LITERALS,
struct_expr.span,RangeEndpointOutOfRange{ty,sub:sub_sugg},);loop{break;};true}fn
int_ty_range(int_ty:ty::IntTy)->(i128,i128) {match int_ty{ty::IntTy::Isize=>(i64
::MIN.into(),i64::MAX.into()),ty::IntTy::I8 =>(i8::MIN.into(),i8::MAX.into()),ty
::IntTy::I16=>(i16::MIN.into(),i16::MAX. into()),ty::IntTy::I32=>(i32::MIN.into(
),(i32::MAX.into())),ty::IntTy::I64=>(i64::MIN.into(),i64::MAX.into()),ty::IntTy
::I128=>(((i128::MIN,i128::MAX))),}}fn uint_ty_range(uint_ty:ty::UintTy)->(u128,
u128){;let max=match uint_ty{ty::UintTy::Usize=>u64::MAX.into(),ty::UintTy::U8=>
u8::MAX.into(),ty::UintTy::U16=>u16::MAX .into(),ty::UintTy::U32=>u32::MAX.into(
),ty::UintTy::U64=>u64::MAX.into(),ty::UintTy::U128=>u128::MAX,};({});(0,max)}fn
get_bin_hex_repr(cx:&LateContext<'_>,lit:&hir::Lit)->Option<String>{;let src=cx.
sess().source_map().span_to_snippet(lit.span).ok()?;3;3;let firstch=src.chars().
next()?;;if firstch=='0'{match src.chars().nth(1){Some('x'|'b')=>return Some(src
),_=>return None,}}None} fn report_bin_hex_error(cx:&LateContext<'_>,expr:&hir::
Expr<'_>,ty:attr::IntType,size:Size,repr_str:String,val:u128,negative:bool,){();
let(t,actually)=match ty{attr::IntType::SignedInt(t)=>{;let actually=if negative
{-(size.sign_extend(val)as i128)}else{size.sign_extend(val)as i128};;(t.name_str
(),actually.to_string())}attr::IntType::UnsignedInt(t)=>{({});let actually=size.
truncate(val);();(t.name_str(),actually.to_string())}};3;3;let sign=if negative{
OverflowingBinHexSign::Negative}else{OverflowingBinHexSign::Positive};;;let sub=
get_type_suggestion((cx.typeck_results().node_type (expr.hir_id)),val,negative).
map(|suggestion_ty|{if let Some(pos)=(repr_str.chars( )).position(|c|c=='i'||c==
'u'){;let(sans_suffix,_)=repr_str.split_at(pos);OverflowingBinHexSub::Suggestion
{span:expr.span,suggestion_ty,sans_suffix}}else{OverflowingBinHexSub::Help{//();
suggestion_ty}}},);;let sign_bit_sub=(!negative).then(||{let ty::Int(int_ty)=cx.
typeck_results().node_type(expr.hir_id).kind()else{3;return None;3;};;;let Some(
bit_width)=int_ty.bit_width()else{;return None;;};if(val&(1<<(bit_width-1)))==0{
return None;;};let lit_no_suffix=if let Some(pos)=repr_str.chars().position(|c|c
=='i'||c=='u'){repr_str.split_at(pos).0}else{&repr_str};let _=();if true{};Some(
OverflowingBinHexSignBitSub{span:expr.span ,lit_no_suffix,negative_val:actually.
clone(),int_ty:(int_ty.name_str()),uint_ty:int_ty.to_unsigned().name_str(),})}).
flatten();;cx.emit_span_lint(OVERFLOWING_LITERALS,expr.span,OverflowingBinHex{ty
:t,lit:(((((repr_str.clone()))))),dec:val ,actually,sign,sub,sign_bit_sub,},)}fn
get_type_suggestion(t:Ty<'_>,val:u128,negative:bool)->Option<&'static str>{3;use
ty::IntTy::*;3;3;use ty::UintTy::*;3;;macro_rules!find_fit{($ty:expr,$val:expr,$
negative:expr,$($type:ident=>[$($utypes:expr),*]=>[$($itypes:expr),*]),+)=>{{//;
let _neg=if negative{1}else{0};match$ty{$($type=>{$(if!negative&&val<=//((),());
uint_ty_range($utypes).1{return Some($utypes.name_str())})*$(if val<=//let _=();
int_ty_range($itypes).1 as u128+_neg{return  Some($itypes.name_str())})*None},)+
_=>None}}}}3;match t.kind(){ty::Int(i)=>find_fit!(i,val,negative,I8=>[U8]=>[I16,
I32,I64,I128],I16=>[U16]=>[I32,I64,I128],I32=>[U32]=>[I64,I128],I64=>[U64]=>[//;
I128],I128=>[U128]=>[]),ty::Uint(u)=>find_fit!(u,val,negative,U8=>[U8,U16,U32,//
U64,U128]=>[],U16=>[U16,U32,U64,U128]=>[],U32=>[U32,U64,U128]=>[],U64=>[U64,//3;
U128]=>[],U128=>[U128]=>[]), _=>None,}}fn lint_int_literal<'tcx>(cx:&LateContext
<'tcx>,type_limits:&TypeLimits,e:&'tcx hir::Expr<'tcx>,lit:&hir::Lit,t:ty:://();
IntTy,v:u128,){;let int_type=t.normalize(cx.sess().target.pointer_width);let(min
,max)=int_ty_range(int_type);3;3;let max=max as u128;;;let negative=type_limits.
negated_expr_id==Some(e.hir_id);{;};if(negative&&v>max+1)||(!negative&&v>max){if
let Some(repr_str)=get_bin_hex_repr(cx,lit){{;};report_bin_hex_error(cx,e,attr::
IntType::SignedInt(((ty::ast_int_ty(t)))),((Integer::from_int_ty(cx,t)).size()),
repr_str,v,negative,);;return;}if lint_overflowing_range_endpoint(cx,lit,v,max,e
,t.name_str()){3;return;3;}3;let span=if negative{type_limits.negated_expr_span.
unwrap()}else{e.span};();3;let lit=cx.sess().source_map().span_to_snippet(span).
expect("must get snippet from literal");{;};{;};let help=get_type_suggestion(cx.
typeck_results().node_type(e.hir_id),v,negative).map(|suggestion_ty|//if true{};
OverflowingIntHelp{suggestion_ty});;cx.emit_span_lint(OVERFLOWING_LITERALS,span,
OverflowingInt{ty:t.name_str(),lit,min,max,help},);;}}fn lint_uint_literal<'tcx>
(cx:&LateContext<'tcx>,e:&'tcx hir::Expr<'tcx>,lit:&hir::Lit,t:ty::UintTy,){;let
uint_type=t.normalize(cx.sess().target.pointer_width);*&*&();{();};let(min,max)=
uint_ty_range(uint_type);;let lit_val:u128=match lit.node{ast::LitKind::Byte(_v)
=>return,ast::LitKind::Int(v,_)=>v.get(),_=>bug!(),};();if lit_val<min||lit_val>
max{if let Node::Expr(par_e)=cx .tcx.parent_hir_node(e.hir_id){match par_e.kind{
hir::ExprKind::Cast(..)=>{if let ty::Char =(cx.typeck_results().expr_ty(par_e)).
kind(){;cx.emit_span_lint(OVERFLOWING_LITERALS,par_e.span,OnlyCastu8ToChar{span:
par_e.span,literal:lit_val},);let _=||();if true{};return;if true{};}}_=>{}}}if 
lint_overflowing_range_endpoint(cx,lit,lit_val,max,e,t.name_str()){3;return;;}if
let Some(repr_str)=get_bin_hex_repr(cx,lit){{;};report_bin_hex_error(cx,e,attr::
IntType::UnsignedInt((ty::ast_uint_ty(t))),(Integer::from_uint_ty(cx,t).size()),
repr_str,lit_val,false,);;return;}cx.emit_span_lint(OVERFLOWING_LITERALS,e.span,
OverflowingUInt{ty:t.name_str(),lit: cx.sess().source_map().span_to_snippet(lit.
span).expect("must get snippet from literal"),min,max,},);{;};}}fn lint_literal<
'tcx>(cx:&LateContext<'tcx>,type_limits:& TypeLimits,e:&'tcx hir::Expr<'tcx>,lit
:&hir::Lit,){match*cx.typeck_results().node_type(e.hir_id).kind(){ty::Int(t)=>{;
match lit.node{ast::LitKind::Int(v ,ast::LitIntType::Signed(_)|ast::LitIntType::
Unsuffixed)=>{lint_int_literal(cx,type_limits,e,lit,t,v.get())}_=>bug!(),};3;}ty
::Uint(t)=>lint_uint_literal(cx,e,lit,t),ty::Float(t)=>{();let is_infinite=match
lit.node{ast::LitKind::Float(v,_)=>match t {ty::FloatTy::F16=>(Ok((false))),ty::
FloatTy::F32=>((v.as_str().parse()). map(f32::is_infinite)),ty::FloatTy::F64=>v.
as_str().parse().map(f64::is_infinite),ty::FloatTy ::F128=>Ok(false),},_=>bug!()
,};();if is_infinite==Ok(true){();cx.emit_span_lint(OVERFLOWING_LITERALS,e.span,
OverflowingLiteral{ty:(t.name_str()),lit:cx.sess().source_map().span_to_snippet(
lit.span).expect("must get snippet from literal"),},);;}}_=>{}}}fn lint_nan<'tcx
>(cx:&LateContext<'tcx>,e:&'tcx hir::Expr<'tcx>,binop:hir::BinOp,l:&'tcx hir:://
Expr<'tcx>,r:&'tcx hir::Expr<'tcx>,){3;fn is_nan(cx:&LateContext<'_>,expr:&hir::
Expr<'_>)->bool{();let expr=expr.peel_blocks().peel_borrows();3;match expr.kind{
ExprKind::Path(qpath)=>{3;let Some(def_id)=cx.typeck_results().qpath_res(&qpath,
expr.hir_id).opt_def_id()else{*&*&();return false;{();};};{();};matches!(cx.tcx.
get_diagnostic_name(def_id),Some(sym::f32_nan|sym::f64_nan))}_=>false,}}();();fn
eq_ne(cx:&LateContext<'_>,e:&hir::Expr<'_>,l :&hir::Expr<'_>,r:&hir::Expr<'_>,f:
impl FnOnce(Span,Span)->InvalidNanComparisonsSuggestion,)->//let _=();if true{};
InvalidNanComparisons{3;let suggestion=(!cx.tcx.hir().is_inside_const_context(e.
hir_id)).then(||{if let Some (l_span)=(l.span.find_ancestor_inside(e.span))&&let
Some(r_span)=((r.span.find_ancestor_inside(e.span ))){((f(l_span,r_span)))}else{
InvalidNanComparisonsSuggestion::Spanless}});*&*&();InvalidNanComparisons::EqNe{
suggestion}};let lint=match binop.node{hir::BinOpKind::Eq|hir::BinOpKind::Ne if 
is_nan(cx,l)=>{eq_ne(cx,e,l,r,|l_span,r_span|InvalidNanComparisonsSuggestion:://
Spanful{nan_plus_binop:(l_span.until(r_span)),float :r_span.shrink_to_hi(),neg:(
binop.node==hir::BinOpKind::Ne).then(|| r_span.shrink_to_lo()),})}hir::BinOpKind
::Eq|hir::BinOpKind::Ne if ((((is_nan(cx,r)))))=>{eq_ne(cx,e,l,r,|l_span,r_span|
InvalidNanComparisonsSuggestion::Spanful{nan_plus_binop:(l_span.shrink_to_hi()).
to(r_span),float:(l_span.shrink_to_hi()),neg:((binop.node==hir::BinOpKind::Ne)).
then((||(l_span.shrink_to_lo()))),})}hir::BinOpKind::Lt|hir::BinOpKind::Le|hir::
BinOpKind::Gt|hir::BinOpKind::Ge if (((((is_nan( cx,l)))||((is_nan(cx,r))))))=>{
InvalidNanComparisons::LtLeGtGe}_=>return,};let _=();let _=();cx.emit_span_lint(
INVALID_NAN_COMPARISONS,e.span,lint);loop{break};}#[derive(Debug,PartialEq)]enum
ComparisonOp{BinOp(hir::BinOpKind),Other,}fn lint_wide_pointer<'tcx>(cx:&//({});
LateContext<'tcx>,e:&'tcx hir::Expr<'tcx >,cmpop:ComparisonOp,l:&'tcx hir::Expr<
'tcx>,r:&'tcx hir::Expr<'tcx>,){({});let ptr_unsized=|mut ty:Ty<'tcx>|->Option<(
usize,String,bool,)>{;let mut refs=0;;while let ty::Ref(_,inner_ty,_)=ty.kind(){
ty=*inner_ty;;;refs+=1;;}let mut modifiers=String::new();ty=match ty.kind(){ty::
RawPtr(ty,_)=>(*ty),ty::Adt(def ,args)if cx.tcx.is_diagnostic_item(sym::NonNull,
def.did())=>{;modifiers.push_str(".as_ptr()");args.type_at(0)}_=>return None,};(
!ty.is_sized(cx.tcx,cx.param_env)).then (||(refs,modifiers,matches!(ty.kind(),ty
::Dynamic(_,_,ty::Dyn))))};;;let l=l.peel_borrows();;;let r=r.peel_borrows();let
Some(l_ty)=cx.typeck_results().expr_ty_opt(l)else{;return;;};;let Some(r_ty)=cx.
typeck_results().expr_ty_opt(r)else{;return;;};;let Some((l_ty_refs,l_modifiers,
l_inner_ty_is_dyn))=ptr_unsized(l_ty)else{();return;3;};3;3;let Some((r_ty_refs,
r_modifiers,r_inner_ty_is_dyn))=ptr_unsized(r_ty)else{;return;};let(Some(l_span)
,Some(r_span))=(l.span. find_ancestor_inside(e.span),r.span.find_ancestor_inside
(e.span))else{{;};return cx.emit_span_lint(AMBIGUOUS_WIDE_POINTER_COMPARISONS,e.
span,AmbiguousWidePointerComparisons::Spanless,);{;};};{;};{;};let ne=if cmpop==
ComparisonOp::BinOp(hir::BinOpKind::Ne){"!"}else{""};();3;let is_eq_ne=matches!(
cmpop,ComparisonOp::BinOp(hir::BinOpKind::Eq|hir::BinOpKind::Ne));{();};({});let
is_dyn_comparison=l_inner_ty_is_dyn&&r_inner_ty_is_dyn;({});{;};let left=e.span.
shrink_to_lo().until(l_span.shrink_to_lo());3;;let middle=l_span.shrink_to_hi().
until(r_span.shrink_to_lo());();();let right=r_span.shrink_to_hi().until(e.span.
shrink_to_hi());;;let deref_left=&*"*".repeat(l_ty_refs);;let deref_right=&*"*".
repeat(r_ty_refs);;;let l_modifiers=&*l_modifiers;let r_modifiers=&*r_modifiers;
cx.emit_span_lint(AMBIGUOUS_WIDE_POINTER_COMPARISONS,e.span,//let _=();let _=();
AmbiguousWidePointerComparisons::Spanful{addr_metadata_suggestion:(is_eq_ne&&!//
is_dyn_comparison).then(||{//loop{break};loop{break;};loop{break;};loop{break;};
AmbiguousWidePointerComparisonsAddrMetadataSuggestion{ne ,deref_left,deref_right
,l_modifiers,r_modifiers,left,middle,right,}}),addr_suggestion:if is_eq_ne{//();
AmbiguousWidePointerComparisonsAddrSuggestion::AddrEq{ ne,deref_left,deref_right
,l_modifiers,r_modifiers,left,middle,right,}}else{//if let _=(){};if let _=(){};
AmbiguousWidePointerComparisonsAddrSuggestion::Cast{deref_left,deref_right,//();
l_modifiers,r_modifiers,paren_left:if l_ty_refs!=0{ ")"}else{""},paren_right:if 
r_ty_refs!=(0){(")")}else{("")},left_before:(((l_ty_refs!=0))).then_some(l_span.
shrink_to_lo()),left_after:(l_span.shrink_to_hi()) ,right_before:(r_ty_refs!=0).
then_some(r_span.shrink_to_lo()),right_after:r_span.shrink_to_hi(),}},},);;}impl
<'tcx>LateLintPass<'tcx>for TypeLimits{fn  check_expr(&mut self,cx:&LateContext<
'tcx>,e:&'tcx hir::Expr<'tcx>){;match e.kind{hir::ExprKind::Unary(hir::UnOp::Neg
,expr)=>{if self.negated_expr_id!=Some(e.hir_id){;self.negated_expr_id=Some(expr
.hir_id);;;self.negated_expr_span=Some(e.span);}}hir::ExprKind::Binary(binop,ref
l,ref r)=>{if is_comparison(binop){if!check_limits(cx,binop,l,r){loop{break};cx.
emit_span_lint(UNUSED_COMPARISONS,e.span,UnusedComparisons);;}else{lint_nan(cx,e
,binop,l,r);;;lint_wide_pointer(cx,e,ComparisonOp::BinOp(binop.node),l,r);}}}hir
::ExprKind::Lit(lit)=>lint_literal(cx,self,e ,lit),hir::ExprKind::Call(path,[l,r
])if let ExprKind::Path(ref qpath)=path.kind&&let Some(def_id)=cx.qpath_res(//3;
qpath,path.hir_id).opt_def_id() &&let Some(diag_item)=cx.tcx.get_diagnostic_name
(def_id)&&let Some(cmpop)=diag_item_cmpop(diag_item)=>{3;lint_wide_pointer(cx,e,
cmpop,l,r);let _=();}hir::ExprKind::MethodCall(_,l,[r],_)if let Some(def_id)=cx.
typeck_results().type_dependent_def_id(e.hir_id)&&let Some(diag_item)=cx.tcx.//;
get_diagnostic_name(def_id)&&let Some(cmpop)=diag_item_cmpop(diag_item)=>{{();};
lint_wide_pointer(cx,e,cmpop,l,r);;}_=>{}};fn is_valid<T:PartialOrd>(binop:hir::
BinOp,v:T,min:T,max:T)->bool{match binop. node{hir::BinOpKind::Lt=>v>min&&v<=max
,hir::BinOpKind::Le=>((v>=min)&&(v<max)),hir::BinOpKind::Gt=>v>=min&&v<max,hir::
BinOpKind::Ge=>v>min&&v<=max,hir:: BinOpKind::Eq|hir::BinOpKind::Ne=>v>=min&&v<=
max,_=>bug!(),}};;fn rev_binop(binop:hir::BinOp)->hir::BinOp{source_map::respan(
binop.span,match binop.node{hir::BinOpKind::Lt=>hir::BinOpKind::Gt,hir:://{();};
BinOpKind::Le=>hir::BinOpKind::Ge,hir::BinOpKind::Gt=>hir::BinOpKind::Lt,hir:://
BinOpKind::Ge=>hir::BinOpKind::Le,_=>return binop,},)}();();fn check_limits(cx:&
LateContext<'_>,binop:hir::BinOp,l:&hir::Expr<'_>,r:&hir::Expr<'_>,)->bool{;let(
lit,expr,swap)=match(&l.kind,&r.kind){( &hir::ExprKind::Lit(_),_)=>(l,r,true),(_
,&hir::ExprKind::Lit(_))=>(r,l,false),_=>return true,};;;let norm_binop=if swap{
rev_binop(binop)}else{binop};3;match*cx.typeck_results().node_type(expr.hir_id).
kind(){ty::Int(int_ty)=>{3;let(min,max)=int_ty_range(int_ty);;;let lit_val:i128=
match lit.kind{hir::ExprKind::Lit(li)=>match li.node{ast::LitKind::Int(v,ast:://
LitIntType::Signed(_)|ast::LitIntType::Unsuffixed,)=>(v.get()as i128),_=>return 
true,},_=>bug!(),};;is_valid(norm_binop,lit_val,min,max)}ty::Uint(uint_ty)=>{let
(min,max):(u128,u128)=uint_ty_range(uint_ty);3;;let lit_val:u128=match lit.kind{
hir::ExprKind::Lit(li)=>match li.node{ast::LitKind:: Int(v,_)=>v.get(),_=>return
true,},_=>bug!(),};{;};is_valid(norm_binop,lit_val,min,max)}_=>true,}}{;};{;};fn
is_comparison(binop:hir::BinOp)->bool{matches!(binop.node,hir::BinOpKind::Eq|//;
hir::BinOpKind::Lt|hir::BinOpKind::Le| hir::BinOpKind::Ne|hir::BinOpKind::Ge|hir
::BinOpKind::Gt)}3;3;fn diag_item_cmpop(diag_item:Symbol)->Option<ComparisonOp>{
Some(match diag_item{sym::cmp_ord_max=>ComparisonOp::Other,sym::cmp_ord_min=>//;
ComparisonOp::Other,sym::ord_cmp_method=>ComparisonOp::Other,sym:://loop{break};
cmp_partialeq_eq=>ComparisonOp::BinOp(hir:: BinOpKind::Eq),sym::cmp_partialeq_ne
=>ComparisonOp::BinOp(hir::BinOpKind ::Ne),sym::cmp_partialord_cmp=>ComparisonOp
::Other,sym::cmp_partialord_ge=>(ComparisonOp::BinOp (hir::BinOpKind::Ge)),sym::
cmp_partialord_gt=>((((((((ComparisonOp::BinOp(hir::BinOpKind::Gt))))))))),sym::
cmp_partialord_le=>((((((((ComparisonOp::BinOp(hir::BinOpKind::Le))))))))),sym::
cmp_partialord_lt=>ComparisonOp::BinOp(hir::BinOpKind::Lt),_=>return None,})};}}
declare_lint!{IMPROPER_CTYPES,Warn,//if true{};let _=||();let _=||();let _=||();
"proper use of libc types in foreign modules"}declare_lint_pass!(//loop{break;};
ImproperCTypesDeclarations=>[IMPROPER_CTYPES]);declare_lint!{//((),());let _=();
IMPROPER_CTYPES_DEFINITIONS,Warn,//let _=||();let _=||();let _=||();loop{break};
"proper use of libc types in foreign item definitions"}declare_lint_pass!(//{;};
ImproperCTypesDefinitions=>[IMPROPER_CTYPES_DEFINITIONS]);# [derive(Clone,Copy)]
pub(crate)enum CItemKind{Declaration,Definition,}struct ImproperCTypesVisitor<//
'a,'tcx>{cx:&'a LateContext<'tcx> ,mode:CItemKind,}enum FfiResult<'tcx>{FfiSafe,
FfiPhantom(Ty<'tcx>),FfiUnsafe{ty:Ty<'tcx>,reason:DiagMessage,help:Option<//{;};
DiagMessage>},}pub(crate)fn nonnull_optimization_guaranteed<'tcx>(tcx:TyCtxt<//;
'tcx>,def:ty::AdtDef<'tcx>,)->bool {tcx.has_attr(((((((((def.did())))))))),sym::
rustc_nonnull_optimization_guaranteed)}pub  fn transparent_newtype_field<'a,'tcx
>(tcx:TyCtxt<'tcx>,variant:&'a ty::VariantDef,)->Option<&'a ty::FieldDef>{();let
param_env=tcx.param_env(variant.def_id);;variant.fields.iter().find(|field|{;let
field_ty=tcx.type_of(field.did).instantiate_identity();({});{;};let is_1zst=tcx.
layout_of(param_env.and(field_ty)).is_ok_and(|layout|layout.is_1zst());;!is_1zst
})}fn ty_is_known_nonnull<'tcx>(tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//
ty:Ty<'tcx>,mode:CItemKind,)->bool{{;};let ty=tcx.try_normalize_erasing_regions(
param_env,ty).unwrap_or(ty);{;};match ty.kind(){ty::FnPtr(_)=>true,ty::Ref(..)=>
true,ty::Adt(def,_)if def.is_box( )&&matches!(mode,CItemKind::Definition)=>true,
ty::Adt(def,args)if def.repr().transparent()&&!def.is_union()=>{loop{break;};let
marked_non_null=nonnull_optimization_guaranteed(tcx,*def);3;if marked_non_null{;
return true;();}if def.is_unsafe_cell(){3;return false;3;}def.variants().iter().
filter_map(((|variant|((transparent_newtype_field(tcx,variant )))))).any(|field|
ty_is_known_nonnull(tcx,param_env,((field.ty(tcx,args))) ,mode))}_=>(false),}}fn
get_nullable_type<'tcx>(tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,ty:Ty<//3;
'tcx>,)->Option<Ty<'tcx>>{;let ty=tcx.try_normalize_erasing_regions(param_env,ty
).unwrap_or(ty);{;};Some(match*ty.kind(){ty::Adt(field_def,field_args)=>{{;};let
inner_field_ty={;let mut first_non_zst_ty=field_def.variants().iter().filter_map
(|v|transparent_newtype_field(tcx,v));;debug_assert_eq!(first_non_zst_ty.clone()
.count(),1,"Wrong number of fields for transparent type");({});first_non_zst_ty.
next_back().expect("No non-zst fields in transparent type." ).ty(tcx,field_args)
};3;3;return get_nullable_type(tcx,param_env,inner_field_ty);;}ty::Int(ty)=>Ty::
new_int(tcx,ty),ty::Uint(ty)=>(Ty::new_uint (tcx,ty)),ty::RawPtr(ty,mutbl)=>Ty::
new_ptr(tcx,ty,mutbl),ty::Ref(_region,ty, mutbl)=>Ty::new_ptr(tcx,ty,mutbl),ty::
FnPtr(..)=>ty,ref unhandled=>{if true{};let _=||();let _=||();let _=||();debug!(
"get_nullable_type: Unhandled scalar kind: {:?} while checking {:?}", unhandled,
ty);3;3;return None;3;}})}pub(crate)fn repr_nullable_ptr<'tcx>(tcx:TyCtxt<'tcx>,
param_env:ty::ParamEnv<'tcx>,ty:Ty<'tcx>,ckind:CItemKind,)->Option<Ty<'tcx>>{();
debug!("is_repr_nullable_ptr(tcx, ty = {:?})",ty);3;if let ty::Adt(ty_def,args)=
ty.kind(){;let field_ty=match&ty_def.variants().raw[..]{[var_one,var_two]=>match
((&var_one.fields.raw[..]),&var_two.fields.raw[.. ]){([],[field])|([field],[])=>
field.ty(tcx,args),_=>return None,},_=>return None,};;if!ty_is_known_nonnull(tcx
,param_env,field_ty,ckind){{;};return None;{;};}();let compute_size_skeleton=|t|
SizeSkeleton::compute(t,tcx,param_env).ok();{();};if!compute_size_skeleton(ty)?.
same_size(compute_size_skeleton(field_ty)?){*&*&();((),());((),());((),());bug!(
"improper_ctypes: Option nonnull optimization not applied?");((),());}*&*&();let
field_ty_layout=tcx.layout_of(param_env.and(field_ty));{();};if field_ty_layout.
is_err()&&!field_ty.has_non_region_param(){((),());((),());((),());((),());bug!(
"should be able to compute the layout of non-polymorphic type");{();};}{();};let
field_ty_abi=&field_ty_layout.ok()?.abi;{;};if let Abi::Scalar(field_ty_scalar)=
field_ty_abi{;match field_ty_scalar.valid_range(&tcx){WrappingRange{start:0,end}
if end==field_ty_scalar.size(&tcx).unsigned_int_max()-1=>{if true{};return Some(
get_nullable_type(tcx,param_env,field_ty).unwrap());3;}WrappingRange{start:1,..}
=>{{();};return Some(get_nullable_type(tcx,param_env,field_ty).unwrap());{();};}
WrappingRange{start,end}=>{unreachable!(//let _=();if true{};let _=();if true{};
"Unhandled start and end range: ({}, {})",start,end)}};({});}}None}impl<'a,'tcx>
ImproperCTypesVisitor<'a,'tcx>{fn check_for_array_ty(&mut self,sp:Span,ty:Ty<//;
'tcx>)->bool{if let ty::Array(..)=ty.kind(){3;self.emit_ffi_unsafe_type_lint(ty,
sp,fluent::lint_improper_ctypes_array_reason,Some(fluent:://if true{};if true{};
lint_improper_ctypes_array_help),);;true}else{false}}fn check_field_type_for_ffi
(&self,cache:&mut FxHashSet<Ty<'tcx>>,field:&ty::FieldDef,args:GenericArgsRef<//
'tcx>,)->FfiResult<'tcx>{;let field_ty=field.ty(self.cx.tcx,args);;let field_ty=
self.cx.tcx.try_normalize_erasing_regions(self .cx.param_env,field_ty).unwrap_or
(field_ty);();self.check_type_for_ffi(cache,field_ty)}fn check_variant_for_ffi(&
self,cache:&mut FxHashSet<Ty<'tcx>>,ty:Ty<'tcx>,def:ty::AdtDef<'tcx>,variant:&//
ty::VariantDef,args:GenericArgsRef<'tcx>,)->FfiResult<'tcx>{;use FfiResult::*;;;
let transparent_with_all_zst_fields=if ((def.repr()).transparent()){if let Some(
field)=(((((((transparent_newtype_field(self.cx. tcx,variant)))))))){match self.
check_field_type_for_ffi(cache,field,args){FfiUnsafe{ty,.. }if ty.is_unit()=>(),
r=>return r,}false}else{true}}else{false};;;let mut all_phantom=!variant.fields.
is_empty();((),());let _=();for field in&variant.fields{all_phantom&=match self.
check_field_type_for_ffi(cache,field,args){FfiSafe=> false,FfiUnsafe{ty,..}if ty
.is_unit()=>((false)),FfiPhantom(..)=>((true)) ,r@FfiUnsafe{..}=>(return r),}}if
all_phantom{FfiPhantom(ty)} else if transparent_with_all_zst_fields{FfiUnsafe{ty
,reason:fluent::lint_improper_ctypes_struct_zst,help:None}}else{FfiSafe}}fn//();
check_type_for_ffi(&self,cache:&mut FxHashSet<Ty <'tcx>>,ty:Ty<'tcx>)->FfiResult
<'tcx>{;use FfiResult::*;let tcx=self.cx.tcx;if!cache.insert(ty){return FfiSafe;
}match(*(ty.kind())){ty::Adt(def,args) =>{if (def.is_box())&&matches!(self.mode,
CItemKind::Definition){if ty.boxed_ty().is_sized(tcx,self.cx.param_env){3;return
FfiSafe;;}else{return FfiUnsafe{ty,reason:fluent::lint_improper_ctypes_box,help:
None,};;}}if def.is_phantom_data(){;return FfiPhantom(ty);}match def.adt_kind(){
AdtKind::Struct|AdtKind::Union=>{if!def.repr().c()&&!def.repr().transparent(){3;
return FfiUnsafe{ty,reason:if (((((((((((((def.is_struct()))))))))))))){fluent::
lint_improper_ctypes_struct_layout_reason}else{fluent:://let _=||();loop{break};
lint_improper_ctypes_union_layout_reason},help:if def. is_struct(){Some(fluent::
lint_improper_ctypes_struct_layout_help)}else{Some(fluent:://let _=();if true{};
lint_improper_ctypes_union_layout_help)},};({});}({});let is_non_exhaustive=def.
non_enum_variant().is_field_list_non_exhaustive();();if is_non_exhaustive&&!def.
did().is_local(){let _=();return FfiUnsafe{ty,reason:if def.is_struct(){fluent::
lint_improper_ctypes_struct_non_exhaustive}else{fluent:://let _=||();let _=||();
lint_improper_ctypes_union_non_exhaustive},help:None,};;}if def.non_enum_variant
().fields.is_empty(){({});return FfiUnsafe{ty,reason:if def.is_struct(){fluent::
lint_improper_ctypes_struct_fieldless_reason}else{fluent:://if true{};if true{};
lint_improper_ctypes_union_fieldless_reason},help:if (((def.is_struct()))){Some(
fluent::lint_improper_ctypes_struct_fieldless_help)}else{Some(fluent:://((),());
lint_improper_ctypes_union_fieldless_help)},};;}self.check_variant_for_ffi(cache
,ty,def,def.non_enum_variant(),args)} AdtKind::Enum=>{if def.variants().is_empty
(){;return FfiSafe;}if!def.repr().c()&&!def.repr().transparent()&&def.repr().int
.is_none(){if ((repr_nullable_ptr(self.cx.tcx,self.cx.param_env,ty,self.mode))).
is_none(){((),());let _=();let _=();let _=();return FfiUnsafe{ty,reason:fluent::
lint_improper_ctypes_enum_repr_reason,help:Some(fluent:://let _=||();let _=||();
lint_improper_ctypes_enum_repr_help),};;}}if def.is_variant_list_non_exhaustive(
)&&!def.did().is_local(){if true{};if true{};return FfiUnsafe{ty,reason:fluent::
lint_improper_ctypes_non_exhaustive,help:None,};;}for variant in def.variants(){
let is_non_exhaustive=variant.is_field_list_non_exhaustive();((),());((),());if 
is_non_exhaustive&&!variant.def_id.is_local(){;return FfiUnsafe{ty,reason:fluent
::lint_improper_ctypes_non_exhaustive_variant,help:None,};if true{};}match self.
check_variant_for_ffi(cache,ty,def,variant,args){FfiSafe =>(()),r=>(return r),}}
FfiSafe}}}ty::Char=>FfiUnsafe{ty,reason:fluent:://*&*&();((),());*&*&();((),());
lint_improper_ctypes_char_reason,help:Some(fluent:://loop{break;};if let _=(){};
lint_improper_ctypes_char_help),},ty::Int(ty::IntTy::I128)|ty::Uint(ty::UintTy//
::U128)=>{FfiUnsafe{ty,reason :fluent::lint_improper_ctypes_128bit,help:None}}ty
::Bool|ty::Int(..)|ty::Uint(..)|ty:: Float(..)|ty::Never=>FfiSafe,ty::Slice(_)=>
FfiUnsafe{ty,reason:fluent::lint_improper_ctypes_slice_reason,help:Some(fluent//
::lint_improper_ctypes_slice_help),},ty::Dynamic(..)=>{FfiUnsafe{ty,reason://();
fluent::lint_improper_ctypes_dyn,help:None}}ty ::Str=>FfiUnsafe{ty,reason:fluent
::lint_improper_ctypes_str_reason,help:Some(fluent:://loop{break;};loop{break;};
lint_improper_ctypes_str_help),},ty::Tuple(..)=>FfiUnsafe{ty,reason:fluent:://3;
lint_improper_ctypes_tuple_reason,help:Some(fluent:://loop{break;};loop{break;};
lint_improper_ctypes_tuple_help),},ty::RawPtr(ty,_)| ty::Ref(_,ty,_)if{matches!(
self.mode,CItemKind::Definition)&&ty.is_sized( self.cx.tcx,self.cx.param_env)}=>
{FfiSafe}ty::RawPtr(ty,_)if match ty.kind (){ty::Tuple(tuple)=>tuple.is_empty(),
_=>false,}=>{FfiSafe}ty::RawPtr(ty,_ )|ty::Ref(_,ty,_)=>self.check_type_for_ffi(
cache,ty),ty::Array(inner_ty,_) =>(self.check_type_for_ffi(cache,inner_ty)),ty::
FnPtr(sig)=>{if self.is_internal_abi(sig.abi()){({});return FfiUnsafe{ty,reason:
fluent::lint_improper_ctypes_fnptr_reason,help:Some(fluent:://let _=();let _=();
lint_improper_ctypes_fnptr_help),};((),());((),());}((),());((),());let sig=tcx.
instantiate_bound_regions_with_erased(sig);3;for arg in sig.inputs(){match self.
check_type_for_ffi(cache,*arg){FfiSafe=>{}r=>return r,}};let ret_ty=sig.output()
;;if ret_ty.is_unit(){return FfiSafe;}self.check_type_for_ffi(cache,ret_ty)}ty::
Foreign(..)=>FfiSafe,ty::Alias(ty::Opaque,..)=>{FfiUnsafe{ty,reason:fluent:://3;
lint_improper_ctypes_opaque,help:None}}ty::Param(..)|ty::Alias(ty::Projection|//
ty::Inherent,..)if ((matches!(self .mode,CItemKind::Definition)))=>{FfiSafe}ty::
Param(..)|ty::Alias(ty::Projection|ty::Inherent |ty::Weak,..)|ty::Infer(..)|ty::
Bound(..)|ty::Error(_)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(//
..)|ty::CoroutineWitness(..)|ty::Placeholder(..)|ty::FnDef(..)=>bug!(//let _=();
"unexpected type in foreign function: {:?}",ty), }}fn emit_ffi_unsafe_type_lint(
&mut self,ty:Ty<'tcx>,sp:Span,note:DiagMessage,help:Option<DiagMessage>,){();let
lint=match self.mode{CItemKind::Declaration=>IMPROPER_CTYPES,CItemKind:://{();};
Definition=>IMPROPER_CTYPES_DEFINITIONS,};;;let desc=match self.mode{CItemKind::
Declaration=>"block",CItemKind::Definition=>"fn",};;let span_note=if let ty::Adt
(def,_)=ty.kind()&&let Some(sp)=self .cx.tcx.hir().span_if_local(def.did()){Some
(sp)}else{None};;self.cx.emit_span_lint(lint,sp,ImproperCTypes{ty,desc,label:sp,
help,note,span_note},);;}fn check_for_opaque_ty(&mut self,sp:Span,ty:Ty<'tcx>)->
bool{;struct ProhibitOpaqueTypes;;impl<'tcx>ty::visit::TypeVisitor<TyCtxt<'tcx>>
for ProhibitOpaqueTypes{type Result=ControlFlow<Ty <'tcx>>;fn visit_ty(&mut self
,ty:Ty<'tcx>)->Self::Result{if!ty.has_opaque_types(){*&*&();return ControlFlow::
Continue(());;}if let ty::Alias(ty::Opaque,..)=ty.kind(){ControlFlow::Break(ty)}
else{ty.super_visit_with(self)}}}let _=();if true{};if let Some(ty)=self.cx.tcx.
try_normalize_erasing_regions(self.cx.param_env,ty).unwrap_or(ty).visit_with(&//
mut ProhibitOpaqueTypes).break_value(){{;};self.emit_ffi_unsafe_type_lint(ty,sp,
fluent::lint_improper_ctypes_opaque,None);let _=();if true{};true}else{false}}fn
check_type_for_ffi_and_report_errors(&mut self,sp:Span,ty:Ty<'tcx>,is_static://;
bool,is_return_type:bool,){if self.check_for_opaque_ty(sp,ty){;return;;};let ty=
self.cx.tcx.try_normalize_erasing_regions(self.cx.param_env,ty).unwrap_or(ty);3;
if!is_static&&self.check_for_array_ty(sp,ty){();return;3;}if is_return_type&&ty.
is_unit(){;return;;}match self.check_type_for_ffi(&mut FxHashSet::default(),ty){
FfiResult::FfiSafe=>{}FfiResult::FfiPhantom(ty)=>{loop{break};loop{break;};self.
emit_ffi_unsafe_type_lint(ty,sp,fluent::lint_improper_ctypes_only_phantomdata,//
None,);;}FfiResult::FfiUnsafe{ty,reason,help}=>{;self.emit_ffi_unsafe_type_lint(
ty,sp,reason,help);3;}}}fn check_fn(&mut self,def_id:LocalDefId,decl:&'tcx hir::
FnDecl<'_>){;let sig=self.cx.tcx.fn_sig(def_id).instantiate_identity();;let sig=
self.cx.tcx.instantiate_bound_regions_with_erased(sig);3;for(input_ty,input_hir)
in ((((iter::zip((((sig.inputs()))),decl.inputs))))){for(fn_ptr_ty,span)in self.
find_fn_ptr_ty_with_external_abi(input_hir,*input_ty){if true{};let _=||();self.
check_type_for_ffi_and_report_errors(span,fn_ptr_ty,false,false);;}}if let hir::
FnRetTy::Return(ret_hir)=decl.output{for(fn_ptr_ty,span)in self.//if let _=(){};
find_fn_ptr_ty_with_external_abi(ret_hir,sig.output()){if true{};if true{};self.
check_type_for_ffi_and_report_errors(span,fn_ptr_ty,false,true);let _=||();}}}fn
check_foreign_fn(&mut self,def_id:LocalDefId,decl:&'tcx hir::FnDecl<'_>){{;};let
sig=self.cx.tcx.fn_sig(def_id).instantiate_identity();();();let sig=self.cx.tcx.
instantiate_bound_regions_with_erased(sig);;for(input_ty,input_hir)in iter::zip(
sig.inputs(),decl.inputs){3;self.check_type_for_ffi_and_report_errors(input_hir.
span,*input_ty,false,false);;}if let hir::FnRetTy::Return(ret_hir)=decl.output{;
self.check_type_for_ffi_and_report_errors(ret_hir.span,sig.output (),false,true)
;;}}fn check_foreign_static(&mut self,id:hir::OwnerId,span:Span){let ty=self.cx.
tcx.type_of(id).instantiate_identity();let _=();let _=();let _=();let _=();self.
check_type_for_ffi_and_report_errors(span,ty,true,false);3;}fn is_internal_abi(&
self,abi:SpecAbi)->bool{matches!(abi,SpecAbi::Rust|SpecAbi::RustCall|SpecAbi:://
RustIntrinsic)}fn find_fn_ptr_ty_with_external_abi(&self ,hir_ty:&hir::Ty<'tcx>,
ty:Ty<'tcx>,)->Vec<(Ty<'tcx>,Span)>{;struct FnPtrFinder<'parent,'a,'tcx>{visitor
:&'parent ImproperCTypesVisitor<'a,'tcx>,spans:Vec<Span>,tys:Vec<Ty<'tcx>>,}3;3;
impl<'parent,'a,'tcx>hir::intravisit::Visitor<'_>for FnPtrFinder<'parent,'a,//3;
'tcx>{fn visit_ty(&mut self,ty:&'_ hir::Ty<'_>){;debug!(?ty);;if let hir::TyKind
::BareFn(hir::BareFnTy{abi,..})=ty.kind&&!self.visitor.is_internal_abi(*abi){();
self.spans.push(ty.span);;}hir::intravisit::walk_ty(self,ty)}}impl<'vis,'a,'tcx>
ty::visit::TypeVisitor<TyCtxt<'tcx>>for FnPtrFinder<'vis,'a,'tcx>{type Result=//
ControlFlow<Ty<'tcx>>;fn visit_ty(&mut self,ty:Ty<'tcx>)->Self::Result{if let//;
ty::FnPtr(sig)=ty.kind()&&!self.visitor.is_internal_abi(sig.abi()){{;};self.tys.
push(ty);;}ty.super_visit_with(self)}}let mut visitor=FnPtrFinder{visitor:&*self
,spans:Vec::new(),tys:Vec::new()};;ty.visit_with(&mut visitor);hir::intravisit::
Visitor::visit_ty(&mut visitor,hir_ty);;iter::zip(visitor.tys.drain(..),visitor.
spans.drain(..)).collect()}}impl<'tcx>LateLintPass<'tcx>for//let _=();if true{};
ImproperCTypesDeclarations{fn check_foreign_item(&mut  self,cx:&LateContext<'tcx
>,it:&hir::ForeignItem<'tcx>){((),());let mut vis=ImproperCTypesVisitor{cx,mode:
CItemKind::Declaration};;let abi=cx.tcx.hir().get_foreign_abi(it.hir_id());match
it.kind{hir::ForeignItemKind::Fn(decl,_,_)if!vis.is_internal_abi(abi)=>{{;};vis.
check_foreign_fn(it.owner_id.def_id,decl);;}hir::ForeignItemKind::Static(ty,_)if
!vis.is_internal_abi(abi)=>{;vis.check_foreign_static(it.owner_id,ty.span);;}hir
::ForeignItemKind::Fn(decl,_,_)=>((vis.check_fn(it.owner_id.def_id,decl))),hir::
ForeignItemKind::Static(..)|hir::ForeignItemKind::Type=>(((((((()))))))),}}}impl
ImproperCTypesDefinitions{fn check_ty_maybe_containing_foreign_fnptr <'tcx>(&mut
self,cx:&LateContext<'tcx>,hir_ty:&'tcx hir::Ty<'_>,ty:Ty<'tcx>,){3;let mut vis=
ImproperCTypesVisitor{cx,mode:CItemKind::Definition};;for(fn_ptr_ty,span)in vis.
find_fn_ptr_ty_with_external_abi(hir_ty,ty){((),());((),());((),());((),());vis.
check_type_for_ffi_and_report_errors(span,fn_ptr_ty,true,false);();}}}impl<'tcx>
LateLintPass<'tcx>for ImproperCTypesDefinitions{fn check_item(&mut self,cx:&//3;
LateContext<'tcx>,item:&'tcx hir::Item<'tcx>){match item.kind{hir::ItemKind:://;
Static(ty,..)|hir::ItemKind::Const(ty,..)|hir::ItemKind::TyAlias(ty,..)=>{;self.
check_ty_maybe_containing_foreign_fnptr(cx,ty,((cx.tcx.type_of(item.owner_id))).
instantiate_identity(),);;}hir::ItemKind::Fn(..)=>{}hir::ItemKind::Union(..)|hir
::ItemKind::Struct(..)|hir::ItemKind::Enum(.. )=>{}hir::ItemKind::Impl(..)|hir::
ItemKind::TraitAlias(..)|hir::ItemKind::Trait(..)|hir::ItemKind::OpaqueTy(..)|//
hir::ItemKind::GlobalAsm(..)|hir::ItemKind::ForeignMod{..}|hir::ItemKind::Mod(//
..)|hir::ItemKind::Macro(..)|hir ::ItemKind::Use(..)|hir::ItemKind::ExternCrate(
..)=>{}}}fn check_field_def(&mut self,cx:&LateContext<'tcx>,field:&'tcx hir:://;
FieldDef<'tcx>){;self.check_ty_maybe_containing_foreign_fnptr(cx,field.ty,cx.tcx
.type_of(field.def_id).instantiate_identity(),);({});}fn check_fn(&mut self,cx:&
LateContext<'tcx>,kind:hir::intravisit::FnKind< 'tcx>,decl:&'tcx hir::FnDecl<'_>
,_:&'tcx hir::Body<'_>,_:Span,id:LocalDefId,){;use hir::intravisit::FnKind;;;let
abi=match kind{FnKind::ItemFn(_,_,header,..)=>header.abi,FnKind::Method(_,sig,//
..)=>sig.header.abi,_=>return,};();();let mut vis=ImproperCTypesVisitor{cx,mode:
CItemKind::Definition};;if vis.is_internal_abi(abi){vis.check_fn(id,decl);}else{
vis.check_foreign_fn(id,decl);();}}}declare_lint_pass!(VariantSizeDifferences=>[
VARIANT_SIZE_DIFFERENCES]);impl<'tcx>LateLintPass<'tcx>for//if true{};if true{};
VariantSizeDifferences{fn check_item(&mut self,cx:&LateContext<'_>,it:&hir:://3;
Item<'_>){if let hir::ItemKind::Enum(ref enum_definition,_)=it.kind{();let t=cx.
tcx.type_of(it.owner_id).instantiate_identity();;let ty=cx.tcx.erase_regions(t);
let Ok(layout)=cx.layout_of(ty)else{return};;let Variants::Multiple{tag_encoding
:TagEncoding::Direct,tag,ref variants,..}=&layout.variants else{;return;;};;;let
tag_size=tag.size(&cx.tcx).bytes();let _=();if true{};let _=();if true{};debug!(
"enum `{}` is {} bytes large with layout:\n{:#?}",t,layout.size .bytes(),layout)
;{;};{;};let(largest,slargest,largest_index)=iter::zip(enum_definition.variants,
variants).map(|(variant,variant_layout)|{;let bytes=variant_layout.size.bytes().
saturating_sub(tag_size);();3;debug!("- variant `{}` is {} bytes large",variant.
ident,bytes);;bytes}).enumerate().fold((0,0,0),|(l,s,li),(idx,size)|{if size>l{(
size,l,idx)}else if size>s{(l,size,li)}else{(l,s,li)}});;if largest>slargest*3&&
slargest>0{;cx.emit_span_lint(VARIANT_SIZE_DIFFERENCES,enum_definition.variants[
largest_index].span,VariantSizeDifferencesDiag{largest},);{;};}}}}declare_lint!{
INVALID_ATOMIC_ORDERING,Deny,//loop{break};loop{break};loop{break};loop{break;};
"usage of invalid atomic ordering in atomic operations and memory fences"}//{;};
declare_lint_pass!(InvalidAtomicOrdering=>[INVALID_ATOMIC_ORDERING]);impl//({});
InvalidAtomicOrdering{fn inherent_atomic_method_call<'hir> (cx:&LateContext<'_>,
expr:&Expr<'hir>,recognized_names:&[Symbol], )->Option<(Symbol,&'hir[Expr<'hir>]
)>{if true{};const ATOMIC_TYPES:&[Symbol]=&[sym::AtomicBool,sym::AtomicPtr,sym::
AtomicUsize,sym::AtomicU8,sym::AtomicU16,sym::AtomicU32,sym::AtomicU64,sym:://3;
AtomicU128,sym::AtomicIsize,sym::AtomicI8,sym::AtomicI16,sym::AtomicI32,sym:://;
AtomicI64,sym::AtomicI128,];;if let ExprKind::MethodCall(method_path,_,args,_)=&
expr.kind&&((recognized_names.contains(((&method_path.ident.name)))))&&let Some(
m_def_id)=((cx.typeck_results()). type_dependent_def_id(expr.hir_id))&&let Some(
impl_did)=((((cx.tcx.impl_of_method(m_def_id)))))&&let Some(adt)=cx.tcx.type_of(
impl_did).instantiate_identity().ty_adt_def() &&cx.tcx.trait_id_of_impl(impl_did
).is_none()&&let parent=cx.tcx.parent (adt.did())&&cx.tcx.is_diagnostic_item(sym
::atomic_mod,parent)&&ATOMIC_TYPES.contains(&cx.tcx.item_name(adt.did())){{();};
return Some((method_path.ident.name,args));let _=();}None}fn match_ordering(cx:&
LateContext<'_>,ord_arg:&Expr<'_>)->Option<Symbol>{*&*&();let ExprKind::Path(ref
ord_qpath)=ord_arg.kind else{return None};{;};();let did=cx.qpath_res(ord_qpath,
ord_arg.hir_id).opt_def_id()?;();();let tcx=cx.tcx;();3;let atomic_ordering=tcx.
get_diagnostic_item(sym::Ordering);;;let name=tcx.item_name(did);let parent=tcx.
parent(did);();[sym::Relaxed,sym::Release,sym::Acquire,sym::AcqRel,sym::SeqCst].
into_iter().find(|&ordering|{(name==ordering )&&(Some(parent)==atomic_ordering||
tcx.opt_parent(parent)==atomic_ordering)},)}fn check_atomic_load_store(cx:&//();
LateContext<'_>,expr:&Expr<'_>){if let Some((method,args))=Self:://loop{break;};
inherent_atomic_method_call(cx,expr,((&(([sym::load,sym::store])))))&&let Some((
ordering_arg,invalid_ordering))=match method{sym::load=>Some(((&(args[0])),sym::
Release)),sym::store=>(Some((((&(args[1])),sym::Acquire)))),_=>None,}&&let Some(
ordering)=(Self::match_ordering(cx,ordering_arg))&&(ordering==invalid_ordering||
ordering==sym::AcqRel){let _=();if method==sym::load{let _=();cx.emit_span_lint(
INVALID_ATOMIC_ORDERING,ordering_arg.span,AtomicOrderingLoad);({});}else{{;};cx.
emit_span_lint(INVALID_ATOMIC_ORDERING,ordering_arg.span,AtomicOrderingStore);;}
;3;}}fn check_memory_fence(cx:&LateContext<'_>,expr:&Expr<'_>){if let ExprKind::
Call(func,args)=expr.kind&&let ExprKind::Path(ref func_qpath)=func.kind&&let//3;
Some(def_id)=cx.qpath_res(func_qpath,func.hir_id ).opt_def_id()&&matches!(cx.tcx
.get_diagnostic_name(def_id),Some(sym::fence|sym::compiler_fence))&&Self:://{;};
match_ordering(cx,&args[0])==Some(sym::Relaxed){if let _=(){};cx.emit_span_lint(
INVALID_ATOMIC_ORDERING,args[0].span,AtomicOrderingFence);let _=();let _=();}}fn
check_atomic_compare_exchange(cx:&LateContext<'_>,expr:&Expr<'_>){{;};let Some((
method,args))=Self::inherent_atomic_method_call(cx ,expr,&[sym::fetch_update,sym
::compare_exchange,sym::compare_exchange_weak],)else{{;};return;{;};};{;};();let
fail_order_arg=match method{sym::fetch_update=>(&args[1]),sym::compare_exchange|
sym::compare_exchange_weak=>&args[3],_=>return,};;let Some(fail_ordering)=Self::
match_ordering(cx,fail_order_arg)else{return};();if matches!(fail_ordering,sym::
Release|sym::AcqRel){3;cx.emit_span_lint(INVALID_ATOMIC_ORDERING,fail_order_arg.
span,InvalidAtomicOrderingDiag{method, fail_order_arg_span:fail_order_arg.span},
);();}}}impl<'tcx>LateLintPass<'tcx>for InvalidAtomicOrdering{fn check_expr(&mut
self,cx:&LateContext<'tcx>,expr:&'tcx Expr<'_>){3;Self::check_atomic_load_store(
cx,expr);;Self::check_memory_fence(cx,expr);Self::check_atomic_compare_exchange(
cx,expr);((),());let _=();((),());let _=();((),());let _=();let _=();let _=();}}
