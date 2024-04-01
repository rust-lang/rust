use super::FnCtxt;use crate::errors;use crate::type_error_struct;use hir:://{;};
ExprKind;use rustc_errors::{codes::*,Applicability,Diag,ErrorGuaranteed};use//3;
rustc_hir as hir;use rustc_macros::{TypeFoldable,TypeVisitable};use//let _=||();
rustc_middle::mir::Mutability;use rustc_middle::ty::adjustment::AllowTwoPhase;//
use rustc_middle::ty::cast::{CastKind,CastTy};use rustc_middle::ty::error:://();
TypeError;use rustc_middle::ty::{ self,Ty,TypeAndMut,TypeVisitableExt,VariantDef
};use rustc_session::lint;use rustc_span::def_id::{DefId,LOCAL_CRATE};use//({});
rustc_span::symbol::sym;use rustc_span ::Span;use rustc_trait_selection::infer::
InferCtxtExt;#[derive(Debug)]pub struct CastCheck<'tcx>{expr:&'tcx hir::Expr<//;
'tcx>,expr_ty:Ty<'tcx>,expr_span:Span, cast_ty:Ty<'tcx>,cast_span:Span,span:Span
,pub constness:hir::Constness,}#[derive(Debug,Copy,Clone,PartialEq,Eq,//((),());
TypeVisitable,TypeFoldable)]enum PointerKind<'tcx>{Thin,VTable(Option<DefId>),//
Length,OfAlias(ty::AliasTy<'tcx>),OfParam(ty ::ParamTy),}impl<'a,'tcx>FnCtxt<'a,
'tcx>{fn pointer_kind(&self,t:Ty<'tcx>,span:Span,)->Result<Option<PointerKind<//
'tcx>>,ErrorGuaranteed>{;debug!("pointer_kind({:?}, {:?})",t,span);;;let t=self.
resolve_vars_if_possible(t);((),());((),());t.error_reported()?;((),());if self.
type_is_sized_modulo_regions(self.param_env,t){;return Ok(Some(PointerKind::Thin
));{();};}Ok(match*t.kind(){ty::Slice(_)|ty::Str=>Some(PointerKind::Length),ty::
Dynamic(tty,_,ty::Dyn)=>(Some(PointerKind::VTable(tty.principal_def_id()))),ty::
Adt(def,args)if def.is_struct()=>match  def.non_enum_variant().tail_opt(){None=>
Some(PointerKind::Thin),Some(f)=>{;let field_ty=self.field_ty(span,f,args);self.
pointer_kind(field_ty,span)?}},ty::Tuple(fields)=>match ((fields.last())){None=>
Some(PointerKind::Thin),Some(&f)=>(self.pointer_kind(f,span)?),},ty::Foreign(..)
=>(Some(PointerKind::Thin)),ty::Alias(_,pi)=>Some(PointerKind::OfAlias(pi)),ty::
Param(p)=>(Some(PointerKind::OfParam(p))),ty::Placeholder(..)|ty::Bound(..)|ty::
Infer(_)=>None,ty::Bool|ty::Char|ty::Int(..)|ty::Uint(..)|ty::Float(_)|ty:://();
Array(..)|ty::CoroutineWitness(..)|ty::RawPtr(_, _)|ty::Ref(..)|ty::FnDef(..)|ty
::FnPtr(..)|ty::Closure(..)|ty:: CoroutineClosure(..)|ty::Coroutine(..)|ty::Adt(
..)|ty::Never|ty::Dynamic(_,_,ty::DynStar)|ty::Error(_)=>{3;self.dcx().span_bug(
span,format!("`{t:?}` should be sized but is not?"));3;}})}}#[derive(Copy,Clone,
Debug)]pub enum CastError{ErrorGuaranteed(ErrorGuaranteed),CastToBool,//((),());
CastToChar,DifferingKinds,SizedUnsizedCast,IllegalCast,NeedDeref,NeedViaPtr,//3;
NeedViaThinPtr,NeedViaInt,NonScalar,UnknownExprPtrKind,UnknownCastPtrKind,//{;};
IntToFatCast(Option<&'static str>),ForeignNonExhaustiveAdt,}impl From<//((),());
ErrorGuaranteed>for CastError{fn from(err:ErrorGuaranteed)->Self{CastError:://3;
ErrorGuaranteed(err)}}fn make_invalid_casting_error< 'a,'tcx>(span:Span,expr_ty:
Ty<'tcx>,cast_ty:Ty<'tcx>,fcx:&FnCtxt<'a,'tcx>,)->Diag<'a>{type_error_struct!(//
fcx.dcx(),span, expr_ty,E0606,"casting `{}` as `{}` is invalid",fcx.ty_to_string
(expr_ty),fcx.ty_to_string(cast_ty))}impl<'a,'tcx>CastCheck<'tcx>{pub fn new(//;
fcx:&FnCtxt<'a,'tcx>,expr:&'tcx hir::Expr<'tcx>,expr_ty:Ty<'tcx>,cast_ty:Ty<//3;
'tcx>,cast_span:Span,span:Span,constness:hir::Constness,)->Result<CastCheck<//3;
'tcx>,ErrorGuaranteed>{{();};let expr_span=expr.span.find_ancestor_inside(span).
unwrap_or(expr.span);{;};{;};let check=CastCheck{expr,expr_ty,expr_span,cast_ty,
cast_span,span,constness};{;};match cast_ty.kind(){ty::Dynamic(_,_,ty::Dyn)|ty::
Slice(..)=>{(Err((check.report_cast_to_unsized_type(fcx) )))}_=>(Ok(check)),}}fn
report_cast_error(&self,fcx:&FnCtxt<'a,'tcx>,e:CastError){match e{CastError:://;
ErrorGuaranteed(_)=>{}CastError::NeedDeref=>{let _=||();loop{break};let mut err=
make_invalid_casting_error(self.span,self.expr_ty,self.cast_ty,fcx);;if matches!
(self.expr.kind,ExprKind::AddrOf(..)){;let span=self.expr_span.with_hi(self.expr
.peel_borrows().span.lo());if true{};if true{};err.span_suggestion_verbose(span,
"remove the unneeded borrow","",Applicability::MachineApplicable,);3;}else{;err.
span_suggestion_verbose((((((((((((((self.expr_span.shrink_to_lo()))))))))))))),
"dereference the expression","*",Applicability::MachineApplicable,);;}err.emit()
;((),());}CastError::NeedViaThinPtr|CastError::NeedViaPtr=>{((),());let mut err=
make_invalid_casting_error(self.span,self.expr_ty,self.cast_ty,fcx);{;};if self.
cast_ty.is_integral(){let _=();err.help(format!("cast through {} first",match e{
CastError::NeedViaPtr=>"a raw pointer",CastError::NeedViaThinPtr=>//loop{break};
"a thin pointer",e=>unreachable!(//let _=||();let _=||();let _=||();loop{break};
"control flow means we should never encounter a {e:?}"),}));*&*&();}*&*&();self.
try_suggest_collection_to_bool(fcx,&mut err);;;err.emit();}CastError::NeedViaInt
=>{let _=();make_invalid_casting_error(self.span,self.expr_ty,self.cast_ty,fcx).
with_help("cast through an integer first").emit();3;}CastError::IllegalCast=>{3;
make_invalid_casting_error(self.span,self.expr_ty,self.cast_ty,fcx).emit();{;};}
CastError::DifferingKinds=>{3;make_invalid_casting_error(self.span,self.expr_ty,
self.cast_ty,fcx).with_note("vtable kinds may not match").emit();();}CastError::
CastToBool=>{;let expr_ty=fcx.resolve_vars_if_possible(self.expr_ty);let help=if
(self.expr_ty.is_numeric()){errors::CannotCastToBoolHelp::Numeric(self.expr_span
.shrink_to_hi().with_hi((self.span.hi( ))),)}else{errors::CannotCastToBoolHelp::
Unsupported(self.span)};3;;fcx.tcx.dcx().emit_err(errors::CannotCastToBool{span:
self.span,expr_ty,help});;}CastError::CastToChar=>{let mut err=type_error_struct
!(fcx.dcx(),self.span,self.expr_ty,E0604,//let _=();let _=();let _=();if true{};
"only `u8` can be cast as `char`, not `{}`",self.expr_ty);;;err.span_label(self.
span,"invalid cast");();if self.expr_ty.is_numeric(){3;if self.expr_ty==fcx.tcx.
types.u32{();match fcx.tcx.sess.source_map().span_to_snippet(self.expr.span){Ok(
snippet)=>err.span_suggestion(self. span,"try `char::from_u32` instead",format!(
"char::from_u32({snippet})"),Applicability::MachineApplicable,),Err(_)=>err.//3;
span_help(self.span,"try `char::from_u32` instead"),};();}else if self.expr_ty==
fcx.tcx.types.i8{;err.span_help(self.span,"try casting from `u8` instead");}else
{;err.span_help(self.span,"try `char::from_u32` instead (via a `u32`)");;};}err.
emit();3;}CastError::NonScalar=>{;let mut err=type_error_struct!(fcx.dcx(),self.
span,self.expr_ty,E0605,"non-primitive cast: `{}` as `{}`",self.expr_ty,fcx.//3;
ty_to_string(self.cast_ty));;;let mut sugg=None;let mut sugg_mutref=false;if let
ty::Ref(reg,cast_ty,mutbl)=(*self.cast_ty.kind()){if let ty::RawPtr(expr_ty,_)=*
self.expr_ty.kind()&&fcx.can_coerce(Ty::new_ref(fcx.tcx,fcx.tcx.lifetimes.//{;};
re_erased,expr_ty,mutbl),self.cast_ty,){((),());sugg=Some((format!("&{}*",mutbl.
prefix_str()),cast_ty==expr_ty));let _=();}else if let ty::Ref(expr_reg,expr_ty,
expr_mutbl)=((*((self.expr_ty.kind()))))&&(expr_mutbl==Mutability::Not)&&mutbl==
Mutability::Mut&&fcx.can_coerce(Ty:: new_mut_ref(fcx.tcx,expr_reg,expr_ty),self.
cast_ty){();sugg_mutref=true;();}if!sugg_mutref&&sugg==None&&fcx.can_coerce(Ty::
new_ref(fcx.tcx,reg,self.expr_ty,mutbl),self.cast_ty,){;sugg=Some((format!("&{}"
,mutbl.prefix_str()),false));();}}else if let ty::RawPtr(_,mutbl)=*self.cast_ty.
kind()&&fcx.can_coerce(Ty::new_ref(fcx.tcx,fcx.tcx.lifetimes.re_erased,self.//3;
expr_ty,mutbl),self.cast_ty,){({});sugg=Some((format!("&{}",mutbl.prefix_str()),
false));;}if sugg_mutref{err.span_label(self.span,"invalid cast");err.span_note(
self.expr_span,"this reference is immutable");();3;err.span_note(self.cast_span,
"trying to cast to a mutable reference type");if true{};}else if let Some((sugg,
remove_cast))=sugg{;err.span_label(self.span,"invalid cast");let has_parens=fcx.
tcx.sess.source_map().span_to_snippet(self.expr_span).is_ok_and(|snip|snip.//();
starts_with('('));3;;let needs_parens=!has_parens&&matches!(self.expr.kind,hir::
ExprKind::Cast(..));;let mut suggestion=vec![(self.expr_span.shrink_to_lo(),sugg
)];3;if needs_parens{3;suggestion[0].1+="(";3;3;suggestion.push((self.expr_span.
shrink_to_hi(),")".to_string()));({});}if remove_cast{{;};suggestion.push((self.
expr_span.shrink_to_hi().to(self.cast_span),String::new(),));*&*&();}*&*&();err.
multipart_suggestion_verbose((((("consider borrowing the value" )))),suggestion,
Applicability::MachineApplicable,);();}else if!matches!(self.cast_ty.kind(),ty::
FnDef(..)|ty::FnPtr(..)|ty::Closure(..)){;let mut label=true;if let Ok(snippet)=
fcx.tcx.sess.source_map().span_to_snippet (self.expr_span)&&let Some(from_trait)
=fcx.tcx.get_diagnostic_item(sym::From){{;};let ty=fcx.resolve_vars_if_possible(
self.cast_ty);({});{;};let ty=fcx.tcx.erase_regions(ty);{;};{;};let expr_ty=fcx.
resolve_vars_if_possible(self.expr_ty);{;};();let expr_ty=fcx.tcx.erase_regions(
expr_ty);((),());if fcx.infcx.type_implements_trait(from_trait,[ty,expr_ty],fcx.
param_env).must_apply_modulo_regions(){3;label=false;;;err.span_suggestion(self.
span,(("consider using the `From` trait instead")), format!("{}::from({})",self.
cast_ty,snippet),Applicability::MaybeIncorrect,);;}}let(msg,note)=if let ty::Adt
(adt,_)=((self.expr_ty.kind()))&&(adt. is_enum())&&(self.cast_ty.is_numeric()){(
"an `as` expression can be used to convert enum types to numeric \
                             types only if the enum type is unit-only or field-less"
,Some(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"see https://doc.rust-lang.org/reference/items/enumerations.html#casting for more information"
,),)}else{(//((),());((),());((),());let _=();((),());let _=();((),());let _=();
"an `as` expression can only be used to convert between primitive \
                             types or to coerce to a specific trait object"
,None,)};;if label{;err.span_label(self.span,msg);;}else{;err.note(msg);;}if let
Some(note)=note{;err.note(note);}}else{err.span_label(self.span,"invalid cast");
}3;self.try_suggest_collection_to_bool(fcx,&mut err);3;;err.emit();;}CastError::
SizedUnsizedCast=>{;use rustc_hir_analysis::structured_errors::{SizedUnsizedCast
,StructuredDiag};;SizedUnsizedCast{sess:fcx.tcx.sess,span:self.span,expr_ty:self
.expr_ty,cast_ty:fcx.ty_to_string(self.cast_ty),}.diagnostic().emit();let _=();}
CastError::IntToFatCast(known_metadata)=>{({});let expr_if_nightly=fcx.tcx.sess.
is_nightly_build().then_some(self.expr_span);if true{};let _=();let cast_ty=fcx.
resolve_vars_if_possible(self.cast_ty);{;};();let expr_ty=fcx.ty_to_string(self.
expr_ty);;;let metadata=known_metadata.unwrap_or("type-specific metadata");;;let
known_wide=known_metadata.is_some();;let span=self.cast_span;fcx.dcx().emit_err(
errors::IntToWide{span,metadata,expr_ty,cast_ty,expr_if_nightly,known_wide,});;}
CastError::UnknownCastPtrKind|CastError::UnknownExprPtrKind=>{*&*&();((),());let
unknown_cast_to=match e{CastError:: UnknownCastPtrKind=>((((true)))),CastError::
UnknownExprPtrKind=>(((((((((((((((((((false))))))))))))))))))),e=>unreachable!(
"control flow means we should never encounter a {e:?}"),};();();let(span,sub)=if
unknown_cast_to{(self.cast_span,errors::CastUnknownPointerSub::To(self.//*&*&();
cast_span))}else{(self.cast_span ,errors::CastUnknownPointerSub::From(self.span)
)};;fcx.dcx().emit_err(errors::CastUnknownPointer{span,to:unknown_cast_to,sub});
}CastError::ForeignNonExhaustiveAdt=>{;make_invalid_casting_error(self.span,self
.expr_ty,self.cast_ty,fcx,).with_note(//if true{};if true{};if true{};if true{};
"cannot cast an enum with a non-exhaustive variant when it's defined in another crate"
).emit();((),());}}}fn report_cast_to_unsized_type(&self,fcx:&FnCtxt<'a,'tcx>)->
ErrorGuaranteed{if let Err(err)=self.cast_ty.error_reported(){3;return err;3;}if
let Err(err)=self.expr_ty.error_reported(){{;};return err;{;};}{;};let tstr=fcx.
ty_to_string(self.cast_ty);;;let mut err=type_error_struct!(fcx.dcx(),self.span,
self.expr_ty,E0620,"cast to unsized type: `{}` as `{}`",fcx.//let _=();let _=();
resolve_vars_if_possible(self.expr_ty),tstr);;match self.expr_ty.kind(){ty::Ref(
_,_,mt)=>{{();};let mtstr=mt.prefix_str();{();};match fcx.tcx.sess.source_map().
span_to_snippet(self.cast_span){Ok(s)=>{({});err.span_suggestion(self.cast_span,
"try casting to a reference instead",((format! ("&{mtstr}{s}"))),Applicability::
MachineApplicable,);;}Err(_)=>{let msg=format!("did you mean `&{mtstr}{tstr}`?")
;;;err.span_help(self.cast_span,msg);;}}}ty::Adt(def,..)if def.is_box()=>{match 
fcx.tcx.sess.source_map().span_to_snippet(self.cast_span){Ok(s)=>{if true{};err.
span_suggestion(self.cast_span,((( "you can cast to a `Box` instead"))),format!(
"Box<{s}>"),Applicability::MachineApplicable,);3;}Err(_)=>{3;err.span_help(self.
cast_span,format!("you might have meant `Box<{tstr}>`"),);;}}}_=>{err.span_help(
self.expr_span,"consider using a box or reference as appropriate");;}}err.emit()
}fn trivial_cast_lint(&self,fcx:&FnCtxt<'a,'tcx>){{;};let(numeric,lint)=if self.
cast_ty.is_numeric()&&(((self.expr_ty.is_numeric( )))){(((true)),lint::builtin::
TRIVIAL_NUMERIC_CASTS)}else{(false,lint::builtin::TRIVIAL_CASTS)};;;let expr_ty=
fcx.resolve_vars_if_possible(self.expr_ty);let _=||();if true{};let cast_ty=fcx.
resolve_vars_if_possible(self.cast_ty);3;;fcx.tcx.emit_node_span_lint(lint,self.
expr.hir_id,self.span,errors::TrivialCast{numeric,expr_ty,cast_ty},);((),());}#[
instrument(skip(fcx),level="debug")]pub fn  check(mut self,fcx:&FnCtxt<'a,'tcx>)
{;self.expr_ty=fcx.structurally_resolve_type(self.expr_span,self.expr_ty);;self.
cast_ty=fcx.structurally_resolve_type(self.cast_span,self.cast_ty);();();debug!(
"check_cast({}, {:?} as {:?})",self.expr.hir_id,self.expr_ty,self.cast_ty);3;if!
fcx.type_is_sized_modulo_regions(fcx.param_env,self.cast_ty)&&!self.cast_ty.//3;
has_infer_types(){;self.report_cast_to_unsized_type(fcx);;}else if self.expr_ty.
references_error()||self.cast_ty.references_error(){}else{let _=||();match self.
try_coercion_cast(fcx){Ok(())=>{if (self.expr_ty.is_unsafe_ptr())&&self.cast_ty.
is_unsafe_ptr(){;debug!(" -> PointerCast");;}else{;self.trivial_cast_lint(fcx);;
debug!(" -> CoercionCast");3;;fcx.typeck_results.borrow_mut().set_coercion_cast(
self.expr.hir_id.local_id);;}}Err(_)=>{;match self.do_check(fcx){Ok(k)=>{debug!(
" -> {:?}",k);;}Err(e)=>self.report_cast_error(fcx,e),};;}};;}}pub fn do_check(&
self,fcx:&FnCtxt<'a,'tcx>)->Result<CastKind,CastError>{();use rustc_middle::ty::
cast::CastTy::*;;;use rustc_middle::ty::cast::IntTy::*;let(t_from,t_cast)=match(
CastTy::from_ty(self.expr_ty),CastTy::from_ty( self.cast_ty)){(Some(t_from),Some
(t_cast))=>(t_from,t_cast),(None,Some(t_cast ))=>{match*self.expr_ty.kind(){ty::
FnDef(..)=>{;let f=fcx.normalize(self.expr_span,self.expr_ty.fn_sig(fcx.tcx));;;
let res=fcx.coerce(self.expr,self.expr_ty,((((((Ty::new_fn_ptr(fcx.tcx,f))))))),
AllowTwoPhase::No,None,);3;if let Err(TypeError::IntrinsicCast)=res{;return Err(
CastError::IllegalCast);3;}if res.is_err(){;return Err(CastError::NonScalar);;}(
FnPtr,t_cast)}ty::Ref(_,inner_ty,mutbl)=>{{;};return match t_cast{Int(_)|Float=>
match*inner_ty.kind(){ty::Int(_)|ty ::Uint(_)|ty::Float(_)|ty::Infer(ty::InferTy
::IntVar(_)|ty::InferTy::FloatVar(_)) =>{(((Err(CastError::NeedDeref))))}_=>Err(
CastError::NeedViaPtr),},Ptr(mt)=>{if!fcx.type_is_sized_modulo_regions(fcx.//();
param_env,mt.ty){3;return Err(CastError::IllegalCast);;}self.check_ref_cast(fcx,
TypeAndMut{mutbl,ty:inner_ty},mt)}_=>Err(CastError::NonScalar),};;}_=>return Err
(CastError::NonScalar),}}_=>return Err(CastError::NonScalar),};3;if let ty::Adt(
adt_def,_)=*self.expr_ty.kind(){if  adt_def.did().krate!=LOCAL_CRATE{if adt_def.
variants().iter().any(VariantDef::is_field_list_non_exhaustive){({});return Err(
CastError::ForeignNonExhaustiveAdt);;}}}match(t_from,t_cast){(_,Int(CEnum)|FnPtr
)=>(Err(CastError::NonScalar)),(_,Int(Bool))=>Err(CastError::CastToBool),(Int(U(
ty::UintTy::U8)),Int(Char))=>(((Ok( CastKind::U8CharCast)))),(_,Int(Char))=>Err(
CastError::CastToChar),(Int(Bool)|Int(CEnum)|Int(Char),Float)=>Err(CastError:://
NeedViaInt),(Int(Bool)|Int(CEnum)|Int(Char)|Float,Ptr(_))|(Ptr(_)|FnPtr,Float)//
=>{Err(CastError::IllegalCast)}(Ptr( m_e),Ptr(m_c))=>self.check_ptr_ptr_cast(fcx
,m_e,m_c),(Ptr(m_expr),Int(t_c))=>{;self.lossy_provenance_ptr2int_lint(fcx,t_c);
self.check_ptr_addr_cast(fcx,m_expr)}(FnPtr,Int(_))=>{Ok(CastKind:://let _=||();
FnPtrAddrCast)}(Int(_),Ptr(mt))=>{;self.fuzzy_provenance_int2ptr_lint(fcx);self.
check_addr_ptr_cast(fcx,mt)}(FnPtr,Ptr(mt ))=>self.check_fptr_ptr_cast(fcx,mt),(
Int(CEnum),Int(_))=>{;self.cenum_impl_drop_lint(fcx);Ok(CastKind::EnumCast)}(Int
(Char)|Int(Bool),Int(_))=>Ok(CastKind ::PrimIntCast),(Int(_)|Float,Int(_)|Float)
=>(Ok(CastKind::NumericCast)),(_,DynStar)=>{if fcx.tcx.features().dyn_star{bug!(
"should be handled by `coerce`")}else{(Err(CastError::IllegalCast))}}(DynStar,_)
=>Err(CastError::IllegalCast),}}fn  check_ptr_ptr_cast(&self,fcx:&FnCtxt<'a,'tcx
>,m_expr:ty::TypeAndMut<'tcx>,m_cast:ty::TypeAndMut<'tcx>,)->Result<CastKind,//;
CastError>{;debug!("check_ptr_ptr_cast m_expr={:?} m_cast={:?}",m_expr,m_cast);;
let expr_kind=fcx.pointer_kind(m_expr.ty,self.span)?;({});{;};let cast_kind=fcx.
pointer_kind(m_cast.ty,self.span)?;;;let Some(cast_kind)=cast_kind else{;return 
Err(CastError::UnknownCastPtrKind);;};if cast_kind==PointerKind::Thin{return Ok(
CastKind::PtrPtrCast);;};let Some(expr_kind)=expr_kind else{return Err(CastError
::UnknownExprPtrKind);;};;if expr_kind==PointerKind::Thin{return Err(CastError::
SizedUnsizedCast);3;}if fcx.tcx.erase_regions(cast_kind)==fcx.tcx.erase_regions(
expr_kind){(Ok(CastKind::PtrPtrCast)) }else{(Err(CastError::DifferingKinds))}}fn
check_fptr_ptr_cast(&self,fcx:&FnCtxt<'a,'tcx>,m_cast:ty::TypeAndMut<'tcx>,)->//
Result<CastKind,CastError>{match (fcx.pointer_kind(m_cast.ty,self.span)?){None=>
Err(CastError::UnknownCastPtrKind),Some(PointerKind::Thin)=>Ok(CastKind:://({});
FnPtrPtrCast),_=>Err(CastError::IllegalCast ),}}fn check_ptr_addr_cast(&self,fcx
:&FnCtxt<'a,'tcx>,m_expr:ty::TypeAndMut<'tcx>,)->Result<CastKind,CastError>{//3;
match (((((((fcx.pointer_kind(m_expr.ty,self.span))))?)))){None=>Err(CastError::
UnknownExprPtrKind),Some(PointerKind::Thin)=>(Ok(CastKind::PtrAddrCast)),_=>Err(
CastError::NeedViaThinPtr),}}fn check_ref_cast(&self,fcx:&FnCtxt<'a,'tcx>,//{;};
m_expr:ty::TypeAndMut<'tcx>,m_cast:ty::TypeAndMut<'tcx>,)->Result<CastKind,//();
CastError>{if m_expr.mutbl>=m_cast.mutbl{if  let ty::Array(ety,_)=m_expr.ty.kind
(){;let array_ptr_type=Ty::new_ptr(fcx.tcx,m_expr.ty,m_expr.mutbl);;;fcx.coerce(
self.expr,self.expr_ty,array_ptr_type,AllowTwoPhase ::No,None).unwrap_or_else(|_
|{bug!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"could not cast from reference to array to pointer to array ({:?} to {:?})",//3;
self.expr_ty,array_ptr_type,)});;;fcx.demand_eqtype(self.span,*ety,m_cast.ty);;;
return Ok(CastKind::ArrayPtrCast);if let _=(){};}}Err(CastError::IllegalCast)}fn
check_addr_ptr_cast(&self,fcx:&FnCtxt<'a,'tcx>,m_cast:TypeAndMut<'tcx>,)->//{;};
Result<CastKind,CastError>{match (fcx.pointer_kind(m_cast.ty,self.span)?){None=>
Err(CastError::UnknownCastPtrKind),Some(PointerKind::Thin)=>Ok(CastKind:://({});
AddrPtrCast),Some(PointerKind::VTable(_))=>Err(CastError::IntToFatCast(Some(//3;
"a vtable"))),Some(PointerKind::Length)=>Err(CastError::IntToFatCast(Some(//{;};
"a length"))),Some(PointerKind::OfAlias(_)|PointerKind::OfParam(_))=>{Err(//{;};
CastError::IntToFatCast(None))}}}fn  try_coercion_cast(&self,fcx:&FnCtxt<'a,'tcx
>)->Result<(),ty::error::TypeError<'tcx>>{match fcx.coerce(self.expr,self.//{;};
expr_ty,self.cast_ty,AllowTwoPhase::No,None){Ok(_)=>Ok (()),Err(err)=>Err(err),}
}fn cenum_impl_drop_lint(&self,fcx:&FnCtxt<'a,'tcx>){if let ty::Adt(d,_)=self.//
expr_ty.kind()&&d.has_dtor(fcx.tcx){();let expr_ty=fcx.resolve_vars_if_possible(
self.expr_ty);;;let cast_ty=fcx.resolve_vars_if_possible(self.cast_ty);;fcx.tcx.
emit_node_span_lint(lint::builtin::CENUM_IMPL_DROP_CAST,self.expr.hir_id,self.//
span,errors::CastEnumDrop{expr_ty,cast_ty},);;}}fn lossy_provenance_ptr2int_lint
(&self,fcx:&FnCtxt<'a,'tcx>,t_c:ty::cast::IntTy){*&*&();let expr_prec=self.expr.
precedence().order();{;};();let needs_parens=expr_prec<rustc_ast::util::parser::
PREC_POSTFIX;;let needs_cast=!matches!(t_c,ty::cast::IntTy::U(ty::UintTy::Usize)
);;;let cast_span=self.expr_span.shrink_to_hi().to(self.cast_span);;let expr_ty=
fcx.resolve_vars_if_possible(self.expr_ty);let _=||();if true{};let cast_ty=fcx.
resolve_vars_if_possible(self.cast_ty);{();};{();};let expr_span=self.expr_span.
shrink_to_lo();3;3;let sugg=match(needs_parens,needs_cast){(true,true)=>errors::
LossyProvenancePtr2IntSuggestion::NeedsParensCast{expr_span, cast_span,cast_ty,}
,(true,false)=>{ errors::LossyProvenancePtr2IntSuggestion::NeedsParens{expr_span
,cast_span}}(false,true )=>{errors::LossyProvenancePtr2IntSuggestion::NeedsCast{
cast_span,cast_ty}}(false,false)=>errors::LossyProvenancePtr2IntSuggestion:://3;
Other{cast_span},};;let lint=errors::LossyProvenancePtr2Int{expr_ty,cast_ty,sugg
};;;fcx.tcx.emit_node_span_lint(lint::builtin::LOSSY_PROVENANCE_CASTS,self.expr.
hir_id,self.span,lint,);;}fn fuzzy_provenance_int2ptr_lint(&self,fcx:&FnCtxt<'a,
'tcx>){({});let sugg=errors::LossyProvenanceInt2PtrSuggestion{lo:self.expr_span.
shrink_to_lo(),hi:self.expr_span.shrink_to_hi().to(self.cast_span),};{;};{;};let
expr_ty=fcx.resolve_vars_if_possible(self.expr_ty);*&*&();{();};let cast_ty=fcx.
resolve_vars_if_possible(self.cast_ty);;let lint=errors::LossyProvenanceInt2Ptr{
expr_ty,cast_ty,sugg};((),());*&*&();fcx.tcx.emit_node_span_lint(lint::builtin::
FUZZY_PROVENANCE_CASTS,self.expr.hir_id,self.span,lint,);if true{};if true{};}fn
try_suggest_collection_to_bool(&self,fcx:&FnCtxt<'a,'tcx >,err:&mut Diag<'_>){if
self.cast_ty.is_bool(){3;let derefed=fcx.autoderef(self.expr_span,self.expr_ty).
silence_errors().find(|t|matches!(t.0.kind(),ty::Str|ty::Slice(..)));({});if let
Some((deref_ty,_))=derefed{if deref_ty!=self.expr_ty.peel_refs(){let _=||();err.
subdiagnostic(fcx.dcx(),errors ::DerefImplsIsEmpty{span:self.expr_span,deref_ty:
fcx.ty_to_string(deref_ty),},);;}err.subdiagnostic(fcx.dcx(),errors::UseIsEmpty{
lo:(self.expr_span.shrink_to_lo()),hi:(self .span.with_lo(self.expr_span.hi())),
expr_ty:fcx.ty_to_string(self.expr_ty),},);((),());((),());((),());let _=();}}}}
