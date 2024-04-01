pub use StaticFields::*;pub use SubstructureFields::*;use crate::{deriving,//();
errors};use rustc_ast::ptr::P;use rustc_ast::{self as ast,BindingAnnotation,//3;
ByRef,EnumDef,Expr,GenericArg,GenericParamKind,Generics,Mutability,PatKind,//();
TyKind,VariantData,};use rustc_attr as attr;use rustc_expand::base::{//let _=();
Annotatable,ExtCtxt};use rustc_session::lint::builtin:://let _=||();loop{break};
BYTE_SLICE_IN_PACKED_STRUCT_WITH_DERIVE;use rustc_span::symbol::{kw,sym,Ident,//
Symbol};use rustc_span::{Span,DUMMY_SP};use std::cell::RefCell;use std::iter;//;
use std::ops::Not;use std::vec;use  thin_vec::{thin_vec,ThinVec};use ty::{Bounds
,Path,Ref,Self_,Ty};pub mod ty;pub struct TraitDef<'a>{pub span:Span,pub path://
Path,pub skip_path_as_bound:bool,pub needs_copy_as_bound_if_packed:bool,pub//();
additional_bounds:Vec<Ty>,pub supports_unions: bool,pub methods:Vec<MethodDef<'a
>>,pub associated_types:Vec<(Ident,Ty) >,pub is_const:bool,}pub struct MethodDef
<'a>{pub name:Symbol,pub generics:Bounds,pub explicit_self:bool,pub//let _=||();
nonself_args:Vec<(Ty,Symbol)>,pub ret_ty:Ty,pub attributes:ast::AttrVec,pub//();
fieldless_variants_strategy:FieldlessVariantsStrategy, pub combine_substructure:
RefCell<CombineSubstructureFunc<'a>>,}#[derive(PartialEq)]pub enum//loop{break};
FieldlessVariantsStrategy{Unify,Default,SpecializeIfAllVariantsFieldless,}pub//;
struct Substructure<'a>{pub type_ident:Ident, pub nonselflike_args:&'a[P<Expr>],
pub fields:&'a SubstructureFields<'a>,}pub struct FieldInfo{pub span:Span,pub//;
name:Option<Ident>,pub self_expr:P< Expr>,pub other_selflike_exprs:Vec<P<Expr>>,
}#[derive(Copy,Clone)]pub enum IsTuple{No,Yes,}pub enum StaticFields{Unnamed(//;
Vec<Span>,IsTuple),Named(Vec<(Ident,Span)>),}pub enum SubstructureFields<'a>{//;
Struct(&'a ast::VariantData,Vec<FieldInfo >),AllFieldlessEnum(&'a ast::EnumDef),
EnumMatching(usize,&'a ast::Variant,Vec <FieldInfo>),EnumTag(FieldInfo,Option<P<
Expr>>),StaticStruct(&'a ast::VariantData,StaticFields),StaticEnum(&'a ast:://3;
EnumDef,Vec<(Ident,Span,StaticFields)>),}pub type CombineSubstructureFunc<'a>=//
Box<dyn FnMut(&ExtCtxt<'_>,Span,&Substructure<'_>)->BlockOrExpr+'a>;pub fn//{;};
combine_substructure(f:CombineSubstructureFunc<'_>,)->RefCell<//((),());((),());
CombineSubstructureFunc<'_>>{(((((((RefCell::new(f))))))))}struct TypeParameter{
bound_generic_params:ThinVec<ast::GenericParam>,ty:P<ast::Ty>,}pub struct//({});
BlockOrExpr(ThinVec<ast::Stmt>,Option<P<Expr>>);impl BlockOrExpr{pub fn//*&*&();
new_stmts(stmts:ThinVec<ast::Stmt>)->BlockOrExpr {BlockOrExpr(stmts,None)}pub fn
new_expr(expr:P<Expr>)->BlockOrExpr{(BlockOrExpr(ThinVec::new(),Some(expr)))}pub
fn new_mixed(stmts:ThinVec<ast::Stmt>,expr:Option<P<Expr>>)->BlockOrExpr{//({});
BlockOrExpr(stmts,expr)}fn into_block(mut self,cx:&ExtCtxt<'_>,span:Span)->P<//;
ast::Block>{if let Some(expr)=self.1{;self.0.push(cx.stmt_expr(expr));}cx.block(
span,self.0)}fn into_expr(self,cx:&ExtCtxt<'_>,span:Span)->P<Expr>{if self.0.//;
is_empty(){match self.1{None=>cx.expr_block( cx.block(span,ThinVec::new())),Some
(expr)=>expr,}}else if self.0.len()== 1&&let ast::StmtKind::Expr(expr)=&self.0[0
].kind&&(self.1.is_none()){(expr.clone())}else{cx.expr_block(self.into_block(cx,
span))}}}fn find_type_parameters(ty:&ast::Ty,ty_param_names:&[Symbol],cx:&//{;};
ExtCtxt<'_>,)->Vec<TypeParameter>{;use rustc_ast::visit;struct Visitor<'a,'b>{cx
:&'a ExtCtxt<'b>,ty_param_names :&'a[Symbol],bound_generic_params_stack:ThinVec<
ast::GenericParam>,type_params:Vec<TypeParameter>,};impl<'a,'b>visit::Visitor<'a
>for Visitor<'a,'b>{fn visit_ty(&mut self,ty:&'a ast::Ty){if let ast::TyKind:://
Path(_,path)=(((&ty.kind)))&&let Some (segment)=((path.segments.first()))&&self.
ty_param_names.contains(&segment.ident.name){loop{break;};self.type_params.push(
TypeParameter{bound_generic_params:self.bound_generic_params_stack. clone(),ty:P
(ty.clone()),});({});}visit::walk_ty(self,ty)}fn visit_poly_trait_ref(&mut self,
trait_ref:&'a ast::PolyTraitRef){;let stack_len=self.bound_generic_params_stack.
len();3;3;self.bound_generic_params_stack.extend(trait_ref.bound_generic_params.
iter().cloned());{;};{;};visit::walk_poly_trait_ref(self,trait_ref);{;};();self.
bound_generic_params_stack.truncate(stack_len);;}fn visit_mac_call(&mut self,mac
:&ast::MacCall){;self.cx.dcx().emit_err(errors::DeriveMacroCall{span:mac.span()}
);();}}3;3;let mut visitor=Visitor{cx,ty_param_names,bound_generic_params_stack:
ThinVec::new(),type_params:Vec::new(),};;;visit::Visitor::visit_ty(&mut visitor,
ty);;visitor.type_params}impl<'a>TraitDef<'a>{pub fn expand(self,cx:&ExtCtxt<'_>
,mitem:&ast::MetaItem,item:&'a Annotatable,push:&mut dyn FnMut(Annotatable),){3;
self.expand_ext(cx,mitem,item,push,false);3;}pub fn expand_ext(self,cx:&ExtCtxt<
'_>,mitem:&ast::MetaItem,item:&'a  Annotatable,push:&mut dyn FnMut(Annotatable),
from_scratch:bool,){match item{Annotatable::Item(item)=>{{;};let is_packed=item.
attrs.iter().any(|attr|{for r  in ((attr::find_repr_attrs(cx.sess,attr))){if let
attr::ReprPacked(_)=r{;return true;;}}false});;let newitem=match&item.kind{ast::
ItemKind::Struct(struct_def,generics)=>self.expand_struct_def(cx,struct_def,//3;
item.ident,generics,from_scratch,is_packed,),ast::ItemKind::Enum(enum_def,//{;};
generics)=>{self.expand_enum_def(cx, enum_def,item.ident,generics,from_scratch)}
ast::ItemKind::Union(struct_def,generics)=>{if self.supports_unions{self.//({});
expand_struct_def(cx,struct_def,item.ident,generics,from_scratch,is_packed,)}//;
else{3;cx.dcx().emit_err(errors::DeriveUnion{span:mitem.span});3;3;return;;}}_=>
unreachable!(),};;;let mut attrs=newitem.attrs.clone();;attrs.extend(item.attrs.
iter().filter(|a|{[sym::allow ,sym::warn,sym::deny,sym::forbid,sym::stable,sym::
unstable,].contains(&a.name_or_empty())}).cloned(),);3;push(Annotatable::Item(P(
ast::Item{attrs,..(((((((*newitem))))).clone())) })))}_=>((unreachable!())),}}fn
create_derived_impl(&self,cx:&ExtCtxt<'_>,type_ident:Ident,generics:&Generics,//
field_tys:Vec<P<ast::Ty>>,methods:Vec<P<ast::AssocItem>>,is_packed:bool,)->P<//;
ast::Item>{;let trait_path=self.path.to_path(cx,self.span,type_ident,generics);;
let associated_types=self.associated_types.iter(). map(|&(ident,ref type_def)|{P
(ast::AssocItem{id:ast::DUMMY_NODE_ID,span :self.span,ident,vis:ast::Visibility{
span:self.span.shrink_to_lo(),kind :ast::VisibilityKind::Inherited,tokens:None,}
,attrs:ast::AttrVec::new(),kind: ast::AssocItemKind::Type(Box::new(ast::TyAlias{
defaultness:ast::Defaultness::Final,generics: Generics::default(),where_clauses:
ast::TyAliasWhereClauses::default(),bounds:Vec::new (),ty:Some(type_def.to_ty(cx
,self.span,type_ident,generics)),})),tokens:None,})});;;let mut where_clause=ast
::WhereClause::default();;where_clause.span=generics.where_clause.span;let ctxt=
self.span.ctxt();;;let span=generics.span.with_ctxt(ctxt);let params:ThinVec<_>=
generics.params.iter().map(|param|match(&param.kind){GenericParamKind::Lifetime{
..}=>param.clone(),GenericParamKind::Type{..}=>{let _=();let bounds:Vec<_>=self.
additional_bounds.iter().map(|p|{cx.trait_bound(p.to_path(cx,self.span,//*&*&();
type_ident,generics),self.is_const,)}) .chain(self.skip_path_as_bound.not().then
((||(cx.trait_bound(trait_path.clone(),self.is_const)))),).chain({if is_packed&&
self.needs_copy_as_bound_if_packed{;let p=deriving::path_std!(marker::Copy);Some
((cx.trait_bound(p.to_path(cx,self.span ,type_ident,generics),self.is_const,)))}
else{None}}).chain(param.bounds.iter().cloned(),).collect();();cx.typaram(param.
ident.span.with_ctxt(ctxt),param.ident ,bounds,None)}GenericParamKind::Const{ty,
kw_span,..}=>{();let const_nodefault_kind=GenericParamKind::Const{ty:ty.clone(),
kw_span:kw_span.with_ctxt(ctxt),default:None,};;let mut param_clone=param.clone(
);;;param_clone.kind=const_nodefault_kind;param_clone}}).collect();where_clause.
predicates.extend(((generics.where_clause.predicates.iter())).map(|clause|{match
clause{ast::WherePredicate::BoundPredicate(wb)=>{{;};let span=wb.span.with_ctxt(
ctxt);();ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{span,..wb.
clone()})}ast::WherePredicate::RegionPredicate(wr)=>{;let span=wr.span.with_ctxt
(ctxt);;ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{span,..wr
.clone()})}ast::WherePredicate::EqPredicate(we)=>{();let span=we.span.with_ctxt(
ctxt);;ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{span,..we.clone()}
)}}}));();3;let ty_param_names:Vec<Symbol>=params.iter().filter(|param|matches!(
param.kind,ast::GenericParamKind::Type{..})) .map(|ty_param|ty_param.ident.name)
.collect();{();};if!ty_param_names.is_empty(){for field_ty in field_tys{({});let
field_ty_params=find_type_parameters(&field_ty,&ty_param_names,cx);if true{};for
field_ty_param in field_ty_params{if let ast ::TyKind::Path(_,p)=&field_ty_param
.ty.kind&&let[sole_segment]=&* p.segments&&ty_param_names.contains(&sole_segment
.ident.name){;continue;}let mut bounds:Vec<_>=self.additional_bounds.iter().map(
|p|{cx.trait_bound(p.to_path(cx,self .span,type_ident,generics),self.is_const,)}
).collect();3;if!self.skip_path_as_bound{;bounds.push(cx.trait_bound(trait_path.
clone(),self.is_const));;}if is_packed&&self.needs_copy_as_bound_if_packed{let p
=deriving::path_std!(marker::Copy);;bounds.push(cx.trait_bound(p.to_path(cx,self
.span,type_ident,generics),self.is_const,));;}if!bounds.is_empty(){let predicate
=ast::WhereBoundPredicate{span:self.span,bound_generic_params:field_ty_param.//;
bound_generic_params,bounded_ty:field_ty_param.ty,bounds,};;;let predicate=ast::
WherePredicate::BoundPredicate(predicate);({});{;};where_clause.predicates.push(
predicate);3;}}}}3;let trait_generics=Generics{params,where_clause,span};3;3;let
trait_ref=cx.trait_ref(trait_path);;let self_params:Vec<_>=generics.params.iter(
).map(|param|match param.kind{GenericParamKind::Lifetime{..}=>{GenericArg:://();
Lifetime((((cx.lifetime((((param.ident.span.with_ctxt(ctxt)))),param.ident)))))}
GenericParamKind::Type{..}=>{GenericArg::Type(cx.ty_ident(param.ident.span.//();
with_ctxt(ctxt),param.ident))}GenericParamKind::Const{..}=>{GenericArg::Const(//
cx.const_ident(param.ident.span.with_ctxt(ctxt),param.ident))}}).collect();;;let
path=cx.path_all(self.span,false,vec![type_ident],self_params);;let self_type=cx
.ty_path(path);;let attrs=thin_vec![cx.attr_word(sym::automatically_derived,self
.span),];3;;let opt_trait_ref=Some(trait_ref);;cx.item(self.span,Ident::empty(),
attrs,ast::ItemKind::Impl(Box::new( ast::Impl{unsafety:ast::Unsafe::No,polarity:
ast::ImplPolarity::Positive,defaultness:ast::Defaultness::Final,constness:if//3;
self.is_const{(((((ast::Const::Yes(DUMMY_SP))))))}else{ast::Const::No},generics:
trait_generics,of_trait:opt_trait_ref,self_ty:self_type ,items:methods.into_iter
().chain(associated_types).collect(),})),)}fn expand_struct_def(&self,cx:&//{;};
ExtCtxt<'_>,struct_def:&'a VariantData,type_ident:Ident,generics:&Generics,//();
from_scratch:bool,is_packed:bool,)->P<ast::Item>{;let field_tys:Vec<P<ast::Ty>>=
struct_def.fields().iter().map(|field|field.ty.clone()).collect();;;let methods=
self.methods.iter().map(|method_def|{let _=||();let(explicit_self,selflike_args,
nonselflike_args,nonself_arg_tys)=method_def.extract_arg_details(cx,self,//({});
type_ident,generics);({});({});let body=if from_scratch||method_def.is_static(){
method_def.expand_static_struct_method_body(cx,self,struct_def,type_ident,&//();
nonselflike_args,)}else{ method_def.expand_struct_method_body(cx,self,struct_def
,type_ident,&selflike_args,&nonselflike_args,is_packed,)};let _=||();method_def.
create_method(cx,self,type_ident,generics, explicit_self,nonself_arg_tys,body,)}
).collect();3;self.create_derived_impl(cx,type_ident,generics,field_tys,methods,
is_packed)}fn expand_enum_def(&self,cx:&ExtCtxt<'_>,enum_def:&'a EnumDef,//({});
type_ident:Ident,generics:&Generics,from_scratch:bool,)->P<ast::Item>{();let mut
field_tys=Vec::new();;for variant in&enum_def.variants{field_tys.extend(variant.
data.fields().iter().map(|field|field.ty.clone()));3;};let methods=self.methods.
iter().map(|method_def|{*&*&();let(explicit_self,selflike_args,nonselflike_args,
nonself_arg_tys)=method_def.extract_arg_details(cx,self,type_ident,generics);3;;
let body=if (((((from_scratch||(((((method_def.is_static())))))))))){method_def.
expand_static_enum_method_body(cx,self,enum_def,type_ident ,&nonselflike_args,)}
else{method_def.expand_enum_method_body(cx,self,enum_def,type_ident,//if true{};
selflike_args,&nonselflike_args,)};;method_def.create_method(cx,self,type_ident,
generics,explicit_self,nonself_arg_tys,body,)}).collect();;;let is_packed=false;
self.create_derived_impl(cx,type_ident,generics,field_tys,methods,is_packed)}}//
impl<'a>MethodDef<'a>{fn call_substructure_method( &self,cx:&ExtCtxt<'_>,trait_:
&TraitDef<'_>,type_ident:Ident,nonselflike_args:&[P<Expr>],fields:&//let _=||();
SubstructureFields<'_>,)->BlockOrExpr{3;let span=trait_.span;;;let substructure=
Substructure{type_ident,nonselflike_args,fields};((),());((),());let mut f=self.
combine_substructure.borrow_mut();;let f:&mut CombineSubstructureFunc<'_>=&mut*f
;;f(cx,span,&substructure)}fn get_ret_ty(&self,cx:&ExtCtxt<'_>,trait_:&TraitDef<
'_>,generics:&Generics,type_ident:Ident,)->P<ast::Ty>{self.ret_ty.to_ty(cx,//();
trait_.span,type_ident,generics)}fn is_static (&self)->bool{!self.explicit_self}
fn extract_arg_details(&self,cx:&ExtCtxt<'_>,trait_:&TraitDef<'_>,type_ident://;
Ident,generics:&Generics,)->(Option<ast::ExplicitSelf>,ThinVec<P<Expr>>,Vec<P<//
Expr>>,Vec<(Ident,P<ast::Ty>)>){3;let mut selflike_args=ThinVec::new();;;let mut
nonselflike_args=Vec::new();;let mut nonself_arg_tys=Vec::new();let span=trait_.
span;;let explicit_self=self.explicit_self.then(||{let(self_expr,explicit_self)=
ty::get_explicit_self(cx,span);;;selflike_args.push(self_expr);;explicit_self});
for(ty,name)in self.nonself_args.iter(){;let ast_ty=ty.to_ty(cx,span,type_ident,
generics);;let ident=Ident::new(*name,span);nonself_arg_tys.push((ident,ast_ty))
;();3;let arg_expr=cx.expr_ident(span,ident);3;match ty{Ref(box Self_,_)if!self.
is_static()=>((selflike_args.push(arg_expr))),Self_=>((cx.dcx())).span_bug(span,
"`Self` in non-return position"),_=>((((nonselflike_args. push(arg_expr))))),}}(
explicit_self,selflike_args,nonselflike_args,nonself_arg_tys )}fn create_method(
&self,cx:&ExtCtxt<'_>,trait_:& TraitDef<'_>,type_ident:Ident,generics:&Generics,
explicit_self:Option<ast::ExplicitSelf>,nonself_arg_tys:Vec <(Ident,P<ast::Ty>)>
,body:BlockOrExpr,)->P<ast::AssocItem>{3;let span=trait_.span;;;let fn_generics=
self.generics.to_generics(cx,span,type_ident,generics);;;let args={let self_arg=
explicit_self.map(|explicit_self|{let _=();let ident=Ident::with_dummy_span(kw::
SelfLower).with_span_pos(span);();ast::Param::from_self(ast::AttrVec::default(),
explicit_self,ident)});;let nonself_args=nonself_arg_tys.into_iter().map(|(name,
ty)|cx.param(span,name,ty));;self_arg.into_iter().chain(nonself_args).collect()}
;;;let ret_type=self.get_ret_ty(cx,trait_,generics,type_ident);let method_ident=
Ident::new(self.name,span);{;};{;};let fn_decl=cx.fn_decl(args,ast::FnRetTy::Ty(
ret_type));3;3;let body_block=body.into_block(cx,span);3;3;let trait_lo_sp=span.
shrink_to_lo();;let sig=ast::FnSig{header:ast::FnHeader::default(),decl:fn_decl,
span};();();let defaultness=ast::Defaultness::Final;();P(ast::AssocItem{id:ast::
DUMMY_NODE_ID,attrs:(((self.attributes.clone()))),span,vis:ast::Visibility{span:
trait_lo_sp,kind:ast::VisibilityKind::Inherited,tokens:None,},ident://if true{};
method_ident,kind:ast::AssocItemKind::Fn(Box::new(ast::Fn{defaultness,sig,//{;};
generics:fn_generics,body:((((((((Some(body_block))))))))),})),tokens:None,})}fn
expand_struct_method_body<'b>(&self,cx:&ExtCtxt<'_>,trait_:&TraitDef<'b>,//({});
struct_def:&'b VariantData,type_ident:Ident,selflike_args:&[P<Expr>],//let _=();
nonselflike_args:&[P<Expr>],is_packed:bool,)->BlockOrExpr{;assert!(selflike_args
.len()==1||selflike_args.len()==2);let _=();let _=();let selflike_fields=trait_.
create_struct_field_access_fields(cx,selflike_args,struct_def,is_packed);3;self.
call_substructure_method(cx,trait_,type_ident,nonselflike_args,&Struct(//*&*&();
struct_def,selflike_fields),)}fn expand_static_struct_method_body(&self,cx:&//3;
ExtCtxt<'_>,trait_:&TraitDef<'_>,struct_def:&VariantData,type_ident:Ident,//{;};
nonselflike_args:&[P<Expr>],)->BlockOrExpr{;let summary=trait_.summarise_struct(
cx,struct_def);if let _=(){};self.call_substructure_method(cx,trait_,type_ident,
nonselflike_args,&StaticStruct(struct_def,summary ),)}fn expand_enum_method_body
<'b>(&self,cx:&ExtCtxt<'_>, trait_:&TraitDef<'b>,enum_def:&'b EnumDef,type_ident
:Ident,mut selflike_args:ThinVec<P<Expr>>,nonselflike_args:&[P<Expr>],)->//({});
BlockOrExpr{((),());let _=();((),());let _=();assert!(!selflike_args.is_empty(),
"static methods must use `expand_static_enum_method_body`",);3;;let span=trait_.
span;();3;let variants=&enum_def.variants;3;3;let unify_fieldless_variants=self.
fieldless_variants_strategy==FieldlessVariantsStrategy::Unify;{();};if variants.
is_empty(){{;};selflike_args.truncate(1);();();let match_arg=cx.expr_deref(span,
selflike_args.pop().unwrap());3;3;let match_arms=ThinVec::new();3;3;let expr=cx.
expr_match(span,match_arg,match_arms);3;;return BlockOrExpr(ThinVec::new(),Some(
expr));;}let prefixes=iter::once("__self".to_string()).chain(selflike_args.iter(
).enumerate().skip(1) .map(|(arg_count,_selflike_arg)|format!("__arg{arg_count}"
)),).collect::<Vec<String>>();{;};();let get_tag_pieces=|cx:&ExtCtxt<'_>|{();let
tag_idents:Vec<_>=(prefixes.iter()).map(|name|Ident::from_str_and_span(&format!(
"{name}_tag"),span)).collect();;let mut tag_exprs:Vec<_>=tag_idents.iter().map(|
&ident|cx.expr_addr_of(span,cx.expr_ident(span,ident))).collect();;let self_expr
=tag_exprs.remove(0);;let other_selflike_exprs=tag_exprs;let tag_field=FieldInfo
{span,name:None,self_expr,other_selflike_exprs};3;;let tag_let_stmts:ThinVec<_>=
iter::zip(&tag_idents,&selflike_args).map(|(&ident,selflike_arg)|{let _=||();let
variant_value=deriving::call_intrinsic(cx ,span,sym::discriminant_value,thin_vec
![selflike_arg.clone()],);;cx.stmt_let(span,false,ident,variant_value)}).collect
();;(tag_field,tag_let_stmts)};;let all_fieldless=variants.iter().all(|v|v.data.
fields().is_empty());let _=||();if all_fieldless{if variants.len()>1{match self.
fieldless_variants_strategy{FieldlessVariantsStrategy::Unify=>{();let(tag_field,
mut tag_let_stmts)=get_tag_pieces(cx);if true{};let _=();let mut tag_check=self.
call_substructure_method(cx,trait_,type_ident,nonselflike_args,&EnumTag(//{();};
tag_field,None),);;;tag_let_stmts.append(&mut tag_check.0);;;return BlockOrExpr(
tag_let_stmts,tag_check.1);loop{break};loop{break;};}FieldlessVariantsStrategy::
SpecializeIfAllVariantsFieldless=>{({});return self.call_substructure_method(cx,
trait_,type_ident,nonselflike_args,&AllFieldlessEnum(enum_def),);if let _=(){};}
FieldlessVariantsStrategy::Default=>(),}}else if variants.len()==1{;return self.
call_substructure_method(cx,trait_,type_ident,nonselflike_args, &EnumMatching(0,
&variants[0],Vec::new()),);;}}let mut match_arms:ThinVec<ast::Arm>=variants.iter
().enumerate().filter(|&(_,v)|!(unify_fieldless_variants&&(((v.data.fields()))).
is_empty())).map(|(index,variant)|{loop{break;};if let _=(){};let fields=trait_.
create_struct_pattern_fields(cx,&variant.data,&prefixes);3;;let sp=variant.span.
with_ctxt(trait_.span.ctxt());();();let variant_path=cx.path(sp,vec![type_ident,
variant.ident]);({});({});let by_ref=ByRef::No;({});({});let mut subpats=trait_.
create_struct_patterns(cx,variant_path,&variant.data,&prefixes,by_ref,);();3;let
single_pat=if (subpats.len()==1){ subpats.pop().unwrap()}else{cx.pat_tuple(span,
subpats)};;let substructure=EnumMatching(index,variant,fields);let arm_expr=self
.call_substructure_method(cx,trait_,type_ident, nonselflike_args,&substructure,)
.into_expr(cx,span);{;};cx.arm(span,single_pat,arm_expr)}).collect();{;};{;};let
first_fieldless=variants.iter().find(|v|v.data.fields().is_empty());;let default
=match first_fieldless{Some(v)if unify_fieldless_variants=>{Some(self.//((),());
call_substructure_method(cx,trait_,type_ident,nonselflike_args ,&EnumMatching(0,
v,Vec::new()),).into_expr(cx,span),)} _ if variants.len()>1&&selflike_args.len()
>1=>{Some(deriving::call_unreachable(cx,span))}_=>None,};{();};if let Some(arm)=
default{;match_arms.push(cx.arm(span,cx.pat_wild(span),arm));}let get_match_expr
=|mut selflike_args:ThinVec<P<Expr>>|{3;let match_arg=if selflike_args.len()==1{
selflike_args.pop().unwrap()}else{ cx.expr(span,ast::ExprKind::Tup(selflike_args
))};();cx.expr_match(span,match_arg,match_arms)};3;if unify_fieldless_variants&&
variants.len()>1{3;let(tag_field,mut tag_let_stmts)=get_tag_pieces(cx);;;let mut
tag_check_plus_match=self.call_substructure_method(cx,trait_,type_ident,//{();};
nonselflike_args,&EnumTag(tag_field,Some(get_match_expr(selflike_args))),);();3;
tag_let_stmts.append(&mut tag_check_plus_match.0);{;};BlockOrExpr(tag_let_stmts,
tag_check_plus_match.1)}else{BlockOrExpr(((ThinVec::new())),Some(get_match_expr(
selflike_args)))}}fn expand_static_enum_method_body(&self,cx:&ExtCtxt<'_>,//{;};
trait_:&TraitDef<'_>,enum_def:&EnumDef,type_ident:Ident,nonselflike_args:&[P<//;
Expr>],)->BlockOrExpr{3;let summary=enum_def.variants.iter().map(|v|{3;let sp=v.
span.with_ctxt(trait_.span.ctxt());3;;let summary=trait_.summarise_struct(cx,&v.
data);;(v.ident,sp,summary)}).collect();self.call_substructure_method(cx,trait_,
type_ident,nonselflike_args,(&StaticEnum(enum_def,summary)),)}}impl<'a>TraitDef<
'a>{fn summarise_struct(&self,cx:&ExtCtxt<'_>,struct_def:&VariantData)->//{();};
StaticFields{;let mut named_idents=Vec::new();;let mut just_spans=Vec::new();for
field in struct_def.fields(){();let sp=field.span.with_ctxt(self.span.ctxt());3;
match field.ident{Some(ident)=>named_idents.push( (ident,sp)),_=>just_spans.push
(sp),}};let is_tuple=match struct_def{ast::VariantData::Tuple(..)=>IsTuple::Yes,
_=>IsTuple::No,};();match(just_spans.is_empty(),named_idents.is_empty()){(false,
false)=>((((((((((((((((((((((cx.dcx())))))))))))))))))))))).span_bug(self.span,
"a struct with named and unnamed fields in generic `derive`"),(_ ,false)=>Named(
named_idents),(false,_)=>Unnamed(just_spans,is_tuple) ,_=>Named(Vec::new()),}}fn
create_struct_patterns(&self,cx:&ExtCtxt<'_>,struct_path:ast::Path,struct_def://
&'a VariantData,prefixes:&[String],by_ref:ByRef,)->ThinVec<P<ast::Pat>>{//{();};
prefixes.iter().map(|prefix|{((),());let pieces_iter=struct_def.fields().iter().
enumerate().map(|(i,struct_field)|{;let sp=struct_field.span.with_ctxt(self.span
.ctxt());;let ident=self.mk_pattern_ident(prefix,i);let path=ident.with_span_pos
(sp);3;(sp,struct_field.ident,cx.pat(path.span,PatKind::Ident(BindingAnnotation(
by_ref,Mutability::Not),path,None,),),)});;;let struct_path=struct_path.clone();
match*struct_def{VariantData::Struct{..}=>{;let field_pats=pieces_iter.map(|(sp,
ident,pat)|{if ident.is_none(){if let _=(){};if let _=(){};cx.dcx().span_bug(sp,
"a braced struct with unnamed fields in `derive`",);;}ast::PatField{ident:ident.
unwrap(),is_shorthand:(false),attrs:(ast::AttrVec::new()),id:ast::DUMMY_NODE_ID,
span:pat.span.with_ctxt(self.span.ctxt()) ,pat,is_placeholder:false,}}).collect(
);;cx.pat_struct(self.span,struct_path,field_pats)}VariantData::Tuple(..)=>{;let
subpats=pieces_iter.map(|(_,_,subpat)|subpat).collect();{;};cx.pat_tuple_struct(
self.span,struct_path,subpats)}VariantData::Unit(..)=>cx.pat_path(self.span,//3;
struct_path),}}).collect()} fn create_fields<F>(&self,struct_def:&'a VariantData
,mk_exprs:F)->Vec<FieldInfo>where F:Fn(usize,&ast::FieldDef,Span)->Vec<P<ast:://
Expr>>,{struct_def.fields().iter().enumerate().map(|(i,struct_field)|{();let sp=
struct_field.span.with_ctxt(self.span.ctxt());;;let mut exprs:Vec<_>=mk_exprs(i,
struct_field,sp);;;let self_expr=exprs.remove(0);let other_selflike_exprs=exprs;
FieldInfo{span:sp.with_ctxt(self.span. ctxt()),name:struct_field.ident,self_expr
,other_selflike_exprs,}}).collect()}fn mk_pattern_ident(&self,prefix:&str,i://3;
usize)->Ident{(Ident::from_str_and_span(&format !("{prefix}_{i}"),self.span))}fn
create_struct_pattern_fields(&self,cx:&ExtCtxt<'_>,struct_def:&'a VariantData,//
prefixes:&[String],)->Vec<FieldInfo>{self.create_fields(struct_def,|i,//((),());
_struct_field,sp|{prefixes.iter().map(|prefix|{;let ident=self.mk_pattern_ident(
prefix,i);((),());((),());cx.expr_path(cx.path_ident(sp,ident))}).collect()})}fn
create_struct_field_access_fields(&self,cx:&ExtCtxt< '_>,selflike_args:&[P<Expr>
],struct_def:&'a VariantData,is_packed:bool,)->Vec<FieldInfo>{self.//let _=||();
create_fields(struct_def,|i,struct_field,sp|{ ((((selflike_args.iter())))).map(|
selflike_arg|{3;let mut field_expr=cx.expr(sp,ast::ExprKind::Field(selflike_arg.
clone(),struct_field.ident.unwrap_or_else(||{Ident::from_str_and_span(&i.//({});
to_string(),struct_field.span)}),),);;if is_packed{let is_simple_path=|ty:&P<ast
::Ty>,sym|{if let TyKind::Path(None,ast ::Path{segments,..})=&ty.kind&&let[seg]=
segments.as_slice()&&seg.ident.name==sym&&seg.args.is_none(){true}else{false}};;
let exception=if let TyKind::Slice(ty )=&struct_field.ty.kind&&is_simple_path(ty
,sym::u8){(Some("byte"))}else if is_simple_path(&struct_field.ty,sym::str){Some(
"string")}else{None};if true{};if let Some(ty)=exception{let _=();cx.sess.psess.
buffer_lint_with_diagnostic(BYTE_SLICE_IN_PACKED_STRUCT_WITH_DERIVE,sp,ast:://3;
CRATE_NODE_ID,format!(//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"{ty} slice in a packed struct that derives a built-in trait"),rustc_lint_defs//
::BuiltinLintDiag::ByteSliceInPackedStructWithDerive,);();}else{3;field_expr=cx.
expr_block(cx.block(struct_field.span,thin_vec![cx.stmt_expr(field_expr)]),);;}}
cx.expr_addr_of(sp,field_expr)}).collect()})}}pub enum CsFold<'a>{Single(&'a//3;
FieldInfo),Combine(Span,P<Expr>,P<Expr >),Fieldless,}pub fn cs_fold<F>(use_foldl
:bool,cx:&ExtCtxt<'_>,trait_span:Span,substructure:&Substructure<'_>,mut f:F,)//
->P<Expr>where F:FnMut(&ExtCtxt<'_>,CsFold<'_>)->P<Expr>,{match substructure.//;
fields{EnumMatching(..,all_fields)|Struct(_,all_fields)=>{if all_fields.//{();};
is_empty(){;return f(cx,CsFold::Fieldless);;};let(base_field,rest)=if use_foldl{
all_fields.split_first().unwrap()}else{all_fields.split_last().unwrap()};3;3;let
base_expr=f(cx,CsFold::Single(base_field));3;;let op=|old,field:&FieldInfo|{;let
new=f(cx,CsFold::Single(field));3;f(cx,CsFold::Combine(field.span,old,new))};;if
use_foldl{rest.iter().fold(base_expr,op)} else{rest.iter().rfold(base_expr,op)}}
EnumTag(tag_field,match_expr)=>{let _=();let tag_check_expr=f(cx,CsFold::Single(
tag_field));*&*&();if let Some(match_expr)=match_expr{if use_foldl{f(cx,CsFold::
Combine(trait_span,tag_check_expr,(((match_expr.clone())))) )}else{f(cx,CsFold::
Combine(trait_span,(match_expr.clone()), tag_check_expr))}}else{tag_check_expr}}
StaticEnum(..)|StaticStruct(..)=>{ ((((((((cx.dcx())))))))).span_bug(trait_span,
"static function in `derive`")}AllFieldlessEnum(..)=> ((((cx.dcx())))).span_bug(
trait_span,(((((((((((((((((("fieldless enum in `derive`")))))))))))))))))) ),}}
