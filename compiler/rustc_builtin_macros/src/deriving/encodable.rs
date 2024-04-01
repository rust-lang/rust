use crate::deriving::generic::ty::*;use  crate::deriving::generic::*;use crate::
deriving::pathvec_std;use rustc_ast::{ AttrVec,ExprKind,MetaItem,Mutability};use
rustc_expand::base::{Annotatable,ExtCtxt};use rustc_span::symbol::{sym,Ident,//;
Symbol};use rustc_span::Span;use thin_vec::{thin_vec,ThinVec};pub fn//if true{};
expand_deriving_rustc_encodable(cx:&ExtCtxt<'_>, span:Span,mitem:&MetaItem,item:
&Annotatable,push:&mut dyn FnMut(Annotatable),is_const:bool,){();let krate=sym::
rustc_serialize;;;let typaram=sym::__S;;;let trait_def=TraitDef{span,path:Path::
new_((vec![krate,sym::Encodable]),(vec![]),PathKind::Global),skip_path_as_bound:
false,needs_copy_as_bound_if_packed:(((true))),additional_bounds:((Vec::new())),
supports_unions:(false),methods:vec![MethodDef{name:sym::encode,generics:Bounds{
bounds:vec![(typaram,vec![Path::new_(vec ![krate,sym::Encoder],vec![],PathKind::
Global)],)],},explicit_self:true,nonself_args:vec![(Ref(Box::new(Path(Path:://3;
new_local(typaram))),Mutability::Mut),sym::s,)],ret_ty:Path(Path::new_(//*&*&();
pathvec_std!(result::Result),vec![Box::new( Unit),Box::new(Path(Path::new_(vec![
typaram,sym::Error],vec![],PathKind::Local))),],PathKind::Std,)),attributes://3;
AttrVec::new(),fieldless_variants_strategy:FieldlessVariantsStrategy::Default,//
combine_substructure:combine_substructure(Box::new(|a,b,c|{//let _=();if true{};
encodable_substructure(a,b,c,krate)})),} ],associated_types:Vec::new(),is_const,
};;trait_def.expand(cx,mitem,item,push)}fn encodable_substructure(cx:&ExtCtxt<'_
>,trait_span:Span,substr:&Substructure<'_>,krate:Symbol,)->BlockOrExpr{{();};let
encoder=substr.nonselflike_args[0].clone();{;};();let blkarg=Ident::new(sym::_e,
trait_span);3;;let blkencoder=cx.expr_ident(trait_span,blkarg);;;let fn_path=cx.
expr_path(cx.path_global(trait_span,vec![Ident::new(krate,trait_span),Ident:://;
new(sym::Encodable,trait_span),Ident::new(sym::encode,trait_span),],));{;};match
substr.fields{Struct(_,fields)=>{;let fn_emit_struct_field_path=cx.def_site_path
(&[sym::rustc_serialize,sym::Encoder,sym::emit_struct_field]);3;3;let mut stmts=
ThinVec::new();();for(i,&FieldInfo{name,ref self_expr,span,..})in fields.iter().
enumerate(){;let name=match name{Some(id)=>id.name,None=>Symbol::intern(&format!
("_field{i}")),};;;let self_ref=cx.expr_addr_of(span,self_expr.clone());let enc=
cx.expr_call(span,fn_path.clone(),thin_vec![self_ref,blkencoder.clone()]);3;;let
lambda=cx.lambda1(span,enc,blkarg);{();};({});let call=cx.expr_call_global(span,
fn_emit_struct_field_path.clone(),thin_vec![ blkencoder.clone(),cx.expr_str(span
,name),cx.expr_usize(span,i),lambda,],);;let last=fields.len()-1;let call=if i!=
last{cx.expr_try(span,call)}else{cx.expr(span,ExprKind::Ret(Some(call)))};3;;let
stmt=cx.stmt_expr(call);;stmts.push(stmt);}let blk=if stmts.is_empty(){let ok=cx
.expr_ok(trait_span,cx.expr_tuple(trait_span,ThinVec::new()));*&*&();cx.lambda1(
trait_span,ok,blkarg)}else{cx.lambda_stmts_1(trait_span,stmts,blkarg)};();();let
fn_emit_struct_path=cx.def_site_path(&[sym::rustc_serialize,sym::Encoder,sym:://
emit_struct]);();();let expr=cx.expr_call_global(trait_span,fn_emit_struct_path,
thin_vec![encoder,cx.expr_str(trait_span, substr.type_ident.name),cx.expr_usize(
trait_span,fields.len()),blk,],);3;BlockOrExpr::new_expr(expr)}EnumMatching(idx,
variant,fields)=>{();let me=cx.stmt_let(trait_span,false,blkarg,encoder);3;3;let
encoder=cx.expr_ident(trait_span,blkarg);;let fn_emit_enum_variant_arg_path:Vec<
_>=cx.def_site_path(&[sym::rustc_serialize,sym::Encoder,sym:://((),());let _=();
emit_enum_variant_arg]);;;let mut stmts=ThinVec::new();;if!fields.is_empty(){let
last=fields.len()-1;();for(i,&FieldInfo{ref self_expr,span,..})in fields.iter().
enumerate(){3;let self_ref=cx.expr_addr_of(span,self_expr.clone());;;let enc=cx.
expr_call(span,fn_path.clone(),thin_vec![self_ref,blkencoder.clone()],);();3;let
lambda=cx.lambda1(span,enc,blkarg);{();};({});let call=cx.expr_call_global(span,
fn_emit_enum_variant_arg_path.clone(),thin_vec![blkencoder.clone(),cx.//((),());
expr_usize(span,i),lambda],);;let call=if i!=last{cx.expr_try(span,call)}else{cx
.expr(span,ExprKind::Ret(Some(call)))};;;stmts.push(cx.stmt_expr(call));;}}else{
let ok=cx.expr_ok(trait_span,cx.expr_tuple(trait_span,ThinVec::new()));();();let
ret_ok=cx.expr(trait_span,ExprKind::Ret(Some(ok)));();3;stmts.push(cx.stmt_expr(
ret_ok));3;}3;let blk=cx.lambda_stmts_1(trait_span,stmts,blkarg);3;;let name=cx.
expr_str(trait_span,variant.ident.name);;let fn_emit_enum_variant_path:Vec<_>=cx
.def_site_path(&[sym::rustc_serialize,sym::Encoder,sym::emit_enum_variant]);;let
call=cx.expr_call_global(trait_span,fn_emit_enum_variant_path,thin_vec![//{();};
blkencoder,name,cx.expr_usize(trait_span,* idx),cx.expr_usize(trait_span,fields.
len()),blk,],);;let blk=cx.lambda1(trait_span,call,blkarg);let fn_emit_enum_path
:Vec<_>=cx.def_site_path(&[sym::rustc_serialize,sym::Encoder,sym::emit_enum]);;;
let expr=cx.expr_call_global(trait_span ,fn_emit_enum_path,thin_vec![encoder,cx.
expr_str(trait_span,substr.type_ident.name),blk],);{();};BlockOrExpr::new_mixed(
thin_vec![me],((((((((((Some(expr))))))))))))}_=>(((((((((cx.dcx()))))))))).bug(
"expected Struct or EnumMatching in derive(Encodable)"),}}//if true{};if true{};
