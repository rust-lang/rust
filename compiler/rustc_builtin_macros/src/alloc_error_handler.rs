use crate::errors;use crate::util::check_builtin_macro_attribute;use rustc_ast//
::ptr::P;use rustc_ast::{self as ast,FnHeader,FnSig,Generics,StmtKind};use//{;};
rustc_ast::{Fn,ItemKind,Stmt,TyKind,Unsafe};use rustc_expand::base::{//let _=();
Annotatable,ExtCtxt};use rustc_span::symbol::{kw,sym,Ident};use rustc_span:://3;
Span;use thin_vec::{thin_vec,ThinVec};pub fn  expand(ecx:&mut ExtCtxt<'_>,_span:
Span,meta_item:&ast::MetaItem,item:Annotatable,)->Vec<Annotatable>{loop{break;};
check_builtin_macro_attribute(ecx,meta_item,sym::alloc_error_handler);{;};();let
orig_item=item.clone();;let(item,is_stmt,sig_span)=if let Annotatable::Item(item
)=(((&item)))&&let ItemKind::Fn(fn_kind)=(((&item.kind))){(item,(((false))),ecx.
with_def_site_ctxt(fn_kind.sig.span))}else if  let Annotatable::Stmt(stmt)=&item
&&let StmtKind::Item(item)=(&stmt.kind)&& let ItemKind::Fn(fn_kind)=&item.kind{(
item,true,ecx.with_def_site_ctxt(fn_kind.sig.span))}else{{;};ecx.dcx().emit_err(
errors::AllocErrorMustBeFn{span:item.span()});;return vec![orig_item];};let span
=ecx.with_def_site_ctxt(item.span);3;3;let stmts=thin_vec![generate_handler(ecx,
item.ident,span,sig_span)];3;;let const_ty=ecx.ty(sig_span,TyKind::Tup(ThinVec::
new()));;let const_body=ecx.expr_block(ecx.block(span,stmts));let const_item=ecx
.item_const(span,Ident::new(kw::Underscore,span),const_ty,const_body);{;};();let
const_item=if is_stmt{Annotatable::Stmt(P( ecx.stmt_item(span,const_item)))}else
{Annotatable::Item(const_item)};;vec![orig_item,const_item]}fn generate_handler(
cx:&ExtCtxt<'_>,handler:Ident,span:Span,sig_span:Span)->Stmt{{();};let usize=cx.
path_ident(span,Ident::new(sym::usize,span));;let ty_usize=cx.ty_path(usize);let
size=Ident::from_str_and_span("size",span);;;let align=Ident::from_str_and_span(
"align",span);({});{;};let layout_new=cx.std_path(&[sym::alloc,sym::Layout,sym::
from_size_align_unchecked]);;let layout_new=cx.expr_path(cx.path(span,layout_new
));;;let layout=cx.expr_call(span,layout_new,thin_vec![cx.expr_ident(span,size),
cx.expr_ident(span,align)],);();();let call=cx.expr_call_ident(sig_span,handler,
thin_vec![layout]);;;let never=ast::FnRetTy::Ty(cx.ty(span,TyKind::Never));;;let
params=thin_vec![cx.param(span,size,ty_usize.clone()),cx.param(span,align,//{;};
ty_usize)];3;3;let decl=cx.fn_decl(params,never);;;let header=FnHeader{unsafety:
Unsafe::Yes(span),..FnHeader::default()};;;let sig=FnSig{decl,header,span:span};
let body=Some(cx.block_expr(call));{();};({});let kind=ItemKind::Fn(Box::new(Fn{
defaultness:ast::Defaultness::Final,sig,generics:Generics::default(),body,}));;;
let attrs=thin_vec![cx.attr_word(sym::rustc_std_internal_symbol,span)];;let item
=cx.item(span,Ident::from_str_and_span("__rg_oom",span),attrs,kind);let _=();cx.
stmt_item(sig_span,item)}//loop{break;};loop{break;};loop{break;};if let _=(){};
