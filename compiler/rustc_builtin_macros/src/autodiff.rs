#![allow(unused)]

use crate::errors;
//use crate::util::check_builtin_macro_attribute;
//use crate::util::check_autodiff;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, FnHeader, FnSig, Generics, StmtKind};
use rustc_ast::{Fn, ItemKind, Stmt, TyKind, Unsafe};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};
use rustc_span::Symbol;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    //check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);
    //check_builtin_macro_attribute(ecx, meta_item, sym::autodiff);

    dbg!(&meta_item);
    let input = item.clone();
    let orig_item: P<ast::Item> = item.clone().expect_item();
    let mut d_item: P<ast::Item> = item.clone().expect_item();

    // Allow using `#[autodiff(...)]` on a Fn
    let (fn_item, _ty_span)  = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(box ast::Fn { sig, .. }) = &item.kind
    {
        dbg!(&item);
        (item, ecx.with_def_site_ctxt(sig.span))
    } else {
        ecx.sess
            .dcx()
            .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![input];
    };
    let _x: &ItemKind = &fn_item.kind;
    d_item.ident.name =
        Symbol::intern(format!("d_{}", fn_item.ident.name).as_str());
    let orig_annotatable = Annotatable::Item(orig_item.clone());
    let d_annotatable = Annotatable::Item(d_item.clone());
    return vec![orig_annotatable, d_annotatable];
}

// #[rustc_std_internal_symbol]
// unsafe fn __rg_oom(size: usize, align: usize) -> ! {
//     handler(core::alloc::Layout::from_size_align_unchecked(size, align))
// }
//fn generate_handler(cx: &ExtCtxt<'_>, handler: Ident, span: Span, sig_span: Span) -> Stmt {
//    let usize = cx.path_ident(span, Ident::new(sym::usize, span));
//    let ty_usize = cx.ty_path(usize);
//    let size = Ident::from_str_and_span("size", span);
//    let align = Ident::from_str_and_span("align", span);
//
//    let layout_new = cx.std_path(&[sym::alloc, sym::Layout, sym::from_size_align_unchecked]);
//    let layout_new = cx.expr_path(cx.path(span, layout_new));
//    let layout = cx.expr_call(
//        span,
//        layout_new,
//        thin_vec![cx.expr_ident(span, size), cx.expr_ident(span, align)],
//    );
//
//    let call = cx.expr_call_ident(sig_span, handler, thin_vec![layout]);
//
//    let never = ast::FnRetTy::Ty(cx.ty(span, TyKind::Never));
//    let params = thin_vec![cx.param(span, size, ty_usize.clone()), cx.param(span, align, ty_usize)];
//    let decl = cx.fn_decl(params, never);
//    let header = FnHeader { unsafety: Unsafe::Yes(span), ..FnHeader::default() };
//    let sig = FnSig { decl, header, span: span };
//
//    let body = Some(cx.block_expr(call));
//    let kind = ItemKind::Fn(Box::new(Fn {
//        defaultness: ast::Defaultness::Final,
//        sig,
//        generics: Generics::default(),
//        body,
//    }));
//
//    let attrs = thin_vec![cx.attr_word(sym::rustc_std_internal_symbol, span)];
//
//    let item = cx.item(span, Ident::from_str_and_span("__rg_oom", span), attrs, kind);
//    cx.stmt_item(sig_span, item)
//}
