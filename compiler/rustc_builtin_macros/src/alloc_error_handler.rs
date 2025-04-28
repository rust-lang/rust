use rustc_ast::ptr::P;
use rustc_ast::{
    self as ast, Fn, FnHeader, FnSig, Generics, ItemKind, Safety, Stmt, StmtKind, TyKind,
};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::errors;
use crate::util::check_builtin_macro_attribute;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);

    let orig_item = item.clone();

    // Allow using `#[alloc_error_handler]` on an item statement
    // FIXME - if we get deref patterns, use them to reduce duplication here
    let (item, ident, is_stmt, sig_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(fn_kind) = &item.kind
    {
        (item, fn_kind.ident, false, ecx.with_def_site_ctxt(fn_kind.sig.span))
    } else if let Annotatable::Stmt(stmt) = &item
        && let StmtKind::Item(item) = &stmt.kind
        && let ItemKind::Fn(fn_kind) = &item.kind
    {
        (item, fn_kind.ident, true, ecx.with_def_site_ctxt(fn_kind.sig.span))
    } else {
        ecx.dcx().emit_err(errors::AllocErrorMustBeFn { span: item.span() });
        return vec![orig_item];
    };

    // Generate a bunch of new items using the AllocFnFactory
    let span = ecx.with_def_site_ctxt(item.span);

    // Generate item statements for the allocator methods.
    let stmts = thin_vec![generate_handler(ecx, ident, span, sig_span)];

    // Generate anonymous constant serving as container for the allocator methods.
    let const_ty = ecx.ty(sig_span, TyKind::Tup(ThinVec::new()));
    let const_body = ecx.expr_block(ecx.block(span, stmts));
    let const_item = ecx.item_const(span, Ident::new(kw::Underscore, span), const_ty, const_body);
    let const_item = if is_stmt {
        Annotatable::Stmt(P(ecx.stmt_item(span, const_item)))
    } else {
        Annotatable::Item(const_item)
    };

    // Return the original item and the new methods.
    vec![orig_item, const_item]
}

// #[rustc_std_internal_symbol]
// unsafe fn __rg_oom(size: usize, align: usize) -> ! {
//     handler(core::alloc::Layout::from_size_align_unchecked(size, align))
// }
fn generate_handler(cx: &ExtCtxt<'_>, handler: Ident, span: Span, sig_span: Span) -> Stmt {
    let usize = cx.path_ident(span, Ident::new(sym::usize, span));
    let ty_usize = cx.ty_path(usize);
    let size = Ident::from_str_and_span("size", span);
    let align = Ident::from_str_and_span("align", span);

    let layout_new = cx.std_path(&[sym::alloc, sym::Layout, sym::from_size_align_unchecked]);
    let layout_new = cx.expr_path(cx.path(span, layout_new));
    let layout = cx.expr_call(
        span,
        layout_new,
        thin_vec![cx.expr_ident(span, size), cx.expr_ident(span, align)],
    );

    let call = cx.expr_call_ident(sig_span, handler, thin_vec![layout]);

    let never = ast::FnRetTy::Ty(cx.ty(span, TyKind::Never));
    let params = thin_vec![cx.param(span, size, ty_usize.clone()), cx.param(span, align, ty_usize)];
    let decl = cx.fn_decl(params, never);
    let header = FnHeader { safety: Safety::Unsafe(span), ..FnHeader::default() };
    let sig = FnSig { decl, header, span };

    let body = Some(cx.block_expr(call));
    let kind = ItemKind::Fn(Box::new(Fn {
        defaultness: ast::Defaultness::Final,
        sig,
        ident: Ident::from_str_and_span("__rg_oom", span),
        generics: Generics::default(),
        contract: None,
        body,
        define_opaque: None,
    }));

    let attrs = thin_vec![cx.attr_word(sym::rustc_std_internal_symbol, span)];

    let item = cx.item(span, attrs, kind);
    cx.stmt_item(sig_span, item)
}
