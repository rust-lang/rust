use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorMethod, AllocatorMethodInput, AllocatorTy, global_fn_name,
};
use rustc_ast::{
    self as ast, Expr, FnHeader, FnSig, Generics, ItemKind, Mutability, Param, Safety, Stmt,
    StmtKind, Ty,
};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::diagnostics;
use crate::util::check_builtin_macro_attribute;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::global_allocator);

    let orig_item = item.clone();

    // Allow using `#[global_allocator]` on an item statement
    // FIXME - if we get deref patterns, use them to reduce duplication here
    let (item, ident, is_stmt, ty_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Static(ast::StaticItem { ident, ty, .. }) = &item.kind
    {
        (item, *ident, false, ecx.with_def_site_ctxt(ty.span))
    } else if let Annotatable::Stmt(stmt) = &item
        && let StmtKind::Item(item) = &stmt.kind
        && let ItemKind::Static(ast::StaticItem { ident, ty, .. }) = &item.kind
    {
        (item, *ident, true, ecx.with_def_site_ctxt(ty.span))
    } else {
        ecx.dcx().emit_err(diagnostics::AllocMustStatics { span: item.span() });
        return vec![orig_item];
    };

    // Forbid `#[thread_local]` attributes on the item
    if let Some(attr) = item.attrs.iter().find(|x| x.has_name(sym::thread_local)) {
        ecx.dcx()
            .emit_err(diagnostics::AllocCannotThreadLocal { span: item.span, attr: attr.span });
        return vec![orig_item];
    }

    let span = ecx.with_def_site_ctxt(item.span);

    // Generate item statements for the allocator methods.
    let stmts = ALLOCATOR_METHODS
        .iter()
        .map(|method| allocator_fn(ecx, span, ty_span, ident, method))
        .collect();

    // Generate anonymous constant serving as container for the allocator methods.
    let const_item = ecx.item_const_underscore(span, ecx.block(span, stmts));
    let const_item = if is_stmt {
        Annotatable::Stmt(ecx.stmt_item(span, const_item))
    } else {
        Annotatable::Item(const_item)
    };

    // Return the original item and the new methods.
    vec![orig_item, const_item]
}

fn allocator_fn(
    cx: &ExtCtxt<'_>,
    span: Span,
    ty_span: Span,
    global: Ident,
    method: &AllocatorMethod,
) -> Stmt {
    let mut abi_args = ThinVec::new();
    let args = method.inputs.iter().map(|input| arg_ty(cx, span, input, &mut abi_args)).collect();
    let result = call_allocator(cx, ty_span, global, method.name, args);
    let output_ty = match method.output {
        AllocatorTy::ResultPtr => ptr_u8(cx, span),
        AllocatorTy::Unit => cx.ty_unit(span),
        AllocatorTy::Layout | AllocatorTy::Never | AllocatorTy::Usize | AllocatorTy::Ptr => {
            panic!("can't convert `AllocatorTy` to an output")
        }
    };
    let decl = cx.fn_decl(abi_args, ast::FnRetTy::Ty(output_ty));
    let header = FnHeader { safety: Safety::Unsafe(span), ..FnHeader::default() };
    let sig = FnSig { decl, header, span };
    let body = Some(cx.block_expr(result));
    let kind = ItemKind::Fn(cx.item_fn(
        sig,
        Ident::from_str_and_span(&global_fn_name(method.name), span),
        Generics::default(),
        body,
    ));
    let alloc_attr = match method.name {
        sym::alloc => sym::rustc_allocator,
        sym::dealloc => sym::rustc_deallocator,
        sym::realloc => sym::rustc_reallocator,
        sym::alloc_zeroed => sym::rustc_allocator_zeroed,
        _ => unreachable!("Unknown allocator method!"),
    };
    let attrs = thin_vec![
        cx.attr_word(sym::rustc_std_internal_symbol, span),
        cx.attr_word(alloc_attr, span)
    ];
    let item = cx.item(span, attrs, kind);
    cx.stmt_item(ty_span, item)
}

fn call_allocator(
    cx: &ExtCtxt<'_>,
    ty_span: Span,
    global: Ident,
    method: Symbol,
    mut args: ThinVec<Box<Expr>>,
) -> Box<Expr> {
    let method = cx.std_path(ty_span, &[sym::alloc, sym::GlobalAlloc, method]);
    let method = cx.expr_path(method);
    let allocator = cx.path_ident(ty_span, global);
    let allocator = cx.expr_path(allocator);
    let allocator = cx.expr_ref(ty_span, allocator);
    args.insert(0, allocator);

    cx.expr_call(ty_span, method, args)
}

fn arg_ty(
    cx: &ExtCtxt<'_>,
    span: Span,
    input: &AllocatorMethodInput,
    args: &mut ThinVec<Param>,
) -> Box<Expr> {
    match input.ty {
        AllocatorTy::Layout => {
            // If an allocator method is ever introduced having multiple
            // Layout arguments, these argument names need to be
            // disambiguated somehow. Currently the generated code would
            // fail to compile with "identifier is bound more than once in
            // this parameter list".

            let ty_usize = cx.ty_name(span, sym::usize);
            args.push(cx.param(span, sym::size, ty_usize));
            let ty_align = cx.ty_path(cx.std_path(span, &[sym::mem, sym::Alignment]));
            args.push(cx.param(span, sym::align, ty_align));

            let layout_new =
                cx.std_path(span, &[sym::alloc, sym::Layout, sym::from_size_alignment_unchecked]);
            let layout_new = cx.expr_path(layout_new);
            let size = cx.expr_name(span, sym::size);
            let align = cx.expr_name(span, sym::align);
            cx.expr_call(span, layout_new, thin_vec![size, align])
        }
        AllocatorTy::Ptr => {
            let name = Symbol::intern(input.name);
            args.push(cx.param(span, name, ptr_u8(cx, span)));
            cx.expr_name(span, name)
        }
        AllocatorTy::Usize => {
            let name = Symbol::intern(input.name);
            args.push(cx.param(span, name, cx.ty_name(span, sym::usize)));
            cx.expr_name(span, name)
        }
        AllocatorTy::Never | AllocatorTy::ResultPtr | AllocatorTy::Unit => {
            panic!("can't convert AllocatorTy to an argument")
        }
    }
}

fn ptr_u8(cx: &ExtCtxt<'_>, span: Span) -> Box<Ty> {
    let ty_u8 = cx.ty_name(span, sym::u8);
    cx.ty_ptr(span, ty_u8, Mutability::Mut)
}
