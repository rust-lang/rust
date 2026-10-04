use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorMethod, AllocatorMethodInput, AllocatorTy, global_fn_name,
};
use rustc_ast::{
    self as ast, AttrVec, Expr, FnHeader, FnSig, Generics, ItemKind, Mutability, Param, Safety,
    Stmt, StmtKind, Ty,
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

    // Generate a bunch of new items using the AllocFnFactory
    let span = ecx.with_def_site_ctxt(item.span);
    let f = AllocFnFactory { span, ty_span, global: ident, cx: ecx };

    // Generate item statements for the allocator methods.
    let stmts = ALLOCATOR_METHODS.iter().map(|method| f.allocator_fn(method)).collect();

    // Generate anonymous constant serving as container for the allocator methods.
    let const_item = ecx.item_const_underscore(span, ecx.block(span, stmts));
    let const_item = if is_stmt {
        Annotatable::Stmt(Box::new(ecx.stmt_item(span, const_item)))
    } else {
        Annotatable::Item(const_item)
    };

    // Return the original item and the new methods.
    vec![orig_item, const_item]
}

struct AllocFnFactory<'a, 'b> {
    span: Span,
    ty_span: Span,
    global: Ident,
    cx: &'a ExtCtxt<'b>,
}

impl AllocFnFactory<'_, '_> {
    fn allocator_fn(&self, method: &AllocatorMethod) -> Stmt {
        let mut abi_args = ThinVec::new();
        let args = method.inputs.iter().map(|input| self.arg_ty(input, &mut abi_args)).collect();
        let result = self.call_allocator(method.name, args);
        let output_ty = self.ret_ty(&method.output);
        let decl = self.cx.fn_decl(abi_args, ast::FnRetTy::Ty(output_ty));
        let header = FnHeader { safety: Safety::Unsafe(self.span), ..FnHeader::default() };
        let sig = FnSig { decl, header, span: self.span };
        let body = Some(self.cx.block_expr(result));
        let kind = ItemKind::Fn(self.cx.item_fn(
            sig,
            Ident::from_str_and_span(&global_fn_name(method.name), self.span),
            Generics::default(),
            body,
        ));
        let item = self.cx.item(self.span, self.attrs(method), kind);
        self.cx.stmt_item(self.ty_span, item)
    }

    fn call_allocator(&self, method: Symbol, mut args: ThinVec<Box<Expr>>) -> Box<Expr> {
        let method = self.cx.std_path(&[sym::alloc, sym::GlobalAlloc, method]);
        let method = self.cx.expr_path(self.cx.path(self.ty_span, method));
        let allocator = self.cx.path_ident(self.ty_span, self.global);
        let allocator = self.cx.expr_path(allocator);
        let allocator = self.cx.expr_addr_of(self.ty_span, allocator);
        args.insert(0, allocator);

        self.cx.expr_call(self.ty_span, method, args)
    }

    fn attrs(&self, method: &AllocatorMethod) -> AttrVec {
        let alloc_attr = match method.name {
            sym::alloc => sym::rustc_allocator,
            sym::dealloc => sym::rustc_deallocator,
            sym::realloc => sym::rustc_reallocator,
            sym::alloc_zeroed => sym::rustc_allocator_zeroed,
            _ => unreachable!("Unknown allocator method!"),
        };
        thin_vec![
            self.cx.attr_word(sym::rustc_std_internal_symbol, self.span),
            self.cx.attr_word(alloc_attr, self.span)
        ]
    }

    fn arg_ty(&self, input: &AllocatorMethodInput, args: &mut ThinVec<Param>) -> Box<Expr> {
        match input.ty {
            AllocatorTy::Layout => {
                // If an allocator method is ever introduced having multiple
                // Layout arguments, these argument names need to be
                // disambiguated somehow. Currently the generated code would
                // fail to compile with "identifier is bound more than once in
                // this parameter list".

                let ty_usize = self.usize();
                args.push(self.cx.param(self.span, sym::size, ty_usize));
                let ty_align = self.ptr_alignment();
                args.push(self.cx.param(self.span, sym::align, ty_align));

                let layout_new = self.cx.std_path(&[
                    sym::alloc,
                    sym::Layout,
                    sym::from_size_alignment_unchecked,
                ]);
                let layout_new = self.cx.expr_path(self.cx.path(self.span, layout_new));
                let size = self.cx.expr_ident_sym(self.span, sym::size);
                let align = self.cx.expr_ident_sym(self.span, sym::align);
                let layout = self.cx.expr_call(self.span, layout_new, thin_vec![size, align]);
                layout
            }

            AllocatorTy::Ptr => {
                let name = Symbol::intern(input.name);
                args.push(self.cx.param(self.span, name, self.ptr_u8()));
                self.cx.expr_ident_sym(self.span, name)
            }

            AllocatorTy::Usize => {
                let name = Symbol::intern(input.name);
                args.push(self.cx.param(self.span, name, self.usize()));
                self.cx.expr_ident_sym(self.span, name)
            }

            AllocatorTy::Never | AllocatorTy::ResultPtr | AllocatorTy::Unit => {
                panic!("can't convert AllocatorTy to an argument")
            }
        }
    }

    fn ret_ty(&self, ty: &AllocatorTy) -> Box<Ty> {
        match *ty {
            AllocatorTy::ResultPtr => self.ptr_u8(),

            AllocatorTy::Unit => self.cx.ty_unit(self.span),

            AllocatorTy::Layout | AllocatorTy::Never | AllocatorTy::Usize | AllocatorTy::Ptr => {
                panic!("can't convert `AllocatorTy` to an output")
            }
        }
    }

    fn usize(&self) -> Box<Ty> {
        self.cx.ty_sym(self.span, sym::usize)
    }

    fn ptr_alignment(&self) -> Box<Ty> {
        let path = self.cx.std_path(&[sym::mem, sym::Alignment]);
        let path = self.cx.path(self.span, path);
        self.cx.ty_path(path)
    }

    fn ptr_u8(&self) -> Box<Ty> {
        let ty_u8 = self.cx.ty_sym(self.span, sym::u8);
        self.cx.ty_ptr(self.span, ty_u8, Mutability::Mut)
    }
}
