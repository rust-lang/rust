use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorMethod, AllocatorMethodInput, AllocatorTy, global_fn_name,
};
use rustc_ast::ptr::P;
use rustc_ast::{
    self as ast, AttrVec, Expr, Fn, FnHeader, FnSig, Generics, ItemKind, Mutability, Param, Safety,
    Stmt, StmtKind, Ty, TyKind,
};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::errors;
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
        && let ItemKind::Static(box ast::StaticItem { ident, ty, .. }) = &item.kind
    {
        (item, *ident, false, ecx.with_def_site_ctxt(ty.span))
    } else if let Annotatable::Stmt(stmt) = &item
        && let StmtKind::Item(item) = &stmt.kind
        && let ItemKind::Static(box ast::StaticItem { ident, ty, .. }) = &item.kind
    {
        (item, *ident, true, ecx.with_def_site_ctxt(ty.span))
    } else {
        ecx.dcx().emit_err(errors::AllocMustStatics { span: item.span() });
        return vec![orig_item];
    };

    // Generate a bunch of new items using the AllocFnFactory
    let span = ecx.with_def_site_ctxt(item.span);
    let f = AllocFnFactory { span, ty_span, global: ident, cx: ecx };

    // Generate item statements for the allocator methods.
    let stmts = ALLOCATOR_METHODS.iter().map(|method| f.allocator_fn(method)).collect();

    // Generate anonymous constant serving as container for the allocator methods.
    let const_ty = ecx.ty(ty_span, TyKind::Tup(ThinVec::new()));
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
        let kind = ItemKind::Fn(Box::new(Fn {
            defaultness: ast::Defaultness::Final,
            sig,
            ident: Ident::from_str_and_span(&global_fn_name(method.name), self.span),
            generics: Generics::default(),
            contract: None,
            body,
            define_opaque: None,
        }));
        let item = self.cx.item(self.span, self.attrs(), kind);
        self.cx.stmt_item(self.ty_span, item)
    }

    fn call_allocator(&self, method: Symbol, mut args: ThinVec<P<Expr>>) -> P<Expr> {
        let method = self.cx.std_path(&[sym::alloc, sym::GlobalAlloc, method]);
        let method = self.cx.expr_path(self.cx.path(self.ty_span, method));
        let allocator = self.cx.path_ident(self.ty_span, self.global);
        let allocator = self.cx.expr_path(allocator);
        let allocator = self.cx.expr_addr_of(self.ty_span, allocator);
        args.insert(0, allocator);

        self.cx.expr_call(self.ty_span, method, args)
    }

    fn attrs(&self) -> AttrVec {
        thin_vec![self.cx.attr_word(sym::rustc_std_internal_symbol, self.span)]
    }

    fn arg_ty(&self, input: &AllocatorMethodInput, args: &mut ThinVec<Param>) -> P<Expr> {
        match input.ty {
            AllocatorTy::Layout => {
                // If an allocator method is ever introduced having multiple
                // Layout arguments, these argument names need to be
                // disambiguated somehow. Currently the generated code would
                // fail to compile with "identifier is bound more than once in
                // this parameter list".
                let size = Ident::from_str_and_span("size", self.span);
                let align = Ident::from_str_and_span("align", self.span);

                let usize = self.cx.path_ident(self.span, Ident::new(sym::usize, self.span));
                let ty_usize = self.cx.ty_path(usize);
                args.push(self.cx.param(self.span, size, ty_usize.clone()));
                args.push(self.cx.param(self.span, align, ty_usize));

                let layout_new =
                    self.cx.std_path(&[sym::alloc, sym::Layout, sym::from_size_align_unchecked]);
                let layout_new = self.cx.expr_path(self.cx.path(self.span, layout_new));
                let size = self.cx.expr_ident(self.span, size);
                let align = self.cx.expr_ident(self.span, align);
                let layout = self.cx.expr_call(self.span, layout_new, thin_vec![size, align]);
                layout
            }

            AllocatorTy::Ptr => {
                let ident = Ident::from_str_and_span(input.name, self.span);
                args.push(self.cx.param(self.span, ident, self.ptr_u8()));
                self.cx.expr_ident(self.span, ident)
            }

            AllocatorTy::Usize => {
                let ident = Ident::from_str_and_span(input.name, self.span);
                args.push(self.cx.param(self.span, ident, self.usize()));
                self.cx.expr_ident(self.span, ident)
            }

            AllocatorTy::ResultPtr | AllocatorTy::Unit => {
                panic!("can't convert AllocatorTy to an argument")
            }
        }
    }

    fn ret_ty(&self, ty: &AllocatorTy) -> P<Ty> {
        match *ty {
            AllocatorTy::ResultPtr => self.ptr_u8(),

            AllocatorTy::Unit => self.cx.ty(self.span, TyKind::Tup(ThinVec::new())),

            AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                panic!("can't convert `AllocatorTy` to an output")
            }
        }
    }

    fn usize(&self) -> P<Ty> {
        let usize = self.cx.path_ident(self.span, Ident::new(sym::usize, self.span));
        self.cx.ty_path(usize)
    }

    fn ptr_u8(&self) -> P<Ty> {
        let u8 = self.cx.path_ident(self.span, Ident::new(sym::u8, self.span));
        let ty_u8 = self.cx.ty_path(u8);
        self.cx.ty_ptr(self.span, ty_u8, Mutability::Mut)
    }
}
