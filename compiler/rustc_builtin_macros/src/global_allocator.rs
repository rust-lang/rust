use crate::util::check_builtin_macro_attribute;

use rustc_ast::expand::allocator::{
    AllocatorKind, AllocatorMethod, AllocatorTy, ALLOCATOR_METHODS,
};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, AttrVec, Expr, FnHeader, FnSig, Generics, Param, StmtKind};
use rustc_ast::{Fn, ItemKind, Mutability, Stmt, Ty, TyKind, Unsafe};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::global_allocator);

    let orig_item = item.clone();
    let not_static = || {
        ecx.sess.parse_sess.span_diagnostic.span_err(item.span(), "allocators must be statics");
        vec![orig_item.clone()]
    };

    // Allow using `#[global_allocator]` on an item statement
    // FIXME - if we get deref patterns, use them to reduce duplication here
    let (item, is_stmt, ty_span) = match &item {
        Annotatable::Item(item) => match item.kind {
            ItemKind::Static(ref ty, ..) => (item, false, ecx.with_def_site_ctxt(ty.span)),
            _ => return not_static(),
        },
        Annotatable::Stmt(stmt) => match &stmt.kind {
            StmtKind::Item(item_) => match item_.kind {
                ItemKind::Static(ref ty, ..) => (item_, true, ecx.with_def_site_ctxt(ty.span)),
                _ => return not_static(),
            },
            _ => return not_static(),
        },
        _ => return not_static(),
    };

    // Generate a bunch of new items using the AllocFnFactory
    let span = ecx.with_def_site_ctxt(item.span);
    let f =
        AllocFnFactory { span, ty_span, kind: AllocatorKind::Global, global: item.ident, cx: ecx };

    // Generate item statements for the allocator methods.
    let stmts = ALLOCATOR_METHODS.iter().map(|method| f.allocator_fn(method)).collect();

    // Generate anonymous constant serving as container for the allocator methods.
    let const_ty = ecx.ty(ty_span, TyKind::Tup(Vec::new()));
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
    kind: AllocatorKind,
    global: Ident,
    cx: &'b ExtCtxt<'a>,
}

impl AllocFnFactory<'_, '_> {
    fn allocator_fn(&self, method: &AllocatorMethod) -> Stmt {
        let mut abi_args = Vec::new();
        let mut i = 0;
        let mut mk = || {
            let name = Ident::from_str_and_span(&format!("arg{}", i), self.span);
            i += 1;
            name
        };
        let args = method.inputs.iter().map(|ty| self.arg_ty(ty, &mut abi_args, &mut mk)).collect();
        let result = self.call_allocator(method.name, args);
        let (output_ty, output_expr) = self.ret_ty(&method.output, result);
        let decl = self.cx.fn_decl(abi_args, ast::FnRetTy::Ty(output_ty));
        let header = FnHeader { unsafety: Unsafe::Yes(self.span), ..FnHeader::default() };
        let sig = FnSig { decl, header, span: self.span };
        let body = Some(self.cx.block_expr(output_expr));
        let kind = ItemKind::Fn(Box::new(Fn {
            defaultness: ast::Defaultness::Final,
            sig,
            generics: Generics::default(),
            body,
        }));
        let item = self.cx.item(
            self.span,
            Ident::from_str_and_span(&self.kind.fn_name(method.name), self.span),
            self.attrs(),
            kind,
        );
        self.cx.stmt_item(self.ty_span, item)
    }

    fn call_allocator(&self, method: Symbol, mut args: Vec<P<Expr>>) -> P<Expr> {
        let method = self.cx.std_path(&[sym::alloc, sym::GlobalAlloc, method]);
        let method = self.cx.expr_path(self.cx.path(self.ty_span, method));
        let allocator = self.cx.path_ident(self.ty_span, self.global);
        let allocator = self.cx.expr_path(allocator);
        let allocator = self.cx.expr_addr_of(self.ty_span, allocator);
        args.insert(0, allocator);

        self.cx.expr_call(self.ty_span, method, args)
    }

    fn attrs(&self) -> AttrVec {
        let special = sym::rustc_std_internal_symbol;
        let special = self.cx.meta_word(self.span, special);
        thin_vec![self.cx.attribute(special)]
    }

    fn arg_ty(
        &self,
        ty: &AllocatorTy,
        args: &mut Vec<Param>,
        ident: &mut dyn FnMut() -> Ident,
    ) -> P<Expr> {
        match *ty {
            AllocatorTy::Layout => {
                let usize = self.cx.path_ident(self.span, Ident::new(sym::usize, self.span));
                let ty_usize = self.cx.ty_path(usize);
                let size = ident();
                let align = ident();
                args.push(self.cx.param(self.span, size, ty_usize.clone()));
                args.push(self.cx.param(self.span, align, ty_usize));

                let layout_new =
                    self.cx.std_path(&[sym::alloc, sym::Layout, sym::from_size_align_unchecked]);
                let layout_new = self.cx.expr_path(self.cx.path(self.span, layout_new));
                let size = self.cx.expr_ident(self.span, size);
                let align = self.cx.expr_ident(self.span, align);
                let layout = self.cx.expr_call(self.span, layout_new, vec![size, align]);
                layout
            }

            AllocatorTy::Ptr => {
                let ident = ident();
                args.push(self.cx.param(self.span, ident, self.ptr_u8()));
                let arg = self.cx.expr_ident(self.span, ident);
                self.cx.expr_cast(self.span, arg, self.ptr_u8())
            }

            AllocatorTy::Usize => {
                let ident = ident();
                args.push(self.cx.param(self.span, ident, self.usize()));
                self.cx.expr_ident(self.span, ident)
            }

            AllocatorTy::ResultPtr | AllocatorTy::Unit => {
                panic!("can't convert AllocatorTy to an argument")
            }
        }
    }

    fn ret_ty(&self, ty: &AllocatorTy, expr: P<Expr>) -> (P<Ty>, P<Expr>) {
        match *ty {
            AllocatorTy::ResultPtr => {
                // We're creating:
                //
                //      #expr as *mut u8

                let expr = self.cx.expr_cast(self.span, expr, self.ptr_u8());
                (self.ptr_u8(), expr)
            }

            AllocatorTy::Unit => (self.cx.ty(self.span, TyKind::Tup(Vec::new())), expr),

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
