use syntax::ast::{ItemKind, Mutability, Stmt, Ty, TyKind, Unsafety};
use syntax::ast::{self, Arg, Attribute, Expr, FnHeader, Generics, Ident};
use syntax::attr::check_builtin_macro_attribute;
use syntax::ext::allocator::{AllocatorKind, AllocatorMethod, AllocatorTy, ALLOCATOR_METHODS};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ext::hygiene::SyntaxContext;
use syntax::ptr::P;
use syntax::symbol::{kw, sym, Symbol};
use syntax_pos::Span;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::global_allocator);

    let not_static = |item: Annotatable| {
        ecx.parse_sess.span_diagnostic.span_err(item.span(), "allocators must be statics");
        vec![item]
    };
    let item = match item {
        Annotatable::Item(item) => match item.node {
            ItemKind::Static(..) => item,
            _ => return not_static(Annotatable::Item(item)),
        }
        _ => return not_static(item),
    };

    // Generate a bunch of new items using the AllocFnFactory
    let span = item.span.with_ctxt(SyntaxContext::empty().apply_mark(ecx.current_expansion.id));
    let f = AllocFnFactory {
        span,
        kind: AllocatorKind::Global,
        global: item.ident,
        cx: ecx,
    };

    // Generate item statements for the allocator methods.
    let stmts = ALLOCATOR_METHODS.iter().map(|method| f.allocator_fn(method)).collect();

    // Generate anonymous constant serving as container for the allocator methods.
    let const_ty = ecx.ty(span, TyKind::Tup(Vec::new()));
    let const_body = ecx.expr_block(ecx.block(span, stmts));
    let const_item =
        ecx.item_const(span, Ident::with_empty_ctxt(kw::Underscore), const_ty, const_body);

    // Return the original item and the new methods.
    vec![Annotatable::Item(item), Annotatable::Item(const_item)]
}

struct AllocFnFactory<'a, 'b> {
    span: Span,
    kind: AllocatorKind,
    global: Ident,
    cx: &'b ExtCtxt<'a>,
}

impl AllocFnFactory<'_, '_> {
    fn allocator_fn(&self, method: &AllocatorMethod) -> Stmt {
        let mut abi_args = Vec::new();
        let mut i = 0;
        let ref mut mk = || {
            let name = Ident::from_str(&format!("arg{}", i));
            i += 1;
            name
        };
        let args = method
            .inputs
            .iter()
            .map(|ty| self.arg_ty(ty, &mut abi_args, mk))
            .collect();
        let result = self.call_allocator(method.name, args);
        let (output_ty, output_expr) = self.ret_ty(&method.output, result);
        let kind = ItemKind::Fn(
            self.cx.fn_decl(abi_args, ast::FunctionRetTy::Ty(output_ty)),
            FnHeader {
                unsafety: Unsafety::Unsafe,
                ..FnHeader::default()
            },
            Generics::default(),
            self.cx.block_expr(output_expr),
        );
        let item = self.cx.item(
            self.span,
            Ident::from_str(&self.kind.fn_name(method.name)),
            self.attrs(),
            kind,
        );
        self.cx.stmt_item(self.span, item)
    }

    fn call_allocator(&self, method: &str, mut args: Vec<P<Expr>>) -> P<Expr> {
        let method = self.cx.std_path(&[
            Symbol::intern("alloc"),
            Symbol::intern("GlobalAlloc"),
            Symbol::intern(method),
        ]);
        let method = self.cx.expr_path(self.cx.path(self.span, method));
        let allocator = self.cx.path_ident(self.span, self.global);
        let allocator = self.cx.expr_path(allocator);
        let allocator = self.cx.expr_addr_of(self.span, allocator);
        args.insert(0, allocator);

        self.cx.expr_call(self.span, method, args)
    }

    fn attrs(&self) -> Vec<Attribute> {
        let special = sym::rustc_std_internal_symbol;
        let special = self.cx.meta_word(self.span, special);
        vec![self.cx.attribute(self.span, special)]
    }

    fn arg_ty(
        &self,
        ty: &AllocatorTy,
        args: &mut Vec<Arg>,
        ident: &mut dyn FnMut() -> Ident,
    ) -> P<Expr> {
        match *ty {
            AllocatorTy::Layout => {
                let usize = self.cx.path_ident(self.span, Ident::with_empty_ctxt(sym::usize));
                let ty_usize = self.cx.ty_path(usize);
                let size = ident();
                let align = ident();
                args.push(self.cx.arg(self.span, size, ty_usize.clone()));
                args.push(self.cx.arg(self.span, align, ty_usize));

                let layout_new = self.cx.std_path(&[
                    Symbol::intern("alloc"),
                    Symbol::intern("Layout"),
                    Symbol::intern("from_size_align_unchecked"),
                ]);
                let layout_new = self.cx.expr_path(self.cx.path(self.span, layout_new));
                let size = self.cx.expr_ident(self.span, size);
                let align = self.cx.expr_ident(self.span, align);
                let layout = self.cx.expr_call(self.span, layout_new, vec![size, align]);
                layout
            }

            AllocatorTy::Ptr => {
                let ident = ident();
                args.push(self.cx.arg(self.span, ident, self.ptr_u8()));
                let arg = self.cx.expr_ident(self.span, ident);
                self.cx.expr_cast(self.span, arg, self.ptr_u8())
            }

            AllocatorTy::Usize => {
                let ident = ident();
                args.push(self.cx.arg(self.span, ident, self.usize()));
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
        let usize = self.cx.path_ident(self.span, Ident::with_empty_ctxt(sym::usize));
        self.cx.ty_path(usize)
    }

    fn ptr_u8(&self) -> P<Ty> {
        let u8 = self.cx.path_ident(self.span, Ident::with_empty_ctxt(sym::u8));
        let ty_u8 = self.cx.ty_path(u8);
        self.cx.ty_ptr(self.span, ty_u8, Mutability::Mutable)
    }
}
