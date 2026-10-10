use std::iter;

use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::util::literal;
use rustc_ast::{
    self as ast, AnonConst, AttrItem, AttrVec, BlockCheckMode, Expr, LocalKind, MatchKind, PatKind,
    Path, UnOp, attr, token, tokenstream,
};
use rustc_span::{DUMMY_SP, Ident, Span, Spanned, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::base::ExtCtxt;

impl<'a> ExtCtxt<'a> {
    pub fn std_path(&self, span: Span, components: &[Symbol]) -> Path {
        self.std_path_all(span, components, vec![])
    }
    pub fn std_path_all(
        &self,
        span: Span,
        components: &[Symbol],
        args: Vec<ast::GenericArg>,
    ) -> Path {
        let mut segments = ThinVec::with_capacity(components.len() + 1);
        let (last, rest) = components.split_last().expect("cannot construct empty path");
        segments.extend(
            iter::once(Ident::new(kw::DollarCrate, self.with_def_site_ctxt(DUMMY_SP)))
                .chain(rest.iter().copied().map(|name| Ident::new(name, span)))
                .map(|name| ast::PathSegment::from_ident(name)),
        );
        self.build_path(span, segments, Ident::new(*last, span), args)
    }

    pub fn path(&self, span: Span, strs: &[Ident]) -> Path {
        self.path_all(span, strs, vec![])
    }

    pub fn path_ident(&self, span: Span, id: Ident) -> Path {
        let segments = thin_vec![ast::PathSegment::from_ident(id.with_span_pos(span))];
        Path { span, segments }
    }

    pub fn path_sym(&self, span: Span, name: Symbol) -> Path {
        Path::from_ident(Ident::new(name, span))
    }

    pub fn path_all(&self, span: Span, idents: &[Ident], args: Vec<ast::GenericArg>) -> Path {
        let mut segments = ThinVec::with_capacity(idents.len());
        let (last, rest) = idents.split_last().expect("cannot construct empty path");
        segments.extend(
            rest.iter().map(|ident| ast::PathSegment::from_ident(ident.with_span_pos(span))),
        );
        self.build_path(span, segments, last.with_span_pos(span), args)
    }

    fn build_path(
        &self,
        span: Span,
        mut segments: ThinVec<ast::PathSegment>,
        last: Ident,
        args: Vec<ast::GenericArg>,
    ) -> Path {
        let args = if !args.is_empty() {
            let args = args.into_iter().map(ast::AngleBracketedArg::Arg).collect();
            Some(ast::AngleBracketedArgs { args, span }.into())
        } else {
            None
        };
        segments.push(ast::PathSegment { ident: last, id: ast::DUMMY_NODE_ID, args });
        Path { span, segments }
    }

    pub fn macro_call(
        &self,
        span: Span,
        path: Path,
        delim: Delimiter,
        tokens: TokenStream,
    ) -> Box<ast::MacCall> {
        Box::new(ast::MacCall {
            path,
            args: Box::new(ast::DelimArgs {
                dspan: tokenstream::DelimSpan { open: span, close: span },
                delim,
                tokens,
            }),
        })
    }

    pub fn ty(&self, span: Span, kind: ast::TyKind) -> Box<ast::Ty> {
        Box::new(ast::Ty { id: ast::DUMMY_NODE_ID, span, kind })
    }

    pub fn ty_infer(&self, span: Span) -> Box<ast::Ty> {
        self.ty(span, ast::TyKind::Infer)
    }

    pub fn ty_path(&self, path: Path) -> Box<ast::Ty> {
        self.ty(path.span, ast::TyKind::Path(None, path))
    }

    pub fn ty_ident(&self, span: Span, ident: Ident) -> Box<ast::Ty> {
        self.ty_path(self.path_ident(span, ident))
    }

    pub fn ty_sym(&self, span: Span, name: Symbol) -> Box<ast::Ty> {
        self.ty_path(self.path_sym(span, name))
    }

    pub fn anon_const(&self, span: Span, kind: ast::ExprKind) -> ast::AnonConst {
        ast::AnonConst {
            id: ast::DUMMY_NODE_ID,
            value: Box::new(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                kind,
                span,
                attrs: AttrVec::new(),
                tokens: None,
            }),
        }
    }

    pub fn anon_const_block(&self, b: Box<ast::Block>) -> Box<ast::AnonConst> {
        Box::new(self.anon_const(b.span, ast::ExprKind::Block(b, None)))
    }

    pub fn const_ident(&self, span: Span, ident: Ident) -> ast::AnonConst {
        self.anon_const(span, ast::ExprKind::Path(None, self.path_ident(span, ident)))
    }

    pub fn ty_ref(
        &self,
        span: Span,
        ty: Box<ast::Ty>,
        lifetime: Option<ast::Lifetime>,
        mutbl: ast::Mutability,
    ) -> Box<ast::Ty> {
        self.ty(span, ast::TyKind::Ref(lifetime, ty, mutbl))
    }

    pub fn ty_ptr(&self, span: Span, ty: Box<ast::Ty>, mutbl: ast::Mutability) -> Box<ast::Ty> {
        self.ty(span, ast::TyKind::Ptr(ty, mutbl))
    }

    pub fn ty_unit(&self, span: Span) -> Box<ast::Ty> {
        self.ty(span, ast::TyKind::Tup(ThinVec::new()))
    }

    pub fn ty_self(&self, span: Span) -> Box<ast::Ty> {
        self.ty_path(self.path_sym(span, kw::SelfUpper))
    }
    pub fn ty_self_ref(&self, span: Span) -> Box<ast::Ty> {
        self.ty_ref(span, self.ty_self(span), None, ast::Mutability::Not)
    }

    pub fn typaram(
        &self,
        ident: Ident,
        bounds: ast::GenericBounds,
        default: Option<Box<ast::Ty>>,
    ) -> ast::GenericParam {
        ast::GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: AttrVec::new(),
            bounds,
            kind: ast::GenericParamKind::Type { default },
            is_placeholder: false,
            colon_span: None,
        }
    }

    pub fn lifetime_param(&self, ident: Ident, bounds: ast::GenericBounds) -> ast::GenericParam {
        ast::GenericParam {
            id: ast::DUMMY_NODE_ID,
            ident,
            attrs: AttrVec::new(),
            bounds,
            is_placeholder: false,
            kind: ast::GenericParamKind::Lifetime,
            colon_span: None,
        }
    }

    pub fn const_param(
        &self,
        ident: Ident,
        bounds: ast::GenericBounds,
        ty: Box<ast::Ty>,
        default: Option<AnonConst>,
    ) -> ast::GenericParam {
        ast::GenericParam {
            id: ast::DUMMY_NODE_ID,
            ident,
            attrs: AttrVec::new(),
            bounds,
            is_placeholder: false,
            kind: ast::GenericParamKind::Const { ty, span: DUMMY_SP, default },
            colon_span: None,
        }
    }

    pub fn trait_ref(&self, path: Path) -> ast::TraitRef {
        ast::TraitRef { path, ref_id: ast::DUMMY_NODE_ID }
    }

    pub fn poly_trait_ref(&self, span: Span, path: Path, is_const: bool) -> ast::PolyTraitRef {
        ast::PolyTraitRef {
            bound_generic_params: ThinVec::new(),
            modifiers: ast::TraitBoundModifiers {
                polarity: ast::BoundPolarity::Positive,
                constness: if is_const {
                    ast::BoundConstness::Maybe(DUMMY_SP)
                } else {
                    ast::BoundConstness::Never
                },
                asyncness: ast::BoundAsyncness::Normal,
            },
            trait_ref: self.trait_ref(path),
            span,
            parens: ast::Parens::No,
        }
    }

    pub fn trait_bound(&self, path: Path, is_const: bool) -> ast::GenericBound {
        ast::GenericBound::Trait(self.poly_trait_ref(path.span, path, is_const))
    }

    pub fn lifetime(&self, span: Span, ident: Ident) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, ident: ident.with_span_pos(span) }
    }

    pub fn lifetime_static(&self, span: Span) -> ast::Lifetime {
        self.lifetime(span, Ident::new(kw::StaticLifetime, span))
    }

    pub fn stmt_expr(&self, expr: Box<ast::Expr>) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, span: expr.span, kind: ast::StmtKind::Expr(expr) }
    }

    pub fn stmt_let(&self, sp: Span, mutbl: bool, name: Symbol, ex: Box<ast::Expr>) -> ast::Stmt {
        self.stmt_let_ty(sp, mutbl, name, None, ex)
    }

    pub fn stmt_let_ty(
        &self,
        sp: Span,
        mutbl: bool,
        name: Symbol,
        ty: Option<Box<ast::Ty>>,
        ex: Box<ast::Expr>,
    ) -> ast::Stmt {
        let pat = if mutbl {
            self.pat_ident_binding_mode(sp, name, ast::BindingMode::MUT)
        } else {
            self.pat_ident(sp, name)
        };
        let local = Box::new(ast::Local {
            super_: None,
            pat: Box::new(pat),
            ty,
            id: ast::DUMMY_NODE_ID,
            kind: LocalKind::Init(ex),
            span: sp,
            colon_sp: None,
            attrs: AttrVec::new(),
            tokens: None,
        });
        self.stmt_local(local, sp)
    }

    /// Generates `let _: Type;`, which is usually used for type assertions.
    pub fn stmt_let_type_only(&self, span: Span, ty: Box<ast::Ty>) -> ast::Stmt {
        let local = Box::new(ast::Local {
            super_: None,
            pat: Box::new(self.pat_wild(span)),
            ty: Some(ty),
            id: ast::DUMMY_NODE_ID,
            kind: LocalKind::Decl,
            span,
            colon_sp: None,
            attrs: AttrVec::new(),
            tokens: None,
        });
        self.stmt_local(local, span)
    }

    pub fn stmt_local(&self, local: Box<ast::Local>, span: Span) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Let(local), span }
    }

    pub fn stmt_item(&self, sp: Span, item: Box<ast::Item>) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Item(item), span: sp }
    }

    pub fn block_expr(&self, expr: Box<ast::Expr>) -> Box<ast::Block> {
        self.block(
            expr.span,
            thin_vec![ast::Stmt {
                id: ast::DUMMY_NODE_ID,
                span: expr.span,
                kind: ast::StmtKind::Expr(expr),
            }],
        )
    }
    pub fn block(&self, span: Span, stmts: ThinVec<ast::Stmt>) -> Box<ast::Block> {
        Box::new(ast::Block { stmts, id: ast::DUMMY_NODE_ID, rules: BlockCheckMode::Default, span })
    }

    pub fn expr(&self, span: Span, kind: ast::ExprKind) -> Box<ast::Expr> {
        Box::new(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            kind,
            span,
            attrs: AttrVec::new(),
            tokens: None,
        })
    }

    pub fn expr_path(&self, path: Path) -> Box<ast::Expr> {
        self.expr(path.span, ast::ExprKind::Path(None, path))
    }

    pub fn expr_ident(&self, span: Span, id: Ident) -> Box<ast::Expr> {
        self.expr_path(self.path_ident(span, id))
    }
    pub fn expr_self(&self, span: Span) -> Box<ast::Expr> {
        self.expr_ident(span, Ident::new(kw::SelfLower, span))
    }
    pub fn expr_ident_sym(&self, span: Span, sym: Symbol) -> Box<ast::Expr> {
        self.expr_ident(span, Ident::new(sym, span))
    }

    pub fn expr_macro_call(&self, span: Span, call: Box<ast::MacCall>) -> Box<ast::Expr> {
        self.expr(span, ast::ExprKind::MacCall(call))
    }

    pub fn expr_binary(
        &self,
        sp: Span,
        op: ast::BinOpKind,
        lhs: Box<ast::Expr>,
        rhs: Box<ast::Expr>,
    ) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::Binary(Spanned { node: op, span: sp }, lhs, rhs))
    }

    pub fn expr_deref(&self, sp: Span, e: Box<ast::Expr>) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::Unary(UnOp::Deref, e))
    }

    pub fn expr_addr_of(&self, sp: Span, e: Box<ast::Expr>) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::AddrOf(ast::BorrowKind::Ref, ast::Mutability::Not, e))
    }

    pub fn expr_paren(&self, sp: Span, e: Box<ast::Expr>) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::Paren(e))
    }

    pub fn expr_method_call(
        &self,
        span: Span,
        expr: Box<ast::Expr>,
        ident: Ident,
        args: ThinVec<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        let seg = ast::PathSegment::from_ident(ident);
        self.expr(
            span,
            ast::ExprKind::MethodCall(Box::new(ast::MethodCall {
                seg,
                receiver: expr,
                args,
                span,
            })),
        )
    }

    pub fn expr_call(
        &self,
        span: Span,
        expr: Box<ast::Expr>,
        args: ThinVec<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(expr, args))
    }
    pub fn expr_call_ident(
        &self,
        span: Span,
        id: Ident,
        args: ThinVec<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(self.expr_ident(span, id), args))
    }
    pub fn expr_call_global(
        &self,
        sp: Span,
        fn_path: Path,
        args: ThinVec<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        let pathexpr = self.expr_path(fn_path);
        self.expr_call(sp, pathexpr, args)
    }
    pub fn expr_block(&self, b: Box<ast::Block>) -> Box<ast::Expr> {
        self.expr(b.span, ast::ExprKind::Block(b, None))
    }
    pub fn field_imm(&self, span: Span, ident: Ident, e: Box<ast::Expr>) -> ast::ExprField {
        ast::ExprField {
            ident: ident.with_span_pos(span),
            expr: e,
            span,
            is_shorthand: false,
            attrs: AttrVec::new(),
            id: ast::DUMMY_NODE_ID,
            is_placeholder: false,
        }
    }
    pub fn expr_struct(
        &self,
        span: Span,
        path: Path,
        fields: ThinVec<ast::ExprField>,
    ) -> Box<ast::Expr> {
        self.expr(
            span,
            ast::ExprKind::Struct(Box::new(ast::StructExpr {
                qself: None,
                path,
                fields,
                rest: ast::StructRest::None,
            })),
        )
    }
    pub fn expr_struct_ident(
        &self,
        span: Span,
        id: Ident,
        fields: ThinVec<ast::ExprField>,
    ) -> Box<ast::Expr> {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    pub fn expr_usize(&self, span: Span, n: usize) -> Box<ast::Expr> {
        let suffix = Some(ast::UintTy::Usize.name());
        let lit = token::Lit::new(token::Integer, sym::integer(n), suffix);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_u32(&self, span: Span, n: u32) -> Box<ast::Expr> {
        let suffix = Some(ast::UintTy::U32.name());
        let lit = token::Lit::new(token::Integer, sym::integer(n), suffix);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_bool(&self, span: Span, value: bool) -> Box<ast::Expr> {
        let lit = token::Lit::new(token::Bool, if value { kw::True } else { kw::False }, None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_str(&self, span: Span, s: Symbol) -> Box<ast::Expr> {
        let lit = token::Lit::new(token::Str, literal::escape_string_symbol(s), None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_byte_str(&self, span: Span, bytes: Vec<u8>) -> Box<ast::Expr> {
        let lit = token::Lit::new(token::ByteStr, literal::escape_byte_str_symbol(&bytes), None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    /// `[expr1, expr2, ...]`
    pub fn expr_array(&self, sp: Span, exprs: ThinVec<Box<ast::Expr>>) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::Array(exprs))
    }

    /// `&[expr1, expr2, ...]`
    pub fn expr_array_ref(&self, sp: Span, exprs: ThinVec<Box<ast::Expr>>) -> Box<ast::Expr> {
        self.expr_addr_of(sp, self.expr_array(sp, exprs))
    }

    pub fn expr_some(&self, sp: Span, expr: Box<ast::Expr>) -> Box<ast::Expr> {
        let some = self.std_path(sp, &[sym::option, sym::Option, sym::Some]);
        self.expr_call_global(sp, some, thin_vec![expr])
    }

    pub fn expr_none(&self, sp: Span) -> Box<ast::Expr> {
        let none = self.std_path(sp, &[sym::option, sym::Option, sym::None]);
        self.expr_path(none)
    }
    pub fn expr_tuple(&self, sp: Span, exprs: ThinVec<Box<ast::Expr>>) -> Box<ast::Expr> {
        self.expr(sp, ast::ExprKind::Tup(exprs))
    }

    pub fn expr_ok(&self, sp: Span, expr: Box<ast::Expr>) -> Box<ast::Expr> {
        let ok = self.std_path(sp, &[sym::result, sym::Result, sym::Ok]);
        self.expr_call_global(sp, ok, thin_vec![expr])
    }

    pub fn expr_call_intrinsic(
        &self,
        span: Span,
        intrinsic: Symbol,
        args: ThinVec<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        self.expr_call_global(span, self.std_path(span, &[sym::intrinsics, intrinsic]), args)
    }

    pub fn pat(&self, span: Span, kind: PatKind) -> ast::Pat {
        ast::Pat { id: ast::DUMMY_NODE_ID, kind, span }
    }
    pub fn pat_wild(&self, span: Span) -> ast::Pat {
        self.pat(span, PatKind::Wild)
    }
    pub fn pat_ident(&self, span: Span, name: Symbol) -> ast::Pat {
        self.pat_ident_binding_mode(span, name, ast::BindingMode::NONE)
    }

    pub fn pat_ident_binding_mode(
        &self,
        span: Span,
        name: Symbol,
        ann: ast::BindingMode,
    ) -> ast::Pat {
        let pat = PatKind::Ident(ann, Ident::new(name, span), None);
        self.pat(span, pat)
    }
    pub fn pat_path(&self, span: Span, path: Path) -> ast::Pat {
        self.pat(span, PatKind::Path(None, path))
    }
    pub fn pat_tuple_struct(&self, span: Span, path: Path, subpats: ThinVec<ast::Pat>) -> ast::Pat {
        self.pat(span, PatKind::TupleStruct(None, path, subpats))
    }
    pub fn pat_struct(
        &self,
        span: Span,
        path: Path,
        field_pats: ThinVec<ast::PatField>,
    ) -> ast::Pat {
        self.pat(span, PatKind::Struct(None, path, field_pats, ast::PatFieldsRest::None))
    }
    pub fn pat_tuple(&self, span: Span, pats: ThinVec<ast::Pat>) -> ast::Pat {
        self.pat(span, PatKind::Tuple(pats))
    }

    pub fn pat_some(&self, span: Span, pat: ast::Pat) -> ast::Pat {
        let some = self.std_path(span, &[sym::option, sym::Option, sym::Some]);
        self.pat_tuple_struct(span, some, thin_vec![pat])
    }

    pub fn arm(&self, span: Span, pat: ast::Pat, expr: Box<ast::Expr>) -> ast::Arm {
        ast::Arm {
            attrs: AttrVec::new(),
            pat: Box::new(pat),
            guard: None,
            body: Some(expr),
            span,
            id: ast::DUMMY_NODE_ID,
            is_placeholder: false,
        }
    }

    pub fn expr_match(
        &self,
        span: Span,
        arg: Box<ast::Expr>,
        arms: ThinVec<ast::Arm>,
    ) -> Box<Expr> {
        self.expr(span, ast::ExprKind::Match(arg, arms, MatchKind::Prefix))
    }

    pub fn expr_if(
        &self,
        span: Span,
        cond: Box<ast::Expr>,
        then: Box<ast::Expr>,
        els: Option<Box<ast::Expr>>,
    ) -> Box<ast::Expr> {
        let els = els.map(|x| self.expr_block(self.block_expr(x)));
        self.expr(span, ast::ExprKind::If(cond, self.block_expr(then), els))
    }

    pub fn closure(&self, span: Span, ids: Vec<Symbol>, body: Box<ast::Expr>) -> Box<ast::Expr> {
        let fn_decl = self.fn_decl(
            ids.iter().map(|id| self.param(span, *id, self.ty_infer(span))).collect(),
            ast::FnRetTy::Default(span),
        );

        // FIXME -- We are using `span` as the span of the `|...|`
        // part of the closure, but it probably (maybe?) corresponds to
        // the entire closure body. Probably we should extend the API
        // here, but that's not entirely clear.
        self.expr(
            span,
            ast::ExprKind::Closure(Box::new(ast::Closure {
                binder: ast::ClosureBinder::NotPresent,
                capture_clause: ast::CaptureBy::Ref,
                constness: ast::Const::No,
                coroutine_marker: None,
                movability: ast::Movability::Movable,
                fn_decl,
                body,
                fn_decl_span: span,
                // FIXME(SarthakSingh31): This points to the start of the declaration block and
                // not the span of the argument block.
                fn_arg_span: span,
            })),
        )
    }

    pub fn param(&self, span: Span, name: Symbol, ty: Box<ast::Ty>) -> ast::Param {
        let pat = Box::new(self.pat_ident(span, name));
        ast::Param {
            attrs: AttrVec::default(),
            id: ast::DUMMY_NODE_ID,
            pat,
            span,
            ty,
            is_placeholder: false,
        }
    }

    // `self` is unused but keep it as method for the convenience use.
    pub fn fn_decl(&self, inputs: ThinVec<ast::Param>, output: ast::FnRetTy) -> Box<ast::FnDecl> {
        Box::new(ast::FnDecl { inputs, output })
    }

    pub fn item(&self, span: Span, attrs: ast::AttrVec, kind: ast::ItemKind) -> Box<ast::Item> {
        Box::new(ast::Item {
            attrs,
            id: ast::DUMMY_NODE_ID,
            kind,
            vis: ast::Visibility {
                span: span.shrink_to_lo(),
                kind: ast::VisibilityKind::Inherited,
            },
            span,
            tokens: None,
        })
    }

    pub fn item_trait_impl(
        &self,
        span: Span,
        attrs: ast::AttrVec,
        generics: ast::Generics,
        safety: ast::Safety,
        is_const: bool,
        trait_ref: ast::TraitRef,
        self_ty: Box<ast::Ty>,
        items: ThinVec<Box<ast::AssocItem>>,
    ) -> Box<ast::Item> {
        self.item(
            span,
            attrs,
            ast::ItemKind::Impl(ast::Impl {
                generics,
                of_trait: Some(Box::new(ast::TraitImplHeader {
                    safety,
                    polarity: ast::ImplPolarity::Positive,
                    defaultness: ast::Defaultness::Implicit,
                    trait_ref,
                })),
                constness: if is_const { ast::Const::Yes(DUMMY_SP) } else { ast::Const::No },
                self_ty,
                items,
            }),
        )
    }

    pub fn item_static(
        &self,
        span: Span,
        attrs: ast::AttrVec,
        ident: Ident,
        ty: Box<ast::Ty>,
        mutability: ast::Mutability,
        expr: Box<ast::Expr>,
    ) -> Box<ast::Item> {
        self.item(
            span,
            attrs,
            ast::ItemKind::Static(
                ast::StaticItem {
                    ident,
                    ty,
                    safety: ast::Safety::Default,
                    mutability,
                    expr: Some(expr),
                    define_opaque: None,
                    eii_impl: None,
                }
                .into(),
            ),
        )
    }

    pub fn item_const(
        &self,
        span: Span,
        ident: Ident,
        ty: Box<ast::Ty>,
        body: Option<Box<Expr>>,
    ) -> Box<ast::Item> {
        let defaultness = ast::Defaultness::Implicit;
        self.item(
            span,
            AttrVec::new(),
            ast::ItemKind::Const(
                ast::ConstItem {
                    defaultness,
                    ident,
                    // FIXME(generic_const_items): Pass the generics as a parameter.
                    generics: ast::Generics::default(),
                    ty,
                    body,
                    define_opaque: None,
                }
                .into(),
            ),
        )
    }

    pub fn item_const_underscore(&self, span: Span, body: Box<ast::Block>) -> Box<ast::Item> {
        self.item_const(
            span,
            Ident::new(kw::Underscore, span),
            self.ty_unit(span),
            Some(self.expr_block(body)),
        )
    }

    pub fn item_fn(
        &self,
        sig: ast::FnSig,
        ident: Ident,
        generics: ast::Generics,
        body: Option<Box<ast::Block>>,
    ) -> Box<ast::Fn> {
        Box::new(ast::Fn {
            defaultness: ast::Defaultness::Implicit,
            sig,
            ident,
            generics,
            contract: None,
            body,
            define_opaque: None,
            eii_impl: None,
        })
    }

    // Builds `#[name]`.
    pub fn attr_word(&self, name: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_word(g, ast::AttrStyle::Outer, name, span)
    }

    // Builds `#[name = val]`.
    //
    // Note: `span` is used for both the identifier and the value.
    pub fn attr_name_value_str(&self, name: Symbol, val: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_name_value_str(g, ast::AttrStyle::Outer, name, val, span)
    }

    // Builds `#[outer(inner)]`.
    pub fn attr_nested_word(&self, outer: Symbol, inner: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_nested_word(g, ast::AttrStyle::Outer, outer, inner, span)
    }

    // Builds an attribute fully manually.
    pub fn attr_nested(&self, inner: AttrItem, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_from_item(g, inner, None, ast::AttrStyle::Outer, span)
    }

    pub fn empty_generics(&self, span: Span) -> ast::Generics {
        ast::Generics {
            params: ThinVec::new(),
            where_clause: ast::WhereClause {
                has_where_token: false,
                predicates: ThinVec::new(),
                span,
            },
            span,
        }
    }
}
