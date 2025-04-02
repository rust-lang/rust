use rustc_ast::ptr::P;
use rustc_ast::util::literal;
use rustc_ast::{
    self as ast, AnonConst, AttrVec, BlockCheckMode, Expr, LocalKind, MatchKind, PatKind, UnOp,
    attr, token,
};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::base::ExtCtxt;

impl<'a> ExtCtxt<'a> {
    pub fn path(&self, span: Span, strs: Vec<Ident>) -> ast::Path {
        self.path_all(span, false, strs, vec![])
    }
    pub fn path_ident(&self, span: Span, id: Ident) -> ast::Path {
        self.path(span, vec![id])
    }
    pub fn path_global(&self, span: Span, strs: Vec<Ident>) -> ast::Path {
        self.path_all(span, true, strs, vec![])
    }
    pub fn path_all(
        &self,
        span: Span,
        global: bool,
        mut idents: Vec<Ident>,
        args: Vec<ast::GenericArg>,
    ) -> ast::Path {
        assert!(!idents.is_empty());
        let add_root = global && !idents[0].is_path_segment_keyword();
        let mut segments = ThinVec::with_capacity(idents.len() + add_root as usize);
        if add_root {
            segments.push(ast::PathSegment::path_root(span));
        }
        let last_ident = idents.pop().unwrap();
        segments.extend(
            idents.into_iter().map(|ident| ast::PathSegment::from_ident(ident.with_span_pos(span))),
        );
        let args = if !args.is_empty() {
            let args = args.into_iter().map(ast::AngleBracketedArg::Arg).collect();
            Some(ast::AngleBracketedArgs { args, span }.into())
        } else {
            None
        };
        segments.push(ast::PathSegment {
            ident: last_ident.with_span_pos(span),
            id: ast::DUMMY_NODE_ID,
            args,
        });
        ast::Path { span, segments, tokens: None }
    }

    pub fn macro_call(
        &self,
        span: Span,
        path: ast::Path,
        delim: ast::token::Delimiter,
        tokens: ast::tokenstream::TokenStream,
    ) -> P<ast::MacCall> {
        P(ast::MacCall {
            path,
            args: P(ast::DelimArgs {
                dspan: ast::tokenstream::DelimSpan { open: span, close: span },
                delim,
                tokens,
            }),
        })
    }

    pub fn ty_mt(&self, ty: P<ast::Ty>, mutbl: ast::Mutability) -> ast::MutTy {
        ast::MutTy { ty, mutbl }
    }

    pub fn ty(&self, span: Span, kind: ast::TyKind) -> P<ast::Ty> {
        P(ast::Ty { id: ast::DUMMY_NODE_ID, span, kind, tokens: None })
    }

    pub fn ty_infer(&self, span: Span) -> P<ast::Ty> {
        self.ty(span, ast::TyKind::Infer)
    }

    pub fn ty_path(&self, path: ast::Path) -> P<ast::Ty> {
        self.ty(path.span, ast::TyKind::Path(None, path))
    }

    // Might need to take bounds as an argument in the future, if you ever want
    // to generate a bounded existential trait type.
    pub fn ty_ident(&self, span: Span, ident: Ident) -> P<ast::Ty> {
        self.ty_path(self.path_ident(span, ident))
    }

    pub fn anon_const(&self, span: Span, kind: ast::ExprKind) -> ast::AnonConst {
        ast::AnonConst {
            id: ast::DUMMY_NODE_ID,
            value: P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                kind,
                span,
                attrs: AttrVec::new(),
                tokens: None,
            }),
        }
    }

    pub fn const_ident(&self, span: Span, ident: Ident) -> ast::AnonConst {
        self.anon_const(span, ast::ExprKind::Path(None, self.path_ident(span, ident)))
    }

    pub fn ty_ref(
        &self,
        span: Span,
        ty: P<ast::Ty>,
        lifetime: Option<ast::Lifetime>,
        mutbl: ast::Mutability,
    ) -> P<ast::Ty> {
        self.ty(span, ast::TyKind::Ref(lifetime, self.ty_mt(ty, mutbl)))
    }

    pub fn ty_ptr(&self, span: Span, ty: P<ast::Ty>, mutbl: ast::Mutability) -> P<ast::Ty> {
        self.ty(span, ast::TyKind::Ptr(self.ty_mt(ty, mutbl)))
    }

    pub fn typaram(
        &self,
        span: Span,
        ident: Ident,
        bounds: ast::GenericBounds,
        default: Option<P<ast::Ty>>,
    ) -> ast::GenericParam {
        ast::GenericParam {
            ident: ident.with_span_pos(span),
            id: ast::DUMMY_NODE_ID,
            attrs: AttrVec::new(),
            bounds,
            kind: ast::GenericParamKind::Type { default },
            is_placeholder: false,
            colon_span: None,
        }
    }

    pub fn lifetime_param(
        &self,
        span: Span,
        ident: Ident,
        bounds: ast::GenericBounds,
    ) -> ast::GenericParam {
        ast::GenericParam {
            id: ast::DUMMY_NODE_ID,
            ident: ident.with_span_pos(span),
            attrs: AttrVec::new(),
            bounds,
            is_placeholder: false,
            kind: ast::GenericParamKind::Lifetime,
            colon_span: None,
        }
    }

    pub fn const_param(
        &self,
        span: Span,
        ident: Ident,
        bounds: ast::GenericBounds,
        ty: P<ast::Ty>,
        default: Option<AnonConst>,
    ) -> ast::GenericParam {
        ast::GenericParam {
            id: ast::DUMMY_NODE_ID,
            ident: ident.with_span_pos(span),
            attrs: AttrVec::new(),
            bounds,
            is_placeholder: false,
            kind: ast::GenericParamKind::Const { ty, kw_span: DUMMY_SP, default },
            colon_span: None,
        }
    }

    pub fn trait_ref(&self, path: ast::Path) -> ast::TraitRef {
        ast::TraitRef { path, ref_id: ast::DUMMY_NODE_ID }
    }

    pub fn poly_trait_ref(&self, span: Span, path: ast::Path, is_const: bool) -> ast::PolyTraitRef {
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
        }
    }

    pub fn trait_bound(&self, path: ast::Path, is_const: bool) -> ast::GenericBound {
        ast::GenericBound::Trait(self.poly_trait_ref(path.span, path, is_const))
    }

    pub fn lifetime(&self, span: Span, ident: Ident) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, ident: ident.with_span_pos(span) }
    }

    pub fn lifetime_static(&self, span: Span) -> ast::Lifetime {
        self.lifetime(span, Ident::new(kw::StaticLifetime, span))
    }

    pub fn stmt_expr(&self, expr: P<ast::Expr>) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, span: expr.span, kind: ast::StmtKind::Expr(expr) }
    }

    pub fn stmt_let(&self, sp: Span, mutbl: bool, ident: Ident, ex: P<ast::Expr>) -> ast::Stmt {
        self.stmt_let_ty(sp, mutbl, ident, None, ex)
    }

    pub fn stmt_let_ty(
        &self,
        sp: Span,
        mutbl: bool,
        ident: Ident,
        ty: Option<P<ast::Ty>>,
        ex: P<ast::Expr>,
    ) -> ast::Stmt {
        let pat = if mutbl {
            self.pat_ident_binding_mode(sp, ident, ast::BindingMode::MUT)
        } else {
            self.pat_ident(sp, ident)
        };
        let local = P(ast::Local {
            pat,
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
    pub fn stmt_let_type_only(&self, span: Span, ty: P<ast::Ty>) -> ast::Stmt {
        let local = P(ast::Local {
            pat: self.pat_wild(span),
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

    pub fn stmt_semi(&self, expr: P<ast::Expr>) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, span: expr.span, kind: ast::StmtKind::Semi(expr) }
    }

    pub fn stmt_local(&self, local: P<ast::Local>, span: Span) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Let(local), span }
    }

    pub fn stmt_item(&self, sp: Span, item: P<ast::Item>) -> ast::Stmt {
        ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Item(item), span: sp }
    }

    pub fn block_expr(&self, expr: P<ast::Expr>) -> P<ast::Block> {
        self.block(
            expr.span,
            thin_vec![ast::Stmt {
                id: ast::DUMMY_NODE_ID,
                span: expr.span,
                kind: ast::StmtKind::Expr(expr),
            }],
        )
    }
    pub fn block(&self, span: Span, stmts: ThinVec<ast::Stmt>) -> P<ast::Block> {
        P(ast::Block {
            stmts,
            id: ast::DUMMY_NODE_ID,
            rules: BlockCheckMode::Default,
            span,
            tokens: None,
        })
    }

    pub fn expr(&self, span: Span, kind: ast::ExprKind) -> P<ast::Expr> {
        P(ast::Expr { id: ast::DUMMY_NODE_ID, kind, span, attrs: AttrVec::new(), tokens: None })
    }

    pub fn expr_path(&self, path: ast::Path) -> P<ast::Expr> {
        self.expr(path.span, ast::ExprKind::Path(None, path))
    }

    pub fn expr_ident(&self, span: Span, id: Ident) -> P<ast::Expr> {
        self.expr_path(self.path_ident(span, id))
    }
    pub fn expr_self(&self, span: Span) -> P<ast::Expr> {
        self.expr_ident(span, Ident::with_dummy_span(kw::SelfLower))
    }

    pub fn expr_macro_call(&self, span: Span, call: P<ast::MacCall>) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::MacCall(call))
    }

    pub fn expr_binary(
        &self,
        sp: Span,
        op: ast::BinOpKind,
        lhs: P<ast::Expr>,
        rhs: P<ast::Expr>,
    ) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Binary(Spanned { node: op, span: sp }, lhs, rhs))
    }

    pub fn expr_deref(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Unary(UnOp::Deref, e))
    }

    pub fn expr_addr_of(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::AddrOf(ast::BorrowKind::Ref, ast::Mutability::Not, e))
    }

    pub fn expr_paren(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Paren(e))
    }

    pub fn expr_method_call(
        &self,
        span: Span,
        expr: P<ast::Expr>,
        ident: Ident,
        args: ThinVec<P<ast::Expr>>,
    ) -> P<ast::Expr> {
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
        expr: P<ast::Expr>,
        args: ThinVec<P<ast::Expr>>,
    ) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(expr, args))
    }
    pub fn expr_loop(&self, sp: Span, block: P<ast::Block>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Loop(block, None, sp))
    }
    pub fn expr_asm(&self, sp: Span, expr: P<ast::InlineAsm>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::InlineAsm(expr))
    }
    pub fn expr_call_ident(
        &self,
        span: Span,
        id: Ident,
        args: ThinVec<P<ast::Expr>>,
    ) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(self.expr_ident(span, id), args))
    }
    pub fn expr_call_global(
        &self,
        sp: Span,
        fn_path: Vec<Ident>,
        args: ThinVec<P<ast::Expr>>,
    ) -> P<ast::Expr> {
        let pathexpr = self.expr_path(self.path_global(sp, fn_path));
        self.expr_call(sp, pathexpr, args)
    }
    pub fn expr_block(&self, b: P<ast::Block>) -> P<ast::Expr> {
        self.expr(b.span, ast::ExprKind::Block(b, None))
    }
    pub fn field_imm(&self, span: Span, ident: Ident, e: P<ast::Expr>) -> ast::ExprField {
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
        path: ast::Path,
        fields: ThinVec<ast::ExprField>,
    ) -> P<ast::Expr> {
        self.expr(
            span,
            ast::ExprKind::Struct(P(ast::StructExpr {
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
    ) -> P<ast::Expr> {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    pub fn expr_usize(&self, span: Span, n: usize) -> P<ast::Expr> {
        let suffix = Some(ast::UintTy::Usize.name());
        let lit = token::Lit::new(token::Integer, sym::integer(n), suffix);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_u32(&self, span: Span, n: u32) -> P<ast::Expr> {
        let suffix = Some(ast::UintTy::U32.name());
        let lit = token::Lit::new(token::Integer, sym::integer(n), suffix);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_bool(&self, span: Span, value: bool) -> P<ast::Expr> {
        let lit = token::Lit::new(token::Bool, if value { kw::True } else { kw::False }, None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_str(&self, span: Span, s: Symbol) -> P<ast::Expr> {
        let lit = token::Lit::new(token::Str, literal::escape_string_symbol(s), None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    pub fn expr_byte_str(&self, span: Span, bytes: Vec<u8>) -> P<ast::Expr> {
        let lit = token::Lit::new(token::ByteStr, literal::escape_byte_str_symbol(&bytes), None);
        self.expr(span, ast::ExprKind::Lit(lit))
    }

    /// `[expr1, expr2, ...]`
    pub fn expr_array(&self, sp: Span, exprs: ThinVec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Array(exprs))
    }

    /// `&[expr1, expr2, ...]`
    pub fn expr_array_ref(&self, sp: Span, exprs: ThinVec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr_addr_of(sp, self.expr_array(sp, exprs))
    }

    pub fn expr_some(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr> {
        let some = self.std_path(&[sym::option, sym::Option, sym::Some]);
        self.expr_call_global(sp, some, thin_vec![expr])
    }

    pub fn expr_none(&self, sp: Span) -> P<ast::Expr> {
        let none = self.std_path(&[sym::option, sym::Option, sym::None]);
        self.expr_path(self.path_global(sp, none))
    }
    pub fn expr_tuple(&self, sp: Span, exprs: ThinVec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Tup(exprs))
    }

    pub fn expr_unreachable(&self, span: Span) -> P<ast::Expr> {
        self.expr_macro_call(
            span,
            self.macro_call(
                span,
                self.path_global(
                    span,
                    [sym::std, sym::unreachable].map(|s| Ident::new(s, span)).to_vec(),
                ),
                ast::token::Delimiter::Parenthesis,
                ast::tokenstream::TokenStream::default(),
            ),
        )
    }

    pub fn expr_ok(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr> {
        let ok = self.std_path(&[sym::result, sym::Result, sym::Ok]);
        self.expr_call_global(sp, ok, thin_vec![expr])
    }

    pub fn expr_try(&self, sp: Span, head: P<ast::Expr>) -> P<ast::Expr> {
        let ok = self.std_path(&[sym::result, sym::Result, sym::Ok]);
        let ok_path = self.path_global(sp, ok);
        let err = self.std_path(&[sym::result, sym::Result, sym::Err]);
        let err_path = self.path_global(sp, err);

        let binding_variable = Ident::new(sym::__try_var, sp);
        let binding_pat = self.pat_ident(sp, binding_variable);
        let binding_expr = self.expr_ident(sp, binding_variable);

        // `Ok(__try_var)` pattern
        let ok_pat = self.pat_tuple_struct(sp, ok_path, thin_vec![binding_pat.clone()]);

        // `Err(__try_var)` (pattern and expression respectively)
        let err_pat = self.pat_tuple_struct(sp, err_path.clone(), thin_vec![binding_pat]);
        let err_inner_expr =
            self.expr_call(sp, self.expr_path(err_path), thin_vec![binding_expr.clone()]);
        // `return Err(__try_var)`
        let err_expr = self.expr(sp, ast::ExprKind::Ret(Some(err_inner_expr)));

        // `Ok(__try_var) => __try_var`
        let ok_arm = self.arm(sp, ok_pat, binding_expr);
        // `Err(__try_var) => return Err(__try_var)`
        let err_arm = self.arm(sp, err_pat, err_expr);

        // `match head { Ok() => ..., Err() => ... }`
        self.expr_match(sp, head, thin_vec![ok_arm, err_arm])
    }

    pub fn pat(&self, span: Span, kind: PatKind) -> P<ast::Pat> {
        P(ast::Pat { id: ast::DUMMY_NODE_ID, kind, span, tokens: None })
    }
    pub fn pat_wild(&self, span: Span) -> P<ast::Pat> {
        self.pat(span, PatKind::Wild)
    }
    pub fn pat_lit(&self, span: Span, expr: P<ast::Expr>) -> P<ast::Pat> {
        self.pat(span, PatKind::Expr(expr))
    }
    pub fn pat_ident(&self, span: Span, ident: Ident) -> P<ast::Pat> {
        self.pat_ident_binding_mode(span, ident, ast::BindingMode::NONE)
    }

    pub fn pat_ident_binding_mode(
        &self,
        span: Span,
        ident: Ident,
        ann: ast::BindingMode,
    ) -> P<ast::Pat> {
        let pat = PatKind::Ident(ann, ident.with_span_pos(span), None);
        self.pat(span, pat)
    }
    pub fn pat_path(&self, span: Span, path: ast::Path) -> P<ast::Pat> {
        self.pat(span, PatKind::Path(None, path))
    }
    pub fn pat_tuple_struct(
        &self,
        span: Span,
        path: ast::Path,
        subpats: ThinVec<P<ast::Pat>>,
    ) -> P<ast::Pat> {
        self.pat(span, PatKind::TupleStruct(None, path, subpats))
    }
    pub fn pat_struct(
        &self,
        span: Span,
        path: ast::Path,
        field_pats: ThinVec<ast::PatField>,
    ) -> P<ast::Pat> {
        self.pat(span, PatKind::Struct(None, path, field_pats, ast::PatFieldsRest::None))
    }
    pub fn pat_tuple(&self, span: Span, pats: ThinVec<P<ast::Pat>>) -> P<ast::Pat> {
        self.pat(span, PatKind::Tuple(pats))
    }

    pub fn pat_some(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat> {
        let some = self.std_path(&[sym::option, sym::Option, sym::Some]);
        let path = self.path_global(span, some);
        self.pat_tuple_struct(span, path, thin_vec![pat])
    }

    pub fn arm(&self, span: Span, pat: P<ast::Pat>, expr: P<ast::Expr>) -> ast::Arm {
        ast::Arm {
            attrs: AttrVec::new(),
            pat,
            guard: None,
            body: Some(expr),
            span,
            id: ast::DUMMY_NODE_ID,
            is_placeholder: false,
        }
    }

    pub fn arm_unreachable(&self, span: Span) -> ast::Arm {
        self.arm(span, self.pat_wild(span), self.expr_unreachable(span))
    }

    pub fn expr_match(&self, span: Span, arg: P<ast::Expr>, arms: ThinVec<ast::Arm>) -> P<Expr> {
        self.expr(span, ast::ExprKind::Match(arg, arms, MatchKind::Prefix))
    }

    pub fn expr_if(
        &self,
        span: Span,
        cond: P<ast::Expr>,
        then: P<ast::Expr>,
        els: Option<P<ast::Expr>>,
    ) -> P<ast::Expr> {
        let els = els.map(|x| self.expr_block(self.block_expr(x)));
        self.expr(span, ast::ExprKind::If(cond, self.block_expr(then), els))
    }

    pub fn lambda(&self, span: Span, ids: Vec<Ident>, body: P<ast::Expr>) -> P<ast::Expr> {
        let fn_decl = self.fn_decl(
            ids.iter().map(|id| self.param(span, *id, self.ty(span, ast::TyKind::Infer))).collect(),
            ast::FnRetTy::Default(span),
        );

        // FIXME -- We are using `span` as the span of the `|...|`
        // part of the lambda, but it probably (maybe?) corresponds to
        // the entire lambda body. Probably we should extend the API
        // here, but that's not entirely clear.
        self.expr(
            span,
            ast::ExprKind::Closure(Box::new(ast::Closure {
                binder: ast::ClosureBinder::NotPresent,
                capture_clause: ast::CaptureBy::Ref,
                constness: ast::Const::No,
                coroutine_kind: None,
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

    pub fn lambda0(&self, span: Span, body: P<ast::Expr>) -> P<ast::Expr> {
        self.lambda(span, Vec::new(), body)
    }

    pub fn lambda1(&self, span: Span, body: P<ast::Expr>, ident: Ident) -> P<ast::Expr> {
        self.lambda(span, vec![ident], body)
    }

    pub fn lambda_stmts_1(
        &self,
        span: Span,
        stmts: ThinVec<ast::Stmt>,
        ident: Ident,
    ) -> P<ast::Expr> {
        self.lambda1(span, self.expr_block(self.block(span, stmts)), ident)
    }

    pub fn param(&self, span: Span, ident: Ident, ty: P<ast::Ty>) -> ast::Param {
        let arg_pat = self.pat_ident(span, ident);
        ast::Param {
            attrs: AttrVec::default(),
            id: ast::DUMMY_NODE_ID,
            pat: arg_pat,
            span,
            ty,
            is_placeholder: false,
        }
    }

    // `self` is unused but keep it as method for the convenience use.
    pub fn fn_decl(&self, inputs: ThinVec<ast::Param>, output: ast::FnRetTy) -> P<ast::FnDecl> {
        P(ast::FnDecl { inputs, output })
    }

    pub fn item(&self, span: Span, attrs: ast::AttrVec, kind: ast::ItemKind) -> P<ast::Item> {
        P(ast::Item {
            attrs,
            id: ast::DUMMY_NODE_ID,
            kind,
            vis: ast::Visibility {
                span: span.shrink_to_lo(),
                kind: ast::VisibilityKind::Inherited,
                tokens: None,
            },
            span,
            tokens: None,
        })
    }

    pub fn item_static(
        &self,
        span: Span,
        ident: Ident,
        ty: P<ast::Ty>,
        mutability: ast::Mutability,
        expr: P<ast::Expr>,
    ) -> P<ast::Item> {
        self.item(
            span,
            AttrVec::new(),
            ast::ItemKind::Static(
                ast::StaticItem {
                    ident,
                    ty,
                    safety: ast::Safety::Default,
                    mutability,
                    expr: Some(expr),
                    define_opaque: None,
                }
                .into(),
            ),
        )
    }

    pub fn item_const(
        &self,
        span: Span,
        ident: Ident,
        ty: P<ast::Ty>,
        expr: P<ast::Expr>,
    ) -> P<ast::Item> {
        let defaultness = ast::Defaultness::Final;
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
                    expr: Some(expr),
                    define_opaque: None,
                }
                .into(),
            ),
        )
    }

    // Builds `#[name]`.
    pub fn attr_word(&self, name: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_word(g, ast::AttrStyle::Outer, ast::Safety::Default, name, span)
    }

    // Builds `#[name = val]`.
    //
    // Note: `span` is used for both the identifier and the value.
    pub fn attr_name_value_str(&self, name: Symbol, val: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_name_value_str(
            g,
            ast::AttrStyle::Outer,
            ast::Safety::Default,
            name,
            val,
            span,
        )
    }

    // Builds `#[outer(inner)]`.
    pub fn attr_nested_word(&self, outer: Symbol, inner: Symbol, span: Span) -> ast::Attribute {
        let g = &self.sess.psess.attr_id_generator;
        attr::mk_attr_nested_word(
            g,
            ast::AttrStyle::Outer,
            ast::Safety::Default,
            outer,
            inner,
            span,
        )
    }
}
