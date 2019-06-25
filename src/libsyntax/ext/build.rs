use crate::ast::{self, Ident, Generics, Expr, BlockCheckMode, UnOp, PatKind};
use crate::attr;
use crate::source_map::{dummy_spanned, respan, Spanned};
use crate::ext::base::ExtCtxt;
use crate::ptr::P;
use crate::symbol::{kw, sym, Symbol};
use crate::ThinVec;

use rustc_target::spec::abi::Abi;
use syntax_pos::{Pos, Span};

pub trait AstBuilder {
    // Paths
    fn path(&self, span: Span, strs: Vec<ast::Ident> ) -> ast::Path;
    fn path_ident(&self, span: Span, id: ast::Ident) -> ast::Path;
    fn path_global(&self, span: Span, strs: Vec<ast::Ident> ) -> ast::Path;
    fn path_all(&self, sp: Span,
                global: bool,
                idents: Vec<ast::Ident>,
                args: Vec<ast::GenericArg>,
                constraints: Vec<ast::AssocTyConstraint>)
        -> ast::Path;

    fn qpath(&self, self_type: P<ast::Ty>,
             trait_path: ast::Path,
             ident: ast::Ident)
             -> (ast::QSelf, ast::Path);
    fn qpath_all(&self, self_type: P<ast::Ty>,
                trait_path: ast::Path,
                ident: ast::Ident,
                args: Vec<ast::GenericArg>,
                constraints: Vec<ast::AssocTyConstraint>)
                -> (ast::QSelf, ast::Path);

    // types and consts
    fn ty_mt(&self, ty: P<ast::Ty>, mutbl: ast::Mutability) -> ast::MutTy;

    fn ty(&self, span: Span, ty: ast::TyKind) -> P<ast::Ty>;
    fn ty_path(&self, path: ast::Path) -> P<ast::Ty>;
    fn ty_ident(&self, span: Span, idents: ast::Ident) -> P<ast::Ty>;
    fn anon_const(&self, span: Span, expr: ast::ExprKind) -> ast::AnonConst;
    fn const_ident(&self, span: Span, idents: ast::Ident) -> ast::AnonConst;

    fn ty_rptr(&self, span: Span,
               ty: P<ast::Ty>,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability) -> P<ast::Ty>;
    fn ty_ptr(&self, span: Span,
              ty: P<ast::Ty>,
              mutbl: ast::Mutability) -> P<ast::Ty>;

    fn ty_infer(&self, sp: Span) -> P<ast::Ty>;

    fn typaram(&self,
               span: Span,
               id: ast::Ident,
               attrs: Vec<ast::Attribute>,
               bounds: ast::GenericBounds,
               default: Option<P<ast::Ty>>) -> ast::GenericParam;

    fn trait_ref(&self, path: ast::Path) -> ast::TraitRef;
    fn poly_trait_ref(&self, span: Span, path: ast::Path) -> ast::PolyTraitRef;
    fn trait_bound(&self, path: ast::Path) -> ast::GenericBound;
    fn lifetime(&self, span: Span, ident: ast::Ident) -> ast::Lifetime;
    fn lifetime_def(&self,
                    span: Span,
                    ident: ast::Ident,
                    attrs: Vec<ast::Attribute>,
                    bounds: ast::GenericBounds)
                    -> ast::GenericParam;

    // Statements
    fn stmt_expr(&self, expr: P<ast::Expr>) -> ast::Stmt;
    fn stmt_semi(&self, expr: P<ast::Expr>) -> ast::Stmt;
    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident, ex: P<ast::Expr>) -> ast::Stmt;
    fn stmt_let_typed(&self,
                      sp: Span,
                      mutbl: bool,
                      ident: ast::Ident,
                      typ: P<ast::Ty>,
                      ex: P<ast::Expr>)
                      -> ast::Stmt;
    fn stmt_let_type_only(&self, span: Span, ty: P<ast::Ty>) -> ast::Stmt;
    fn stmt_item(&self, sp: Span, item: P<ast::Item>) -> ast::Stmt;

    // Blocks
    fn block(&self, span: Span, stmts: Vec<ast::Stmt>) -> P<ast::Block>;
    fn block_expr(&self, expr: P<ast::Expr>) -> P<ast::Block>;

    // Expressions
    fn expr(&self, span: Span, node: ast::ExprKind) -> P<ast::Expr>;
    fn expr_path(&self, path: ast::Path) -> P<ast::Expr>;
    fn expr_qpath(&self, span: Span, qself: ast::QSelf, path: ast::Path) -> P<ast::Expr>;
    fn expr_ident(&self, span: Span, id: ast::Ident) -> P<ast::Expr>;

    fn expr_self(&self, span: Span) -> P<ast::Expr>;
    fn expr_binary(&self, sp: Span, op: ast::BinOpKind,
                   lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_deref(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_unary(&self, sp: Span, op: ast::UnOp, e: P<ast::Expr>) -> P<ast::Expr>;

    fn expr_addr_of(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_mut_addr_of(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_field_access(&self, span: Span, expr: P<ast::Expr>, ident: ast::Ident) -> P<ast::Expr>;
    fn expr_tup_field_access(&self, sp: Span, expr: P<ast::Expr>,
                             idx: usize) -> P<ast::Expr>;
    fn expr_call(&self, span: Span, expr: P<ast::Expr>, args: Vec<P<ast::Expr>>) -> P<ast::Expr>;
    fn expr_call_ident(&self, span: Span, id: ast::Ident, args: Vec<P<ast::Expr>>) -> P<ast::Expr>;
    fn expr_call_global(&self, sp: Span, fn_path: Vec<ast::Ident>,
                        args: Vec<P<ast::Expr>> ) -> P<ast::Expr>;
    fn expr_method_call(&self, span: Span,
                        expr: P<ast::Expr>, ident: ast::Ident,
                        args: Vec<P<ast::Expr>> ) -> P<ast::Expr>;
    fn expr_block(&self, b: P<ast::Block>) -> P<ast::Expr>;
    fn expr_cast(&self, sp: Span, expr: P<ast::Expr>, ty: P<ast::Ty>) -> P<ast::Expr>;

    fn field_imm(&self, span: Span, name: Ident, e: P<ast::Expr>) -> ast::Field;
    fn expr_struct(&self, span: Span, path: ast::Path, fields: Vec<ast::Field>) -> P<ast::Expr>;
    fn expr_struct_ident(&self, span: Span, id: ast::Ident,
                         fields: Vec<ast::Field>) -> P<ast::Expr>;

    fn expr_lit(&self, sp: Span, lit: ast::LitKind) -> P<ast::Expr>;

    fn expr_usize(&self, span: Span, i: usize) -> P<ast::Expr>;
    fn expr_isize(&self, sp: Span, i: isize) -> P<ast::Expr>;
    fn expr_u8(&self, sp: Span, u: u8) -> P<ast::Expr>;
    fn expr_u16(&self, sp: Span, u: u16) -> P<ast::Expr>;
    fn expr_u32(&self, sp: Span, u: u32) -> P<ast::Expr>;
    fn expr_bool(&self, sp: Span, value: bool) -> P<ast::Expr>;

    fn expr_vec(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr>;
    fn expr_vec_ng(&self, sp: Span) -> P<ast::Expr>;
    fn expr_vec_slice(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr>;
    fn expr_str(&self, sp: Span, s: Symbol) -> P<ast::Expr>;

    fn expr_some(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_none(&self, sp: Span) -> P<ast::Expr>;

    fn expr_break(&self, sp: Span) -> P<ast::Expr>;

    fn expr_tuple(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr>;

    fn expr_fail(&self, span: Span, msg: Symbol) -> P<ast::Expr>;
    fn expr_unreachable(&self, span: Span) -> P<ast::Expr>;

    fn expr_ok(&self, span: Span, expr: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_err(&self, span: Span, expr: P<ast::Expr>) -> P<ast::Expr>;
    fn expr_try(&self, span: Span, head: P<ast::Expr>) -> P<ast::Expr>;

    fn pat(&self, span: Span, pat: PatKind) -> P<ast::Pat>;
    fn pat_wild(&self, span: Span) -> P<ast::Pat>;
    fn pat_lit(&self, span: Span, expr: P<ast::Expr>) -> P<ast::Pat>;
    fn pat_ident(&self, span: Span, ident: ast::Ident) -> P<ast::Pat>;

    fn pat_ident_binding_mode(&self,
                              span: Span,
                              ident: ast::Ident,
                              bm: ast::BindingMode) -> P<ast::Pat>;
    fn pat_path(&self, span: Span, path: ast::Path) -> P<ast::Pat>;
    fn pat_tuple_struct(&self, span: Span, path: ast::Path,
                        subpats: Vec<P<ast::Pat>>) -> P<ast::Pat>;
    fn pat_struct(&self, span: Span, path: ast::Path,
                  field_pats: Vec<Spanned<ast::FieldPat>>) -> P<ast::Pat>;
    fn pat_tuple(&self, span: Span, pats: Vec<P<ast::Pat>>) -> P<ast::Pat>;

    fn pat_some(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat>;
    fn pat_none(&self, span: Span) -> P<ast::Pat>;

    fn pat_ok(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat>;
    fn pat_err(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat>;

    fn arm(&self, span: Span, pats: Vec<P<ast::Pat>>, expr: P<ast::Expr>) -> ast::Arm;
    fn arm_unreachable(&self, span: Span) -> ast::Arm;

    fn expr_match(&self, span: Span, arg: P<ast::Expr>, arms: Vec<ast::Arm> ) -> P<ast::Expr>;
    fn expr_if(&self, span: Span,
               cond: P<ast::Expr>, then: P<ast::Expr>, els: Option<P<ast::Expr>>) -> P<ast::Expr>;
    fn expr_loop(&self, span: Span, block: P<ast::Block>) -> P<ast::Expr>;

    fn lambda_fn_decl(&self,
                      span: Span,
                      fn_decl: P<ast::FnDecl>,
                      body: P<ast::Expr>,
                      fn_decl_span: Span)
                      -> P<ast::Expr>;

    fn lambda(&self, span: Span, ids: Vec<ast::Ident>, body: P<ast::Expr>) -> P<ast::Expr>;
    fn lambda0(&self, span: Span, body: P<ast::Expr>) -> P<ast::Expr>;
    fn lambda1(&self, span: Span, body: P<ast::Expr>, ident: ast::Ident) -> P<ast::Expr>;

    fn lambda_stmts(&self, span: Span, ids: Vec<ast::Ident>,
                    blk: Vec<ast::Stmt>) -> P<ast::Expr>;
    fn lambda_stmts_0(&self, span: Span, stmts: Vec<ast::Stmt>) -> P<ast::Expr>;
    fn lambda_stmts_1(&self, span: Span, stmts: Vec<ast::Stmt>,
                      ident: ast::Ident) -> P<ast::Expr>;

    // Items
    fn item(&self, span: Span,
            name: Ident, attrs: Vec<ast::Attribute> , node: ast::ItemKind) -> P<ast::Item>;

    fn arg(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> ast::Arg;
    // FIXME: unused `self`
    fn fn_decl(&self, inputs: Vec<ast::Arg> , output: ast::FunctionRetTy) -> P<ast::FnDecl>;

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: Vec<ast::Arg> ,
                    output: P<ast::Ty>,
                    generics: Generics,
                    body: P<ast::Block>) -> P<ast::Item>;
    fn item_fn(&self,
               span: Span,
               name: Ident,
               inputs: Vec<ast::Arg> ,
               output: P<ast::Ty>,
               body: P<ast::Block>) -> P<ast::Item>;

    fn variant(&self, span: Span, name: Ident, tys: Vec<P<ast::Ty>> ) -> ast::Variant;
    fn item_enum_poly(&self,
                      span: Span,
                      name: Ident,
                      enum_definition: ast::EnumDef,
                      generics: Generics) -> P<ast::Item>;
    fn item_enum(&self, span: Span, name: Ident, enum_def: ast::EnumDef) -> P<ast::Item>;

    fn item_struct_poly(&self,
                        span: Span,
                        name: Ident,
                        struct_def: ast::VariantData,
                        generics: Generics) -> P<ast::Item>;
    fn item_struct(&self, span: Span, name: Ident, struct_def: ast::VariantData) -> P<ast::Item>;

    fn item_mod(&self, span: Span, inner_span: Span,
                name: Ident, attrs: Vec<ast::Attribute>,
                items: Vec<P<ast::Item>>) -> P<ast::Item>;

    fn item_extern_crate(&self, span: Span, name: Ident) -> P<ast::Item>;

    fn item_static(&self,
                   span: Span,
                   name: Ident,
                   ty: P<ast::Ty>,
                   mutbl: ast::Mutability,
                   expr: P<ast::Expr>)
                   -> P<ast::Item>;

    fn item_const(&self,
                   span: Span,
                   name: Ident,
                   ty: P<ast::Ty>,
                   expr: P<ast::Expr>)
                   -> P<ast::Item>;

    fn item_ty_poly(&self,
                    span: Span,
                    name: Ident,
                    ty: P<ast::Ty>,
                    generics: Generics) -> P<ast::Item>;
    fn item_ty(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> P<ast::Item>;

    fn attribute(&self, sp: Span, mi: ast::MetaItem) -> ast::Attribute;

    fn meta_word(&self, sp: Span, w: ast::Name) -> ast::MetaItem;

    fn meta_list_item_word(&self, sp: Span, w: ast::Name) -> ast::NestedMetaItem;

    fn meta_list(&self,
                 sp: Span,
                 name: ast::Name,
                 mis: Vec<ast::NestedMetaItem> )
                 -> ast::MetaItem;
    fn meta_name_value(&self,
                       sp: Span,
                       name: ast::Name,
                       value: ast::LitKind)
                       -> ast::MetaItem;

    fn item_use(&self, sp: Span,
                vis: ast::Visibility, vp: P<ast::UseTree>) -> P<ast::Item>;
    fn item_use_simple(&self, sp: Span, vis: ast::Visibility, path: ast::Path) -> P<ast::Item>;
    fn item_use_simple_(&self, sp: Span, vis: ast::Visibility,
                        ident: Option<ast::Ident>, path: ast::Path) -> P<ast::Item>;
    fn item_use_list(&self, sp: Span, vis: ast::Visibility,
                     path: Vec<ast::Ident>, imports: &[ast::Ident]) -> P<ast::Item>;
    fn item_use_glob(&self, sp: Span,
                     vis: ast::Visibility, path: Vec<ast::Ident>) -> P<ast::Item>;
}

impl<'a> AstBuilder for ExtCtxt<'a> {
    fn path(&self, span: Span, strs: Vec<ast::Ident> ) -> ast::Path {
        self.path_all(span, false, strs, vec![], vec![])
    }
    fn path_ident(&self, span: Span, id: ast::Ident) -> ast::Path {
        self.path(span, vec![id])
    }
    fn path_global(&self, span: Span, strs: Vec<ast::Ident> ) -> ast::Path {
        self.path_all(span, true, strs, vec![], vec![])
    }
    fn path_all(&self,
                span: Span,
                global: bool,
                mut idents: Vec<ast::Ident> ,
                args: Vec<ast::GenericArg>,
                constraints: Vec<ast::AssocTyConstraint> )
                -> ast::Path {
        assert!(!idents.is_empty());
        let add_root = global && !idents[0].is_path_segment_keyword();
        let mut segments = Vec::with_capacity(idents.len() + add_root as usize);
        if add_root {
            segments.push(ast::PathSegment::path_root(span));
        }
        let last_ident = idents.pop().unwrap();
        segments.extend(idents.into_iter().map(|ident| {
            ast::PathSegment::from_ident(ident.with_span_pos(span))
        }));
        let args = if !args.is_empty() || !constraints.is_empty() {
            ast::AngleBracketedArgs { args, constraints, span }.into()
        } else {
            None
        };
        segments.push(ast::PathSegment {
            ident: last_ident.with_span_pos(span),
            id: ast::DUMMY_NODE_ID,
            args,
        });
        ast::Path { span, segments }
    }

    /// Constructs a qualified path.
    ///
    /// Constructs a path like `<self_type as trait_path>::ident`.
    fn qpath(&self,
             self_type: P<ast::Ty>,
             trait_path: ast::Path,
             ident: ast::Ident)
             -> (ast::QSelf, ast::Path) {
        self.qpath_all(self_type, trait_path, ident, vec![], vec![])
    }

    /// Constructs a qualified path.
    ///
    /// Constructs a path like `<self_type as trait_path>::ident<'a, T, A = Bar>`.
    fn qpath_all(&self,
                 self_type: P<ast::Ty>,
                 trait_path: ast::Path,
                 ident: ast::Ident,
                 args: Vec<ast::GenericArg>,
                 constraints: Vec<ast::AssocTyConstraint>)
                 -> (ast::QSelf, ast::Path) {
        let mut path = trait_path;
        let args = if !args.is_empty() || !constraints.is_empty() {
            ast::AngleBracketedArgs { args, constraints, span: ident.span }.into()
        } else {
            None
        };
        path.segments.push(ast::PathSegment { ident, id: ast::DUMMY_NODE_ID, args });

        (ast::QSelf {
            ty: self_type,
            path_span: path.span,
            position: path.segments.len() - 1
        }, path)
    }

    fn ty_mt(&self, ty: P<ast::Ty>, mutbl: ast::Mutability) -> ast::MutTy {
        ast::MutTy {
            ty,
            mutbl,
        }
    }

    fn ty(&self, span: Span, ty: ast::TyKind) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            span,
            node: ty
        })
    }

    fn ty_path(&self, path: ast::Path) -> P<ast::Ty> {
        self.ty(path.span, ast::TyKind::Path(None, path))
    }

    // Might need to take bounds as an argument in the future, if you ever want
    // to generate a bounded existential trait type.
    fn ty_ident(&self, span: Span, ident: ast::Ident)
        -> P<ast::Ty> {
        self.ty_path(self.path_ident(span, ident))
    }

    fn anon_const(&self, span: Span, expr: ast::ExprKind) -> ast::AnonConst {
        ast::AnonConst {
            id: ast::DUMMY_NODE_ID,
            value: P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: expr,
                span,
                attrs: ThinVec::new(),
            })
        }
    }

    fn const_ident(&self, span: Span, ident: ast::Ident) -> ast::AnonConst {
        self.anon_const(span, ast::ExprKind::Path(None, self.path_ident(span, ident)))
    }

    fn ty_rptr(&self,
               span: Span,
               ty: P<ast::Ty>,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability)
        -> P<ast::Ty> {
        self.ty(span,
                ast::TyKind::Rptr(lifetime, self.ty_mt(ty, mutbl)))
    }

    fn ty_ptr(&self,
              span: Span,
              ty: P<ast::Ty>,
              mutbl: ast::Mutability)
        -> P<ast::Ty> {
        self.ty(span,
                ast::TyKind::Ptr(self.ty_mt(ty, mutbl)))
    }

    fn ty_infer(&self, span: Span) -> P<ast::Ty> {
        self.ty(span, ast::TyKind::Infer)
    }

    fn typaram(&self,
               span: Span,
               ident: ast::Ident,
               attrs: Vec<ast::Attribute>,
               bounds: ast::GenericBounds,
               default: Option<P<ast::Ty>>) -> ast::GenericParam {
        ast::GenericParam {
            ident: ident.with_span_pos(span),
            id: ast::DUMMY_NODE_ID,
            attrs: attrs.into(),
            bounds,
            kind: ast::GenericParamKind::Type {
                default,
            }
        }
    }

    fn trait_ref(&self, path: ast::Path) -> ast::TraitRef {
        ast::TraitRef {
            path,
            ref_id: ast::DUMMY_NODE_ID,
        }
    }

    fn poly_trait_ref(&self, span: Span, path: ast::Path) -> ast::PolyTraitRef {
        ast::PolyTraitRef {
            bound_generic_params: Vec::new(),
            trait_ref: self.trait_ref(path),
            span,
        }
    }

    fn trait_bound(&self, path: ast::Path) -> ast::GenericBound {
        ast::GenericBound::Trait(self.poly_trait_ref(path.span, path),
                                 ast::TraitBoundModifier::None)
    }

    fn lifetime(&self, span: Span, ident: ast::Ident) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, ident: ident.with_span_pos(span) }
    }

    fn lifetime_def(&self,
                    span: Span,
                    ident: ast::Ident,
                    attrs: Vec<ast::Attribute>,
                    bounds: ast::GenericBounds)
                    -> ast::GenericParam {
        let lifetime = self.lifetime(span, ident);
        ast::GenericParam {
            ident: lifetime.ident,
            id: lifetime.id,
            attrs: attrs.into(),
            bounds,
            kind: ast::GenericParamKind::Lifetime,
        }
    }

    fn stmt_expr(&self, expr: P<ast::Expr>) -> ast::Stmt {
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            span: expr.span,
            node: ast::StmtKind::Expr(expr),
        }
    }

    fn stmt_semi(&self, expr: P<ast::Expr>) -> ast::Stmt {
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            span: expr.span,
            node: ast::StmtKind::Semi(expr),
        }
    }

    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident,
                ex: P<ast::Expr>) -> ast::Stmt {
        let pat = if mutbl {
            let binding_mode = ast::BindingMode::ByValue(ast::Mutability::Mutable);
            self.pat_ident_binding_mode(sp, ident, binding_mode)
        } else {
            self.pat_ident(sp, ident)
        };
        let local = P(ast::Local {
            pat,
            ty: None,
            init: Some(ex),
            id: ast::DUMMY_NODE_ID,
            span: sp,
            attrs: ThinVec::new(),
        });
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Local(local),
            span: sp,
        }
    }

    fn stmt_let_typed(&self,
                      sp: Span,
                      mutbl: bool,
                      ident: ast::Ident,
                      typ: P<ast::Ty>,
                      ex: P<ast::Expr>)
                      -> ast::Stmt {
        let pat = if mutbl {
            let binding_mode = ast::BindingMode::ByValue(ast::Mutability::Mutable);
            self.pat_ident_binding_mode(sp, ident, binding_mode)
        } else {
            self.pat_ident(sp, ident)
        };
        let local = P(ast::Local {
            pat,
            ty: Some(typ),
            init: Some(ex),
            id: ast::DUMMY_NODE_ID,
            span: sp,
            attrs: ThinVec::new(),
        });
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Local(local),
            span: sp,
        }
    }

    // Generates `let _: Type;`, which is usually used for type assertions.
    fn stmt_let_type_only(&self, span: Span, ty: P<ast::Ty>) -> ast::Stmt {
        let local = P(ast::Local {
            pat: self.pat_wild(span),
            ty: Some(ty),
            init: None,
            id: ast::DUMMY_NODE_ID,
            span,
            attrs: ThinVec::new(),
        });
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Local(local),
            span,
        }
    }

    fn stmt_item(&self, sp: Span, item: P<ast::Item>) -> ast::Stmt {
        ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Item(item),
            span: sp,
        }
    }

    fn block_expr(&self, expr: P<ast::Expr>) -> P<ast::Block> {
        self.block(expr.span, vec![ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            span: expr.span,
            node: ast::StmtKind::Expr(expr),
        }])
    }
    fn block(&self, span: Span, stmts: Vec<ast::Stmt>) -> P<ast::Block> {
        P(ast::Block {
           stmts,
           id: ast::DUMMY_NODE_ID,
           rules: BlockCheckMode::Default,
           span,
        })
    }

    fn expr(&self, span: Span, node: ast::ExprKind) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node,
            span,
            attrs: ThinVec::new(),
        })
    }

    fn expr_path(&self, path: ast::Path) -> P<ast::Expr> {
        self.expr(path.span, ast::ExprKind::Path(None, path))
    }

    /// Constructs a `QPath` expression.
    fn expr_qpath(&self, span: Span, qself: ast::QSelf, path: ast::Path) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Path(Some(qself), path))
    }

    fn expr_ident(&self, span: Span, id: ast::Ident) -> P<ast::Expr> {
        self.expr_path(self.path_ident(span, id))
    }
    fn expr_self(&self, span: Span) -> P<ast::Expr> {
        self.expr_ident(span, Ident::with_empty_ctxt(kw::SelfLower))
    }

    fn expr_binary(&self, sp: Span, op: ast::BinOpKind,
                   lhs: P<ast::Expr>, rhs: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Binary(Spanned { node: op, span: sp }, lhs, rhs))
    }

    fn expr_deref(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr_unary(sp, UnOp::Deref, e)
    }
    fn expr_unary(&self, sp: Span, op: ast::UnOp, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Unary(op, e))
    }

    fn expr_field_access(&self, sp: Span, expr: P<ast::Expr>, ident: ast::Ident) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Field(expr, ident.with_span_pos(sp)))
    }
    fn expr_tup_field_access(&self, sp: Span, expr: P<ast::Expr>, idx: usize) -> P<ast::Expr> {
        let ident = Ident::from_str(&idx.to_string()).with_span_pos(sp);
        self.expr(sp, ast::ExprKind::Field(expr, ident))
    }
    fn expr_addr_of(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::AddrOf(ast::Mutability::Immutable, e))
    }
    fn expr_mut_addr_of(&self, sp: Span, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::AddrOf(ast::Mutability::Mutable, e))
    }

    fn expr_call(&self, span: Span, expr: P<ast::Expr>, args: Vec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(expr, args))
    }
    fn expr_call_ident(&self, span: Span, id: ast::Ident,
                       args: Vec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Call(self.expr_ident(span, id), args))
    }
    fn expr_call_global(&self, sp: Span, fn_path: Vec<ast::Ident> ,
                      args: Vec<P<ast::Expr>> ) -> P<ast::Expr> {
        let pathexpr = self.expr_path(self.path_global(sp, fn_path));
        self.expr_call(sp, pathexpr, args)
    }
    fn expr_method_call(&self, span: Span,
                        expr: P<ast::Expr>,
                        ident: ast::Ident,
                        mut args: Vec<P<ast::Expr>> ) -> P<ast::Expr> {
        args.insert(0, expr);
        let segment = ast::PathSegment::from_ident(ident.with_span_pos(span));
        self.expr(span, ast::ExprKind::MethodCall(segment, args))
    }
    fn expr_block(&self, b: P<ast::Block>) -> P<ast::Expr> {
        self.expr(b.span, ast::ExprKind::Block(b, None))
    }
    fn field_imm(&self, span: Span, ident: Ident, e: P<ast::Expr>) -> ast::Field {
        ast::Field {
            ident: ident.with_span_pos(span),
            expr: e,
            span,
            is_shorthand: false,
            attrs: ThinVec::new(),
        }
    }
    fn expr_struct(&self, span: Span, path: ast::Path, fields: Vec<ast::Field>) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Struct(path, fields, None))
    }
    fn expr_struct_ident(&self, span: Span,
                         id: ast::Ident, fields: Vec<ast::Field>) -> P<ast::Expr> {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    fn expr_lit(&self, span: Span, lit_kind: ast::LitKind) -> P<ast::Expr> {
        let lit = ast::Lit::from_lit_kind(lit_kind, span);
        self.expr(span, ast::ExprKind::Lit(lit))
    }
    fn expr_usize(&self, span: Span, i: usize) -> P<ast::Expr> {
        self.expr_lit(span, ast::LitKind::Int(i as u128,
                                              ast::LitIntType::Unsigned(ast::UintTy::Usize)))
    }
    fn expr_isize(&self, sp: Span, i: isize) -> P<ast::Expr> {
        if i < 0 {
            let i = (-i) as u128;
            let lit_ty = ast::LitIntType::Signed(ast::IntTy::Isize);
            let lit = self.expr_lit(sp, ast::LitKind::Int(i, lit_ty));
            self.expr_unary(sp, ast::UnOp::Neg, lit)
        } else {
            self.expr_lit(sp, ast::LitKind::Int(i as u128,
                                                ast::LitIntType::Signed(ast::IntTy::Isize)))
        }
    }
    fn expr_u32(&self, sp: Span, u: u32) -> P<ast::Expr> {
        self.expr_lit(sp, ast::LitKind::Int(u as u128,
                                            ast::LitIntType::Unsigned(ast::UintTy::U32)))
    }
    fn expr_u16(&self, sp: Span, u: u16) -> P<ast::Expr> {
        self.expr_lit(sp, ast::LitKind::Int(u as u128,
                                            ast::LitIntType::Unsigned(ast::UintTy::U16)))
    }
    fn expr_u8(&self, sp: Span, u: u8) -> P<ast::Expr> {
        self.expr_lit(sp, ast::LitKind::Int(u as u128, ast::LitIntType::Unsigned(ast::UintTy::U8)))
    }
    fn expr_bool(&self, sp: Span, value: bool) -> P<ast::Expr> {
        self.expr_lit(sp, ast::LitKind::Bool(value))
    }

    fn expr_vec(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Array(exprs))
    }
    fn expr_vec_ng(&self, sp: Span) -> P<ast::Expr> {
        self.expr_call_global(sp, self.std_path(&[sym::vec, sym::Vec, sym::new]),
                              Vec::new())
    }
    fn expr_vec_slice(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr_addr_of(sp, self.expr_vec(sp, exprs))
    }
    fn expr_str(&self, sp: Span, s: Symbol) -> P<ast::Expr> {
        self.expr_lit(sp, ast::LitKind::Str(s, ast::StrStyle::Cooked))
    }

    fn expr_cast(&self, sp: Span, expr: P<ast::Expr>, ty: P<ast::Ty>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Cast(expr, ty))
    }

    fn expr_some(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr> {
        let some = self.std_path(&[sym::option, sym::Option, sym::Some]);
        self.expr_call_global(sp, some, vec![expr])
    }

    fn expr_none(&self, sp: Span) -> P<ast::Expr> {
        let none = self.std_path(&[sym::option, sym::Option, sym::None]);
        let none = self.path_global(sp, none);
        self.expr_path(none)
    }

    fn expr_break(&self, sp: Span) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Break(None, None))
    }

    fn expr_tuple(&self, sp: Span, exprs: Vec<P<ast::Expr>>) -> P<ast::Expr> {
        self.expr(sp, ast::ExprKind::Tup(exprs))
    }

    fn expr_fail(&self, span: Span, msg: Symbol) -> P<ast::Expr> {
        let loc = self.source_map().lookup_char_pos(span.lo());
        let expr_file = self.expr_str(span, Symbol::intern(&loc.file.name.to_string()));
        let expr_line = self.expr_u32(span, loc.line as u32);
        let expr_col = self.expr_u32(span, loc.col.to_usize() as u32 + 1);
        let expr_loc_tuple = self.expr_tuple(span, vec![expr_file, expr_line, expr_col]);
        let expr_loc_ptr = self.expr_addr_of(span, expr_loc_tuple);
        self.expr_call_global(
            span,
            self.std_path(&[sym::rt, sym::begin_panic]),
            vec![
                self.expr_str(span, msg),
                expr_loc_ptr])
    }

    fn expr_unreachable(&self, span: Span) -> P<ast::Expr> {
        self.expr_fail(span, Symbol::intern("internal error: entered unreachable code"))
    }

    fn expr_ok(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr> {
        let ok = self.std_path(&[sym::result, sym::Result, sym::Ok]);
        self.expr_call_global(sp, ok, vec![expr])
    }

    fn expr_err(&self, sp: Span, expr: P<ast::Expr>) -> P<ast::Expr> {
        let err = self.std_path(&[sym::result, sym::Result, sym::Err]);
        self.expr_call_global(sp, err, vec![expr])
    }

    fn expr_try(&self, sp: Span, head: P<ast::Expr>) -> P<ast::Expr> {
        let ok = self.std_path(&[sym::result, sym::Result, sym::Ok]);
        let ok_path = self.path_global(sp, ok);
        let err = self.std_path(&[sym::result, sym::Result, sym::Err]);
        let err_path = self.path_global(sp, err);

        let binding_variable = self.ident_of("__try_var");
        let binding_pat = self.pat_ident(sp, binding_variable);
        let binding_expr = self.expr_ident(sp, binding_variable);

        // `Ok(__try_var)` pattern
        let ok_pat = self.pat_tuple_struct(sp, ok_path, vec![binding_pat.clone()]);

        // `Err(__try_var)` (pattern and expression respectively)
        let err_pat = self.pat_tuple_struct(sp, err_path.clone(), vec![binding_pat]);
        let err_inner_expr = self.expr_call(sp, self.expr_path(err_path),
                                            vec![binding_expr.clone()]);
        // `return Err(__try_var)`
        let err_expr = self.expr(sp, ast::ExprKind::Ret(Some(err_inner_expr)));

        // `Ok(__try_var) => __try_var`
        let ok_arm = self.arm(sp, vec![ok_pat], binding_expr);
        // `Err(__try_var) => return Err(__try_var)`
        let err_arm = self.arm(sp, vec![err_pat], err_expr);

        // `match head { Ok() => ..., Err() => ... }`
        self.expr_match(sp, head, vec![ok_arm, err_arm])
    }


    fn pat(&self, span: Span, pat: PatKind) -> P<ast::Pat> {
        P(ast::Pat { id: ast::DUMMY_NODE_ID, node: pat, span })
    }
    fn pat_wild(&self, span: Span) -> P<ast::Pat> {
        self.pat(span, PatKind::Wild)
    }
    fn pat_lit(&self, span: Span, expr: P<ast::Expr>) -> P<ast::Pat> {
        self.pat(span, PatKind::Lit(expr))
    }
    fn pat_ident(&self, span: Span, ident: ast::Ident) -> P<ast::Pat> {
        let binding_mode = ast::BindingMode::ByValue(ast::Mutability::Immutable);
        self.pat_ident_binding_mode(span, ident, binding_mode)
    }

    fn pat_ident_binding_mode(&self,
                              span: Span,
                              ident: ast::Ident,
                              bm: ast::BindingMode) -> P<ast::Pat> {
        let pat = PatKind::Ident(bm, ident.with_span_pos(span), None);
        self.pat(span, pat)
    }
    fn pat_path(&self, span: Span, path: ast::Path) -> P<ast::Pat> {
        self.pat(span, PatKind::Path(None, path))
    }
    fn pat_tuple_struct(&self, span: Span, path: ast::Path,
                        subpats: Vec<P<ast::Pat>>) -> P<ast::Pat> {
        self.pat(span, PatKind::TupleStruct(path, subpats, None))
    }
    fn pat_struct(&self, span: Span, path: ast::Path,
                  field_pats: Vec<Spanned<ast::FieldPat>>) -> P<ast::Pat> {
        self.pat(span, PatKind::Struct(path, field_pats, false))
    }
    fn pat_tuple(&self, span: Span, pats: Vec<P<ast::Pat>>) -> P<ast::Pat> {
        self.pat(span, PatKind::Tuple(pats, None))
    }

    fn pat_some(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat> {
        let some = self.std_path(&[sym::option, sym::Option, sym::Some]);
        let path = self.path_global(span, some);
        self.pat_tuple_struct(span, path, vec![pat])
    }

    fn pat_none(&self, span: Span) -> P<ast::Pat> {
        let some = self.std_path(&[sym::option, sym::Option, sym::None]);
        let path = self.path_global(span, some);
        self.pat_path(span, path)
    }

    fn pat_ok(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat> {
        let some = self.std_path(&[sym::result, sym::Result, sym::Ok]);
        let path = self.path_global(span, some);
        self.pat_tuple_struct(span, path, vec![pat])
    }

    fn pat_err(&self, span: Span, pat: P<ast::Pat>) -> P<ast::Pat> {
        let some = self.std_path(&[sym::result, sym::Result, sym::Err]);
        let path = self.path_global(span, some);
        self.pat_tuple_struct(span, path, vec![pat])
    }

    fn arm(&self, span: Span, pats: Vec<P<ast::Pat>>, expr: P<ast::Expr>) -> ast::Arm {
        ast::Arm {
            attrs: vec![],
            pats,
            guard: None,
            body: expr,
            span,
        }
    }

    fn arm_unreachable(&self, span: Span) -> ast::Arm {
        self.arm(span, vec![self.pat_wild(span)], self.expr_unreachable(span))
    }

    fn expr_match(&self, span: Span, arg: P<ast::Expr>, arms: Vec<ast::Arm>) -> P<Expr> {
        self.expr(span, ast::ExprKind::Match(arg, arms))
    }

    fn expr_if(&self, span: Span, cond: P<ast::Expr>,
               then: P<ast::Expr>, els: Option<P<ast::Expr>>) -> P<ast::Expr> {
        let els = els.map(|x| self.expr_block(self.block_expr(x)));
        self.expr(span, ast::ExprKind::If(cond, self.block_expr(then), els))
    }

    fn expr_loop(&self, span: Span, block: P<ast::Block>) -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Loop(block, None))
    }

    fn lambda_fn_decl(&self,
                      span: Span,
                      fn_decl: P<ast::FnDecl>,
                      body: P<ast::Expr>,
                      fn_decl_span: Span) // span of the `|...|` part
                      -> P<ast::Expr> {
        self.expr(span, ast::ExprKind::Closure(ast::CaptureBy::Ref,
                                               ast::IsAsync::NotAsync,
                                               ast::Movability::Movable,
                                               fn_decl,
                                               body,
                                               fn_decl_span))
    }

    fn lambda(&self,
              span: Span,
              ids: Vec<ast::Ident>,
              body: P<ast::Expr>)
              -> P<ast::Expr> {
        let fn_decl = self.fn_decl(
            ids.iter().map(|id| self.arg(span, *id, self.ty_infer(span))).collect(),
            ast::FunctionRetTy::Default(span));

        // FIXME -- We are using `span` as the span of the `|...|`
        // part of the lambda, but it probably (maybe?) corresponds to
        // the entire lambda body. Probably we should extend the API
        // here, but that's not entirely clear.
        self.expr(span, ast::ExprKind::Closure(ast::CaptureBy::Ref,
                                               ast::IsAsync::NotAsync,
                                               ast::Movability::Movable,
                                               fn_decl,
                                               body,
                                               span))
    }

    fn lambda0(&self, span: Span, body: P<ast::Expr>) -> P<ast::Expr> {
        self.lambda(span, Vec::new(), body)
    }

    fn lambda1(&self, span: Span, body: P<ast::Expr>, ident: ast::Ident) -> P<ast::Expr> {
        self.lambda(span, vec![ident], body)
    }

    fn lambda_stmts(&self,
                    span: Span,
                    ids: Vec<ast::Ident>,
                    stmts: Vec<ast::Stmt>)
                    -> P<ast::Expr> {
        self.lambda(span, ids, self.expr_block(self.block(span, stmts)))
    }
    fn lambda_stmts_0(&self, span: Span, stmts: Vec<ast::Stmt>) -> P<ast::Expr> {
        self.lambda0(span, self.expr_block(self.block(span, stmts)))
    }
    fn lambda_stmts_1(&self, span: Span, stmts: Vec<ast::Stmt>,
                      ident: ast::Ident) -> P<ast::Expr> {
        self.lambda1(span, self.expr_block(self.block(span, stmts)), ident)
    }

    fn arg(&self, span: Span, ident: ast::Ident, ty: P<ast::Ty>) -> ast::Arg {
        let arg_pat = self.pat_ident(span, ident);
        ast::Arg {
            attrs: ThinVec::default(),
            id: ast::DUMMY_NODE_ID,
            pat: arg_pat,
            ty,
        }
    }

    // FIXME: unused `self`
    fn fn_decl(&self, inputs: Vec<ast::Arg>, output: ast::FunctionRetTy) -> P<ast::FnDecl> {
        P(ast::FnDecl {
            inputs,
            output,
            c_variadic: false
        })
    }

    fn item(&self, span: Span, name: Ident,
            attrs: Vec<ast::Attribute>, node: ast::ItemKind) -> P<ast::Item> {
        // FIXME: Would be nice if our generated code didn't violate
        // Rust coding conventions
        P(ast::Item {
            ident: name,
            attrs,
            id: ast::DUMMY_NODE_ID,
            node,
            vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
            span,
            tokens: None,
        })
    }

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: Vec<ast::Arg> ,
                    output: P<ast::Ty>,
                    generics: Generics,
                    body: P<ast::Block>) -> P<ast::Item> {
        self.item(span,
                  name,
                  Vec::new(),
                  ast::ItemKind::Fn(self.fn_decl(inputs, ast::FunctionRetTy::Ty(output)),
                              ast::FnHeader {
                                  unsafety: ast::Unsafety::Normal,
                                  asyncness: dummy_spanned(ast::IsAsync::NotAsync),
                                  constness: dummy_spanned(ast::Constness::NotConst),
                                  abi: Abi::Rust,
                              },
                              generics,
                              body))
    }

    fn item_fn(&self,
               span: Span,
               name: Ident,
               inputs: Vec<ast::Arg> ,
               output: P<ast::Ty>,
               body: P<ast::Block>
              ) -> P<ast::Item> {
        self.item_fn_poly(
            span,
            name,
            inputs,
            output,
            Generics::default(),
            body)
    }

    fn variant(&self, span: Span, ident: Ident, tys: Vec<P<ast::Ty>> ) -> ast::Variant {
        let fields: Vec<_> = tys.into_iter().map(|ty| {
            ast::StructField {
                span: ty.span,
                ty,
                ident: None,
                vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
                attrs: Vec::new(),
                id: ast::DUMMY_NODE_ID,
            }
        }).collect();

        let vdata = if fields.is_empty() {
            ast::VariantData::Unit(ast::DUMMY_NODE_ID)
        } else {
            ast::VariantData::Tuple(fields, ast::DUMMY_NODE_ID)
        };

        respan(span,
               ast::Variant_ {
                   ident,
                   id: ast::DUMMY_NODE_ID,
                   attrs: Vec::new(),
                   data: vdata,
                   disr_expr: None,
               })
    }

    fn item_enum_poly(&self, span: Span, name: Ident,
                      enum_definition: ast::EnumDef,
                      generics: Generics) -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::Enum(enum_definition, generics))
    }

    fn item_enum(&self, span: Span, name: Ident,
                 enum_definition: ast::EnumDef) -> P<ast::Item> {
        self.item_enum_poly(span, name, enum_definition,
                            Generics::default())
    }

    fn item_struct(&self, span: Span, name: Ident,
                   struct_def: ast::VariantData) -> P<ast::Item> {
        self.item_struct_poly(
            span,
            name,
            struct_def,
            Generics::default()
        )
    }

    fn item_struct_poly(&self, span: Span, name: Ident,
        struct_def: ast::VariantData, generics: Generics) -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::Struct(struct_def, generics))
    }

    fn item_mod(&self, span: Span, inner_span: Span, name: Ident,
                attrs: Vec<ast::Attribute>,
                items: Vec<P<ast::Item>>) -> P<ast::Item> {
        self.item(
            span,
            name,
            attrs,
            ast::ItemKind::Mod(ast::Mod {
                inner: inner_span,
                items,
                inline: true
            })
        )
    }

    fn item_extern_crate(&self, span: Span, name: Ident) -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::ExternCrate(None))
    }

    fn item_static(&self,
                   span: Span,
                   name: Ident,
                   ty: P<ast::Ty>,
                   mutbl: ast::Mutability,
                   expr: P<ast::Expr>)
                   -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::Static(ty, mutbl, expr))
    }

    fn item_const(&self,
                  span: Span,
                  name: Ident,
                  ty: P<ast::Ty>,
                  expr: P<ast::Expr>)
                  -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::Const(ty, expr))
    }

    fn item_ty_poly(&self, span: Span, name: Ident, ty: P<ast::Ty>,
                    generics: Generics) -> P<ast::Item> {
        self.item(span, name, Vec::new(), ast::ItemKind::Ty(ty, generics))
    }

    fn item_ty(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> P<ast::Item> {
        self.item_ty_poly(span, name, ty, Generics::default())
    }

    fn attribute(&self, sp: Span, mi: ast::MetaItem) -> ast::Attribute {
        attr::mk_spanned_attr_outer(sp, attr::mk_attr_id(), mi)
    }

    fn meta_word(&self, sp: Span, w: ast::Name) -> ast::MetaItem {
        attr::mk_word_item(Ident::with_empty_ctxt(w).with_span_pos(sp))
    }

    fn meta_list_item_word(&self, sp: Span, w: ast::Name) -> ast::NestedMetaItem {
        attr::mk_nested_word_item(Ident::with_empty_ctxt(w).with_span_pos(sp))
    }

    fn meta_list(&self, sp: Span, name: ast::Name, mis: Vec<ast::NestedMetaItem>)
                 -> ast::MetaItem {
        attr::mk_list_item(sp, Ident::with_empty_ctxt(name).with_span_pos(sp), mis)
    }

    fn meta_name_value(&self, span: Span, name: ast::Name, lit_kind: ast::LitKind)
                       -> ast::MetaItem {
        attr::mk_name_value_item(span, Ident::with_empty_ctxt(name).with_span_pos(span),
                                 lit_kind, span)
    }

    fn item_use(&self, sp: Span,
                vis: ast::Visibility, vp: P<ast::UseTree>) -> P<ast::Item> {
        P(ast::Item {
            id: ast::DUMMY_NODE_ID,
            ident: Ident::invalid(),
            attrs: vec![],
            node: ast::ItemKind::Use(vp),
            vis,
            span: sp,
            tokens: None,
        })
    }

    fn item_use_simple(&self, sp: Span, vis: ast::Visibility, path: ast::Path) -> P<ast::Item> {
        self.item_use_simple_(sp, vis, None, path)
    }

    fn item_use_simple_(&self, sp: Span, vis: ast::Visibility,
                        rename: Option<ast::Ident>, path: ast::Path) -> P<ast::Item> {
        self.item_use(sp, vis, P(ast::UseTree {
            span: sp,
            prefix: path,
            kind: ast::UseTreeKind::Simple(rename, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID),
        }))
    }

    fn item_use_list(&self, sp: Span, vis: ast::Visibility,
                     path: Vec<ast::Ident>, imports: &[ast::Ident]) -> P<ast::Item> {
        let imports = imports.iter().map(|id| {
            (ast::UseTree {
                span: sp,
                prefix: self.path(sp, vec![*id]),
                kind: ast::UseTreeKind::Simple(None, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID),
            }, ast::DUMMY_NODE_ID)
        }).collect();

        self.item_use(sp, vis, P(ast::UseTree {
            span: sp,
            prefix: self.path(sp, path),
            kind: ast::UseTreeKind::Nested(imports),
        }))
    }

    fn item_use_glob(&self, sp: Span,
                     vis: ast::Visibility, path: Vec<ast::Ident>) -> P<ast::Item> {
        self.item_use(sp, vis, P(ast::UseTree {
            span: sp,
            prefix: self.path(sp, path),
            kind: ast::UseTreeKind::Glob,
        }))
    }
}
