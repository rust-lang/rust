use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::numeric_literal;
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{Visitor, walk_expr, walk_pat, walk_stmt};
use rustc_hir::{
    Block, Body, ConstContext, Expr, ExprKind, FnRetTy, HirId, Lit, Pat, PatExpr, PatExprKind, PatKind, Stmt, StmtKind,
    StructTailExpr,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, FloatTy, IntTy, PolyFnSig, Ty};
use rustc_session::declare_lint_pass;
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of unconstrained numeric literals which may cause default numeric fallback in type
    /// inference.
    ///
    /// Default numeric fallback means that if numeric types have not yet been bound to concrete
    /// types at the end of type inference, then integer type is bound to `i32`, and similarly
    /// floating type is bound to `f64`.
    ///
    /// See [RFC0212](https://github.com/rust-lang/rfcs/blob/master/text/0212-restore-int-fallback.md) for more information about the fallback.
    ///
    /// ### Why restrict this?
    /// To ensure that every numeric type is chosen explicitly rather than implicitly.
    ///
    /// ### Known problems
    /// This lint is implemented using a custom algorithm independent of rustc's inference,
    /// which results in many false positives and false negatives.
    ///
    /// ### Example
    /// ```no_run
    /// let i = 10;
    /// let f = 1.23;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let i = 10_i32;
    /// let f = 1.23_f64;
    /// ```
    #[clippy::version = "1.52.0"]
    pub DEFAULT_NUMERIC_FALLBACK,
    restriction,
    "usage of unconstrained numeric literals which may cause default numeric fallback."
}

declare_lint_pass!(DefaultNumericFallback => [DEFAULT_NUMERIC_FALLBACK]);

impl<'tcx> LateLintPass<'tcx> for DefaultNumericFallback {
    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        // NOTE: this is different from `clippy_utils::is_inside_always_const_context`.
        // Inline const supports type inference.
        let is_parent_const = matches!(
            cx.tcx.hir_body_const_context(cx.tcx.hir_body_owner_def_id(body.id())),
            Some(ConstContext::Const { inline: false } | ConstContext::Static(_))
        );
        let mut visitor = NumericFallbackVisitor::new(cx, is_parent_const);
        visitor.visit_body(body);
    }
}

struct NumericFallbackVisitor<'a, 'tcx> {
    /// Stack manages type bound of exprs. The top element holds current expr type.
    ty_bounds: Vec<ExplicitTyBound>,

    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> NumericFallbackVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, is_parent_const: bool) -> Self {
        Self {
            ty_bounds: vec![if is_parent_const {
                ExplicitTyBound(true)
            } else {
                ExplicitTyBound(false)
            }],
            cx,
        }
    }

    /// Check whether a passed literal has potential to cause fallback or not.
    fn check_lit(&self, lit: Lit, lit_ty: Ty<'tcx>, emit_hir_id: HirId) {
        if !lit.span.in_external_macro(self.cx.sess().source_map())
            && matches!(self.ty_bounds.last(), Some(ExplicitTyBound(false)))
            && matches!(
                lit.node,
                LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed)
            )
        {
            let (suffix, is_float) = match lit_ty.kind() {
                ty::Int(IntTy::I32) => ("i32", false),
                ty::Float(FloatTy::F64) => ("f64", true),
                _ => return,
            };
            span_lint_hir_and_then(
                self.cx,
                DEFAULT_NUMERIC_FALLBACK,
                emit_hir_id,
                lit.span,
                "default numeric fallback might occur",
                |diag| {
                    let src = if let Some(src) = snippet_opt(self.cx, lit.span) {
                        src
                    } else {
                        match lit.node {
                            LitKind::Int(src, _) => format!("{src}"),
                            LitKind::Float(src, _) => format!("{src}"),
                            _ => unreachable!("Default numeric fallback never results in other types"),
                        }
                    };

                    let sugg = numeric_literal::format(&src, Some(suffix), is_float);
                    diag.span_suggestion(lit.span, "consider adding suffix", sugg, Applicability::MaybeIncorrect);
                },
            );
        }
    }
}

impl<'tcx> Visitor<'tcx> for NumericFallbackVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match &expr.kind {
            ExprKind::Block(
                Block {
                    stmts, expr: Some(_), ..
                },
                _,
            ) => {
                if let Some(fn_sig) = self.cx.tcx.parent_hir_node(expr.hir_id).fn_sig()
                    && let FnRetTy::Return(_ty) = fn_sig.decl.output
                {
                    // We cannot check the exact type since it's a `hir::Ty`` which does not implement `is_numeric`
                    self.ty_bounds.push(ExplicitTyBound(true));
                    for stmt in *stmts {
                        self.visit_stmt(stmt);
                    }
                    self.ty_bounds.pop();
                    // Ignore return expr since we know its type was inferred from return ty
                    return;
                }
            },

            // Ignore return expr since we know its type was inferred from return ty
            ExprKind::Ret(_) => return,

            ExprKind::Call(func, args) => {
                if let Some(fn_sig) = fn_sig_opt(self.cx, func.hir_id) {
                    for (expr, bound) in iter::zip(*args, fn_sig.skip_binder().inputs()) {
                        // If is from macro, try to use last bound type (typically pushed when visiting stmt),
                        // otherwise push found arg type, then visit arg,
                        if expr.span.from_expansion() {
                            self.visit_expr(expr);
                        } else {
                            self.ty_bounds.push((*bound).into());
                            self.visit_expr(expr);
                            self.ty_bounds.pop();
                        }
                    }
                    return;
                }
            },

            ExprKind::MethodCall(_, receiver, args, _) => {
                if let Some(def_id) = self.cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    let fn_sig = self.cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
                    for (expr, bound) in iter::zip(iter::once(*receiver).chain(args.iter()), fn_sig.inputs()) {
                        self.ty_bounds.push((*bound).into());
                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }
            },

            ExprKind::Struct(_, fields, base) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                if let Some(adt_def) = ty.ty_adt_def()
                    && adt_def.is_struct()
                    && let Some(variant) = adt_def.variants().iter().next()
                {
                    let fields_def = &variant.fields;

                    // Push field type then visit each field expr.
                    for field in *fields {
                        let bound = fields_def.iter().find_map(|f_def| {
                            if f_def.ident(self.cx.tcx) == field.ident {
                                Some(self.cx.tcx.type_of(f_def.did).instantiate_identity())
                            } else {
                                None
                            }
                        });
                        self.ty_bounds.push(bound.into());
                        self.visit_expr(field.expr);
                        self.ty_bounds.pop();
                    }

                    // Visit base with no bound.
                    if let StructTailExpr::Base(base) = base {
                        self.ty_bounds.push(ExplicitTyBound(false));
                        self.visit_expr(base);
                        self.ty_bounds.pop();
                    }
                    return;
                }
            },

            ExprKind::Lit(lit) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                self.check_lit(*lit, ty, expr.hir_id);
                return;
            },

            _ => {},
        }

        walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'_>) {
        if let PatKind::Expr(&PatExpr {
            hir_id,
            kind: PatExprKind::Lit { lit, .. },
            ..
        }) = pat.kind
        {
            let ty = self.cx.typeck_results().node_type(hir_id);
            self.check_lit(lit, ty, hir_id);
            return;
        }
        walk_pat(self, pat);
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            // we cannot check the exact type since it's a hir::Ty which does not implement `is_numeric`
            StmtKind::Let(local) => self.ty_bounds.push(ExplicitTyBound(local.ty.is_some())),

            _ => self.ty_bounds.push(ExplicitTyBound(false)),
        }

        walk_stmt(self, stmt);
        self.ty_bounds.pop();
    }
}

fn fn_sig_opt<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<PolyFnSig<'tcx>> {
    let node_ty = cx.typeck_results().node_type_opt(hir_id)?;
    // We can't use `Ty::fn_sig` because it automatically performs args, this may result in FNs.
    match node_ty.kind() {
        ty::FnDef(def_id, _) => Some(cx.tcx.fn_sig(*def_id).instantiate_identity()),
        ty::FnPtr(sig_tys, hdr) => Some(sig_tys.with(*hdr)),
        _ => None,
    }
}

/// Wrapper around a `bool` to make the meaning of the value clearer
#[derive(Debug, Clone, Copy)]
struct ExplicitTyBound(pub bool);

impl<'tcx> From<Ty<'tcx>> for ExplicitTyBound {
    fn from(v: Ty<'tcx>) -> Self {
        Self(v.is_numeric())
    }
}

impl<'tcx> From<Option<Ty<'tcx>>> for ExplicitTyBound {
    fn from(v: Option<Ty<'tcx>>) -> Self {
        Self(v.is_some_and(Ty::is_numeric))
    }
}
