use rustc_ast::ast::{LitFloatType, LitIntType, LitKind};
use rustc_hir::{
    intravisit::{walk_expr, walk_stmt, NestedVisitorMap, Visitor},
    Body, Expr, ExprKind, Lit, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::{
    hir::map::Map,
    ty::{self, FloatTy, IntTy, Ty},
};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use if_chain::if_chain;

use crate::utils::span_lint_and_help;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of unconstrained numeric literals which may cause default numeric fallback in type
    /// inference.
    ///
    /// Default numeric fallback means that if numeric types have not yet been bound to concrete
    /// types at the end of type inference, then integer type is bound to `i32`, and similarly
    /// floating type is bound to `f64`.
    ///
    /// See [RFC0212](https://github.com/rust-lang/rfcs/blob/master/text/0212-restore-int-fallback.md) for more information about the fallback.
    ///
    /// **Why is this bad?** For those who are very careful about types, default numeric fallback
    /// can be a pitfall that cause unexpected runtime behavior.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let i = 10;
    /// let f = 1.23;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let i = 10i32;
    /// let f = 1.23f64;
    /// ```
    pub DEFAULT_NUMERIC_FALLBACK,
    restriction,
    "usage of unconstrained numeric literals which may cause default numeric fallback."
}

declare_lint_pass!(DefaultNumericFallback => [DEFAULT_NUMERIC_FALLBACK]);

impl LateLintPass<'_> for DefaultNumericFallback {
    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &'tcx Body<'_>) {
        let mut visitor = NumericFallbackVisitor::new(cx);
        visitor.visit_body(body);
    }
}

struct NumericFallbackVisitor<'a, 'tcx> {
    /// Stack manages type bound of exprs. The top element holds current expr type.
    ty_bounds: Vec<TyBound<'tcx>>,

    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> NumericFallbackVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            ty_bounds: vec![TyBound::Nothing],
            cx,
        }
    }

    /// Check whether a passed literal has potential to cause fallback or not.
    fn check_lit(&self, lit: &Lit, lit_ty: Ty<'tcx>) {
        let ty_bound = self.ty_bounds.last().unwrap();
        if_chain! {
                if matches!(lit.node,
                            LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed));
                if matches!(lit_ty.kind(), ty::Int(IntTy::I32) | ty::Float(FloatTy::F64));
                if !ty_bound.is_integral();
                then {
                    span_lint_and_help(
                        self.cx,
                        DEFAULT_NUMERIC_FALLBACK,
                        lit.span,
                        "default numeric fallback might occur",
                        None,
                        "consider adding suffix to avoid default numeric fallback",
                    );
                }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NumericFallbackVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    #[allow(clippy::too_many_lines)]
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match &expr.kind {
            ExprKind::Call(func, args) => {
                if_chain! {
                    if let ExprKind::Path(ref func_path) = func.kind;
                    if let Some(def_id) = self.cx.qpath_res(func_path, func.hir_id).opt_def_id();
                    then {
                        let fn_sig = self.cx.tcx.fn_sig(def_id).skip_binder();
                        for (expr, bound) in args.iter().zip(fn_sig.inputs().iter()) {
                            // Push found arg type, then visit arg.
                            self.ty_bounds.push(TyBound::Ty(bound));
                            self.visit_expr(expr);
                            self.ty_bounds.pop();
                        }
                        return;
                    }
                }
            },

            ExprKind::MethodCall(_, _, args, _) => {
                if let Some(def_id) = self.cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    let fn_sig = self.cx.tcx.fn_sig(def_id).skip_binder();
                    for (expr, bound) in args.iter().zip(fn_sig.inputs().iter()) {
                        self.ty_bounds.push(TyBound::Ty(bound));
                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }
            },

            ExprKind::Lit(lit) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                self.check_lit(lit, ty);
                return;
            },

            _ => {},
        }

        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Local(local) => {
                if local.ty.is_some() {
                    self.ty_bounds.push(TyBound::Any)
                } else {
                    self.ty_bounds.push(TyBound::Nothing)
                }
            },

            _ => self.ty_bounds.push(TyBound::Nothing),
        }

        walk_stmt(self, stmt);
        self.ty_bounds.pop();
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

#[derive(Debug, Clone, Copy)]
enum TyBound<'ctx> {
    Any,
    Ty(Ty<'ctx>),
    Nothing,
}

impl<'ctx> TyBound<'ctx> {
    fn is_integral(self) -> bool {
        match self {
            TyBound::Any => true,
            TyBound::Ty(t) => t.is_integral(),
            TyBound::Nothing => false,
        }
    }
}
