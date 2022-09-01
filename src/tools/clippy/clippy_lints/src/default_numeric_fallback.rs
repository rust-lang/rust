use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::numeric_literal;
use clippy_utils::source::snippet_opt;
use if_chain::if_chain;
use rustc_ast::ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{
    intravisit::{walk_expr, walk_stmt, Visitor},
    Body, Expr, ExprKind, HirId, Lit, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::{
    lint::in_external_macro,
    ty::{self, FloatTy, IntTy, PolyFnSig, Ty},
};
use rustc_session::{declare_lint_pass, declare_tool_lint};
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
    /// ### Why is this bad?
    /// For those who are very careful about types, default numeric fallback
    /// can be a pitfall that cause unexpected runtime behavior.
    ///
    /// ### Known problems
    /// This lint can only be allowed at the function level or above.
    ///
    /// ### Example
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
    #[clippy::version = "1.52.0"]
    pub DEFAULT_NUMERIC_FALLBACK,
    restriction,
    "usage of unconstrained numeric literals which may cause default numeric fallback."
}

declare_lint_pass!(DefaultNumericFallback => [DEFAULT_NUMERIC_FALLBACK]);

impl<'tcx> LateLintPass<'tcx> for DefaultNumericFallback {
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
        Self { ty_bounds: vec![TyBound::Nothing], cx }
    }

    /// Check whether a passed literal has potential to cause fallback or not.
    fn check_lit(&self, lit: &Lit, lit_ty: Ty<'tcx>, emit_hir_id: HirId) {
        if_chain! {
                if !in_external_macro(self.cx.sess(), lit.span);
                if let Some(ty_bound) = self.ty_bounds.last();
                if matches!(lit.node,
                            LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed));
                if !ty_bound.is_numeric();
                then {
                    let (suffix, is_float) = match lit_ty.kind() {
                        ty::Int(IntTy::I32) => ("i32", false),
                        ty::Float(FloatTy::F64) => ("f64", true),
                        // Default numeric fallback never results in other types.
                        _ => return,
                    };

                    let src = if let Some(src) = snippet_opt(self.cx, lit.span) {
                        src
                    } else {
                        match lit.node {
                            LitKind::Int(src, _) => format!("{}", src),
                            LitKind::Float(src, _) => format!("{}", src),
                            _ => return,
                        }
                    };
                    let sugg = numeric_literal::format(&src, Some(suffix), is_float);
                    span_lint_hir_and_then(
                        self.cx,
                        DEFAULT_NUMERIC_FALLBACK,
                        emit_hir_id,
                        lit.span,
                        "default numeric fallback might occur",
                        |diag| {
                            diag.span_suggestion(lit.span, "consider adding suffix", sugg, Applicability::MaybeIncorrect);
                        }
                    );
                }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NumericFallbackVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match &expr.kind {
            ExprKind::Call(func, args) => {
                if let Some(fn_sig) = fn_sig_opt(self.cx, func.hir_id) {
                    for (expr, bound) in iter::zip(*args, fn_sig.skip_binder().inputs()) {
                        // Push found arg type, then visit arg.
                        self.ty_bounds.push(TyBound::Ty(*bound));
                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }
            }

            ExprKind::MethodCall(_, receiver, args, _) => {
                if let Some(def_id) = self.cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    let fn_sig = self.cx.tcx.fn_sig(def_id).skip_binder();
                    for (expr, bound) in
                        iter::zip(std::iter::once(*receiver).chain(args.iter()), fn_sig.inputs())
                    {
                        self.ty_bounds.push(TyBound::Ty(*bound));
                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }
            }

            ExprKind::Struct(_, fields, base) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                if_chain! {
                    if let Some(adt_def) = ty.ty_adt_def();
                    if adt_def.is_struct();
                    if let Some(variant) = adt_def.variants().iter().next();
                    then {
                        let fields_def = &variant.fields;

                        // Push field type then visit each field expr.
                        for field in fields.iter() {
                            let bound =
                                fields_def
                                    .iter()
                                    .find_map(|f_def| {
                                        if f_def.ident(self.cx.tcx) == field.ident
                                            { Some(self.cx.tcx.type_of(f_def.did)) }
                                        else { None }
                                    });
                            self.ty_bounds.push(bound.into());
                            self.visit_expr(field.expr);
                            self.ty_bounds.pop();
                        }

                        // Visit base with no bound.
                        if let Some(base) = base {
                            self.ty_bounds.push(TyBound::Nothing);
                            self.visit_expr(base);
                            self.ty_bounds.pop();
                        }
                        return;
                    }
                }
            }

            ExprKind::Lit(lit) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                self.check_lit(lit, ty, expr.hir_id);
                return;
            }

            _ => {}
        }

        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Local(local) => {
                if local.ty.is_some() {
                    self.ty_bounds.push(TyBound::Any);
                } else {
                    self.ty_bounds.push(TyBound::Nothing);
                }
            }

            _ => self.ty_bounds.push(TyBound::Nothing),
        }

        walk_stmt(self, stmt);
        self.ty_bounds.pop();
    }
}

fn fn_sig_opt<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<PolyFnSig<'tcx>> {
    let node_ty = cx.typeck_results().node_type_opt(hir_id)?;
    // We can't use `Ty::fn_sig` because it automatically performs substs, this may result in FNs.
    match node_ty.kind() {
        ty::FnDef(def_id, _) => Some(cx.tcx.fn_sig(*def_id)),
        ty::FnPtr(fn_sig) => Some(*fn_sig),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum TyBound<'tcx> {
    Any,
    Ty(Ty<'tcx>),
    Nothing,
}

impl<'tcx> TyBound<'tcx> {
    fn is_numeric(self) -> bool {
        match self {
            TyBound::Any => true,
            TyBound::Ty(t) => t.is_numeric(),
            TyBound::Nothing => false,
        }
    }
}

impl<'tcx> From<Option<Ty<'tcx>>> for TyBound<'tcx> {
    fn from(v: Option<Ty<'tcx>>) -> Self {
        match v {
            Some(t) => TyBound::Ty(t),
            None => TyBound::Nothing,
        }
    }
}
