use rustc_ast::ast::{Label, LitFloatType, LitIntType, LitKind};
use rustc_hir::{
    self as hir,
    intravisit::{walk_expr, walk_stmt, walk_ty, FnKind, NestedVisitorMap, Visitor},
    Body, Expr, ExprKind, FnDecl, FnRetTy, Guard, HirId, Lit, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::{
    hir::map::Map,
    ty::{self, subst::GenericArgKind, FloatTy, IntTy, Ty, TyCtxt},
};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;
use rustc_typeck::hir_ty_to_ty;

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

fn enclosing_body_owner_opt(tcx: TyCtxt<'_>, hir_id: HirId) -> Option<HirId> {
    let hir_map = tcx.hir();
    for (parent, _) in hir_map.parent_iter(hir_id) {
        if let Some(body) = hir_map.maybe_body_owned_by(parent) {
            return Some(hir_map.body_owner(body));
        }
    }
    None
}

impl LateLintPass<'_> for DefaultNumericFallback {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        hir_id: HirId,
    ) {
        let ret_ty_bound = match fn_decl.output {
            FnRetTy::DefaultReturn(_) => None,
            FnRetTy::Return(ty) => Some(ty),
        }
        .and_then(|ty| {
            let mut infer_ty_finder = InferTyFinder::new();
            infer_ty_finder.visit_ty(ty);
            if infer_ty_finder.found {
                None
            } else if enclosing_body_owner_opt(cx.tcx, hir_id).is_some() {
                cx.typeck_results().node_type_opt(ty.hir_id)
            } else {
                Some(hir_ty_to_ty(cx.tcx, ty))
            }
        });

        let mut visitor = NumericFallbackVisitor::new(ret_ty_bound, cx);
        visitor.visit_body(body);
    }
}

struct NumericFallbackVisitor<'a, 'tcx> {
    /// Stack manages type bound of exprs. The top element holds current expr type.
    ty_bounds: Vec<Option<Ty<'tcx>>>,

    /// Ret type bound.
    ret_ty_bound: Option<Ty<'tcx>>,

    /// Break type bounds.
    break_ty_bounds: Vec<(Option<Label>, Option<Ty<'tcx>>)>,

    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> NumericFallbackVisitor<'a, 'tcx> {
    fn new(ret_ty_bound: Option<Ty<'tcx>>, cx: &'a LateContext<'tcx>) -> Self {
        Self {
            ty_bounds: vec![ret_ty_bound],
            ret_ty_bound,
            break_ty_bounds: vec![],
            cx,
        }
    }

    /// Check whether lit cause fallback or not.
    fn check_lit(&self, lit: &Lit, lit_ty: Ty<'tcx>) {
        let ty_bound = self.ty_bounds.last().unwrap();

        let should_lint = match (&lit.node, lit_ty.kind()) {
            (LitKind::Int(_, LitIntType::Unsuffixed), ty::Int(ty::IntTy::I32)) => {
                // In case integer literal is explicitly bound to i32, then suppress lint.
                ty_bound.map_or(true, |ty_bound| !matches!(ty_bound.kind(), ty::Int(IntTy::I32)))
            },

            (LitKind::Float(_, LitFloatType::Unsuffixed), ty::Float(ty::FloatTy::F64)) => {
                // In case float literal is explicitly bound to f64, then suppress lint.
                ty_bound.map_or(true, |ty_bound| !matches!(ty_bound.kind(), ty::Float(FloatTy::F64)))
            },

            _ => false,
        };

        if should_lint {
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

impl<'a, 'tcx> Visitor<'tcx> for NumericFallbackVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    #[allow(clippy::too_many_lines)]
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match (&expr.kind, *self.ty_bounds.last().unwrap()) {
            (ExprKind::Array(_), Some(last_bound)) => {
                if let ty::Array(ty, _) = last_bound.kind() {
                    self.ty_bounds.push(Some(ty))
                } else {
                    self.ty_bounds.push(None)
                }
            },

            (ExprKind::Call(func, args), _) => {
                if_chain! {
                    if let ExprKind::Path(ref func_path) = func.kind;
                    if let Some(def_id) = self.cx.qpath_res(func_path, func.hir_id).opt_def_id();
                    then {
                        let fn_sig = self.cx.tcx.fn_sig(def_id).skip_binder();
                        for (expr, bound) in args.iter().zip(fn_sig.inputs().iter()) {
                            // Push found arg type, then visit arg.
                            self.ty_bounds.push(Some(bound));
                            self.visit_expr(expr);
                            self.ty_bounds.pop();
                        }
                        return;
                    } else {
                        self.ty_bounds.push(None)
                    }
                }
            },

            (ExprKind::MethodCall(_, _, args, _), _) => {
                if let Some(def_id) = self.cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                    let fn_sig = self.cx.tcx.fn_sig(def_id).skip_binder();
                    for (expr, bound) in args.iter().zip(fn_sig.inputs().iter()) {
                        self.ty_bounds.push(Some(bound));
                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }

                self.ty_bounds.push(None)
            },

            (ExprKind::Tup(exprs), Some(last_bound)) => {
                if let ty::Tuple(tys) = last_bound.kind() {
                    for (expr, bound) in exprs.iter().zip(tys.iter()) {
                        if let GenericArgKind::Type(ty) = bound.unpack() {
                            self.ty_bounds.push(Some(ty));
                        } else {
                            self.ty_bounds.push(None);
                        }

                        self.visit_expr(expr);
                        self.ty_bounds.pop();
                    }
                    return;
                }

                self.ty_bounds.push(None)
            },

            (ExprKind::Lit(lit), _) => {
                let ty = self.cx.typeck_results().expr_ty(expr);
                self.check_lit(lit, ty);
                return;
            },

            (ExprKind::If(cond, then, else_), last_bound) => {
                // Cond has no type bound in any situation.
                self.ty_bounds.push(None);
                self.visit_expr(cond);
                self.ty_bounds.pop();

                // Propagate current bound to childs.
                self.ty_bounds.push(last_bound);
                self.visit_expr(then);
                if let Some(else_) = else_ {
                    self.visit_expr(else_);
                }
                self.ty_bounds.pop();
                return;
            },

            (ExprKind::Loop(_, label, ..), last_bound) => {
                self.break_ty_bounds.push((*label, last_bound));
                walk_expr(self, expr);
                self.break_ty_bounds.pop();
                return;
            },

            (ExprKind::Match(arg, arms, _), last_bound) => {
                // Match argument has no type bound.
                self.ty_bounds.push(None);
                self.visit_expr(arg);
                for arm in arms.iter() {
                    self.visit_pat(arm.pat);
                    if let Some(Guard::If(guard)) = arm.guard {
                        self.visit_expr(guard);
                    }
                }
                self.ty_bounds.pop();

                // Propagate current bound.
                self.ty_bounds.push(last_bound);
                for arm in arms.iter() {
                    self.visit_expr(arm.body);
                }
                self.ty_bounds.pop();
                return;
            },

            (ExprKind::Block(..), last_bound) => self.ty_bounds.push(last_bound),

            (ExprKind::Break(destination, _), _) => {
                let ty = destination.label.map_or_else(
                    || self.break_ty_bounds.last().unwrap().1,
                    |dest_label| {
                        self.break_ty_bounds
                            .iter()
                            .rev()
                            .find_map(|(loop_label, ty)| {
                                loop_label.map_or(None, |loop_label| {
                                    if loop_label.ident == dest_label.ident {
                                        Some(*ty)
                                    } else {
                                        None
                                    }
                                })
                            })
                            .unwrap()
                    },
                );
                self.ty_bounds.push(ty);
            },

            (ExprKind::Ret(_), _) => self.ty_bounds.push(self.ret_ty_bound),

            (ExprKind::Struct(qpath, fields, base), _) => {
                if_chain! {
                    if let Some(def_id) = self.cx.qpath_res(qpath, expr.hir_id).opt_def_id();
                    let ty = self.cx.tcx.type_of(def_id);
                    if let Some(adt_def) = ty.ty_adt_def();
                    if adt_def.is_struct();
                    if let Some(variant) = adt_def.variants.iter().next();
                    then {
                        let fields_def = &variant.fields;

                        // Push field type then visit each field expr.
                        for field in fields.iter() {
                            let field_ty =
                                fields_def
                                    .iter()
                                    .find_map(|f_def| {
                                        if f_def.ident == field.ident
                                            { Some(self.cx.tcx.type_of(f_def.did)) }
                                        else { None }
                                    });
                            self.ty_bounds.push(field_ty);
                            self.visit_expr(field.expr);
                            self.ty_bounds.pop();
                        }

                        // Visit base with no bound.
                        if let Some(base) = base {
                            self.ty_bounds.push(None);
                            self.visit_expr(base);
                            self.ty_bounds.pop();
                        }
                        return;
                    }
                }
                self.ty_bounds.push(None);
            },

            _ => self.ty_bounds.push(None),
        }

        walk_expr(self, expr);
        self.ty_bounds.pop();
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Local(local) => {
                let ty = local.ty.and_then(|hir_ty| {
                    let mut infer_ty_finder = InferTyFinder::new();
                    infer_ty_finder.visit_ty(hir_ty);
                    if infer_ty_finder.found {
                        None
                    } else {
                        self.cx.typeck_results().node_type_opt(hir_ty.hir_id)
                    }
                });
                self.ty_bounds.push(ty);
            },

            _ => self.ty_bounds.push(None),
        }

        walk_stmt(self, stmt);
        self.ty_bounds.pop();
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Find `hir::TyKind::Infer` is included in passed typed.
struct InferTyFinder {
    found: bool,
}

impl InferTyFinder {
    fn new() -> Self {
        Self { found: false }
    }
}

impl<'tcx> Visitor<'tcx> for InferTyFinder {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'_>) {
        match ty.kind {
            hir::TyKind::Infer => {
                self.found = true;
            },
            _ => {
                walk_ty(self, ty);
            },
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
