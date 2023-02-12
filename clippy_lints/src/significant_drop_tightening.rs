use crate::FxHashSet;
use clippy_utils::{
    diagnostics::span_lint_and_then,
    get_attr,
    source::{indent_of, snippet},
};
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir::{
    self as hir,
    intravisit::{walk_expr, Visitor},
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{subst::GenericArgKind, Ty, TypeAndMut};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::Ident, Span, DUMMY_SP};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Searches for elements marked with `#[clippy::significant_drop]` that could be early
    /// dropped but are in fact dropped at the end of their scopes. In other words, enforces the
    /// "tightening" of their possible lifetimes.
    ///
    /// ### Why is this bad?
    ///
    /// Elements marked with `#[clippy::has_significant_drop]` are generally synchronizing
    /// primitives that manage shared resources, as such, it is desired to release them as soon as
    /// possible to avoid unnecessary resource contention.
    ///
    /// ### Example
    ///
    /// ```rust,ignore
    /// fn main() {
    ///   let lock = some_sync_resource.lock();
    ///   let owned_rslt = lock.do_stuff_with_resource();
    ///   // Only `owned_rslt` is needed but `lock` is still held.
    ///   do_heavy_computation_that_takes_time(owned_rslt);
    /// }
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust,ignore
    /// fn main() {
    ///     let owned_rslt = some_sync_resource.lock().do_stuff_with_resource();
    ///     do_heavy_computation_that_takes_time(owned_rslt);
    /// }
    /// ```
    #[clippy::version = "1.67.0"]
    pub SIGNIFICANT_DROP_TIGHTENING,
    nursery,
    "Searches for elements marked with `#[clippy::has_significant_drop]` that could be early dropped but are in fact dropped at the end of their scopes"
}

impl_lint_pass!(SignificantDropTightening<'_> => [SIGNIFICANT_DROP_TIGHTENING]);

#[derive(Default)]
pub struct SignificantDropTightening<'tcx> {
    /// Auxiliary structure used to avoid having to verify the same type multiple times.
    seen_types: FxHashSet<Ty<'tcx>>,
}

impl<'tcx> SignificantDropTightening<'tcx> {
    /// Verifies if the expression is of type `drop(some_lock_path)` to assert that the temporary
    /// is already being dropped before the end of its scope.
    fn has_drop(expr: &'tcx hir::Expr<'_>, init_bind_ident: Ident) -> bool {
        if let hir::ExprKind::Call(fun, args) = expr.kind
            && let hir::ExprKind::Path(hir::QPath::Resolved(_, fun_path)) = &fun.kind
            && let [fun_ident, ..] = fun_path.segments
            && fun_ident.ident.name == rustc_span::sym::drop
            && let [first_arg, ..] = args
            && let hir::ExprKind::Path(hir::QPath::Resolved(_, arg_path)) = &first_arg.kind
            && let [first_arg_ps, .. ] = arg_path.segments
        {
            first_arg_ps.ident == init_bind_ident
        }
        else {
            false
        }
    }

    /// Tries to find types marked with `#[has_significant_drop]` of an expression `expr` that is
    /// originated from `stmt` and then performs common logic on `sdap`.
    fn modify_sdap_if_sig_drop_exists(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        idx: usize,
        sdap: &mut SigDropAuxParams,
        stmt: &'tcx hir::Stmt<'_>,
        cb: impl Fn(&mut SigDropAuxParams),
    ) {
        let mut sig_drop_finder = SigDropFinder::new(cx, &mut self.seen_types);
        sig_drop_finder.visit_expr(expr);
        if sig_drop_finder.has_sig_drop {
            cb(sdap);
            if sdap.number_of_stmts > 0 {
                sdap.last_use_stmt_idx = idx;
                sdap.last_use_stmt_span = stmt.span;
                if let hir::ExprKind::MethodCall(_, _, _, span) = expr.kind {
                    sdap.last_use_method_span = span;
                }
            }
            sdap.number_of_stmts = sdap.number_of_stmts.wrapping_add(1);
        }
    }

    /// Shows a generic overall message as well as specialized messages depending on the usage.
    fn set_suggestions(cx: &LateContext<'tcx>, block_span: Span, diag: &mut Diagnostic, sdap: &SigDropAuxParams) {
        match sdap.number_of_stmts {
            0 | 1 => {},
            2 => {
                let indent = " ".repeat(indent_of(cx, sdap.last_use_stmt_span).unwrap_or(0));
                let init_method = snippet(cx, sdap.init_method_span, "..");
                let usage_method = snippet(cx, sdap.last_use_method_span, "..");
                let stmt = if let Some(last_use_bind_span) = sdap.last_use_bind_span {
                    format!(
                        "\n{indent}let {} = {init_method}.{usage_method};",
                        snippet(cx, last_use_bind_span, ".."),
                    )
                } else {
                    format!("\n{indent}{init_method}.{usage_method};")
                };
                diag.span_suggestion_verbose(
                    sdap.init_stmt_span,
                    "merge the temporary construction with its single usage",
                    stmt,
                    Applicability::MaybeIncorrect,
                );
                diag.span_suggestion(
                    sdap.last_use_stmt_span,
                    "remove separated single usage",
                    "",
                    Applicability::MaybeIncorrect,
                );
            },
            _ => {
                diag.span_suggestion(
                    sdap.last_use_stmt_span.shrink_to_hi(),
                    "drop the temporary after the end of its last usage",
                    format!(
                        "\n{}drop({});",
                        " ".repeat(indent_of(cx, sdap.last_use_stmt_span).unwrap_or(0)),
                        sdap.init_bind_ident
                    ),
                    Applicability::MaybeIncorrect,
                );
            },
        }
        diag.note("this might lead to unnecessary resource contention");
        diag.span_label(
            block_span,
            format!(
                "temporary `{}` is currently being dropped at the end of its contained scope",
                sdap.init_bind_ident
            ),
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for SignificantDropTightening<'tcx> {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        let mut sdap = SigDropAuxParams::default();
        for (idx, stmt) in block.stmts.iter().enumerate() {
            match stmt.kind {
                hir::StmtKind::Expr(expr) => self.modify_sdap_if_sig_drop_exists(
                    cx,
                    expr,
                    idx,
                    &mut sdap,
                    stmt,
                    |_| {}
                ),
                hir::StmtKind::Local(local) if let Some(expr) = local.init => self.modify_sdap_if_sig_drop_exists(
                    cx,
                    expr,
                    idx,
                    &mut sdap,
                    stmt,
                    |local_sdap| {
                        if local_sdap.number_of_stmts == 0 {
                            if let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind {
                                local_sdap.init_bind_ident = ident;
                            }
                            if let hir::ExprKind::MethodCall(_, local_expr, _, span) = expr.kind {
                                local_sdap.init_method_span = local_expr.span.to(span);
                            }
                            local_sdap.init_stmt_span = stmt.span;
                        }
                        else if let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind {
                            local_sdap.last_use_bind_span = Some(ident.span);
                        }
                    }
                ),
                hir::StmtKind::Semi(expr) => {
                    if Self::has_drop(expr, sdap.init_bind_ident) {
                        return;
                    }
                    self.modify_sdap_if_sig_drop_exists(cx, expr, idx, &mut sdap, stmt, |_| {});
                },
                _ => continue
            };
        }
        if sdap.number_of_stmts > 1 && {
            let last_stmts_idx = block.stmts.len().wrapping_sub(1);
            sdap.last_use_stmt_idx != last_stmts_idx
        } {
            span_lint_and_then(
                cx,
                SIGNIFICANT_DROP_TIGHTENING,
                sdap.init_bind_ident.span,
                "temporary with significant `Drop` can be early dropped",
                |diag| {
                    Self::set_suggestions(cx, block.span, diag, &sdap);
                },
            );
        }
    }
}

/// Auxiliary parameters used on each block check.
struct SigDropAuxParams {
    /// The binding or variable that references the initial construction of the type marked with
    /// `#[has_significant_drop]`.
    init_bind_ident: Ident,
    /// Similar to `init_bind_ident` but encompasses the right-hand method call.
    init_method_span: Span,
    /// Similar to `init_bind_ident` but encompasses the whole contained statement.
    init_stmt_span: Span,

    /// The last visited binding or variable span within a block that had any referenced inner type
    /// marked with `#[has_significant_drop]`.
    last_use_bind_span: Option<Span>,
    /// Index of the last visited statement within a block that had any referenced inner type
    /// marked with `#[has_significant_drop]`.
    last_use_stmt_idx: usize,
    /// Similar to `last_use_bind_span` but encompasses the whole contained statement.
    last_use_stmt_span: Span,
    /// Similar to `last_use_bind_span` but encompasses the right-hand method call.
    last_use_method_span: Span,

    /// Total number of statements within a block that have any referenced inner type marked with
    /// `#[has_significant_drop]`.
    number_of_stmts: usize,
}

impl Default for SigDropAuxParams {
    fn default() -> Self {
        Self {
            init_bind_ident: Ident::empty(),
            init_method_span: DUMMY_SP,
            init_stmt_span: DUMMY_SP,
            last_use_bind_span: None,
            last_use_method_span: DUMMY_SP,
            last_use_stmt_idx: 0,
            last_use_stmt_span: DUMMY_SP,
            number_of_stmts: 0,
        }
    }
}

/// Checks the existence of the `#[has_significant_drop]` attribute
struct SigDropChecker<'cx, 'sdt, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    seen_types: &'sdt mut FxHashSet<Ty<'tcx>>,
}

impl<'cx, 'sdt, 'tcx> SigDropChecker<'cx, 'sdt, 'tcx> {
    pub(crate) fn new(cx: &'cx LateContext<'tcx>, seen_types: &'sdt mut FxHashSet<Ty<'tcx>>) -> Self {
        seen_types.clear();
        Self { cx, seen_types }
    }

    pub(crate) fn has_sig_drop_attr(&mut self, ty: Ty<'tcx>) -> bool {
        if let Some(adt) = ty.ty_adt_def() {
            let mut iter = get_attr(
                self.cx.sess(),
                self.cx.tcx.get_attrs_unchecked(adt.did()),
                "has_significant_drop",
            );
            if iter.next().is_some() {
                return true;
            }
        }
        match ty.kind() {
            rustc_middle::ty::Adt(a, b) => {
                for f in a.all_fields() {
                    let ty = f.ty(self.cx.tcx, b);
                    if !self.has_seen_ty(ty) && self.has_sig_drop_attr(ty) {
                        return true;
                    }
                }
                for generic_arg in b.iter() {
                    if let GenericArgKind::Type(ty) = generic_arg.unpack() {
                        if self.has_sig_drop_attr(ty) {
                            return true;
                        }
                    }
                }
                false
            },
            rustc_middle::ty::Array(ty, _)
            | rustc_middle::ty::RawPtr(TypeAndMut { ty, .. })
            | rustc_middle::ty::Ref(_, ty, _)
            | rustc_middle::ty::Slice(ty) => self.has_sig_drop_attr(*ty),
            _ => false,
        }
    }

    fn has_seen_ty(&mut self, ty: Ty<'tcx>) -> bool {
        !self.seen_types.insert(ty)
    }
}

/// Performs recursive calls to find any inner type marked with `#[has_significant_drop]`.
struct SigDropFinder<'cx, 'sdt, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    has_sig_drop: bool,
    sig_drop_checker: SigDropChecker<'cx, 'sdt, 'tcx>,
}

impl<'cx, 'sdt, 'tcx> SigDropFinder<'cx, 'sdt, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>, seen_types: &'sdt mut FxHashSet<Ty<'tcx>>) -> Self {
        Self {
            cx,
            has_sig_drop: false,
            sig_drop_checker: SigDropChecker::new(cx, seen_types),
        }
    }
}

impl<'cx, 'sdt, 'tcx> Visitor<'tcx> for SigDropFinder<'cx, 'sdt, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'_>) {
        if self
            .sig_drop_checker
            .has_sig_drop_attr(self.cx.typeck_results().expr_ty(ex))
        {
            self.has_sig_drop = true;
            return;
        }

        match ex.kind {
            hir::ExprKind::MethodCall(_, expr, ..) => {
                self.visit_expr(expr);
            },
            hir::ExprKind::Array(..)
            | hir::ExprKind::Assign(..)
            | hir::ExprKind::AssignOp(..)
            | hir::ExprKind::Binary(..)
            | hir::ExprKind::Box(..)
            | hir::ExprKind::Call(..)
            | hir::ExprKind::Field(..)
            | hir::ExprKind::If(..)
            | hir::ExprKind::Index(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Repeat(..)
            | hir::ExprKind::Ret(..)
            | hir::ExprKind::Tup(..)
            | hir::ExprKind::Unary(..)
            | hir::ExprKind::Yield(..) => {
                walk_expr(self, ex);
            },
            _ => {},
        }
    }
}
