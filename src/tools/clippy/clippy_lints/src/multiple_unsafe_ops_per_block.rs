use clippy_utils::desugar_await;
use clippy_utils::diagnostics::span_lint_and_then;
use hir::def::{DefKind, Res};
use hir::{BlockCheckMode, ExprKind, QPath, UnOp};
use rustc_ast::{BorrowKind, Mutability};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::intravisit::{Visitor, walk_body, walk_expr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, TyCtxt, TypeckResults};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `unsafe` blocks that contain more than one unsafe operation.
    ///
    /// ### Why restrict this?
    /// Combined with `undocumented_unsafe_blocks`,
    /// this lint ensures that each unsafe operation must be independently justified.
    /// Combined with `unused_unsafe`, this lint also ensures
    /// elimination of unnecessary unsafe blocks through refactoring.
    ///
    /// ### Example
    /// ```no_run
    /// /// Reads a `char` from the given pointer.
    /// ///
    /// /// # Safety
    /// ///
    /// /// `ptr` must point to four consecutive, initialized bytes which
    /// /// form a valid `char` when interpreted in the native byte order.
    /// fn read_char(ptr: *const u8) -> char {
    ///     // SAFETY: The caller has guaranteed that the value pointed
    ///     // to by `bytes` is a valid `char`.
    ///     unsafe { char::from_u32_unchecked(*ptr.cast::<u32>()) }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// /// Reads a `char` from the given pointer.
    /// ///
    /// /// # Safety
    /// ///
    /// /// - `ptr` must be 4-byte aligned, point to four consecutive
    /// ///   initialized bytes, and be valid for reads of 4 bytes.
    /// /// - The bytes pointed to by `ptr` must represent a valid
    /// ///   `char` when interpreted in the native byte order.
    /// fn read_char(ptr: *const u8) -> char {
    ///     // SAFETY: `ptr` is 4-byte aligned, points to four consecutive
    ///     // initialized bytes, and is valid for reads of 4 bytes.
    ///     let int_value = unsafe { *ptr.cast::<u32>() };
    ///
    ///     // SAFETY: The caller has guaranteed that the four bytes
    ///     // pointed to by `bytes` represent a valid `char`.
    ///     unsafe { char::from_u32_unchecked(int_value) }
    /// }
    /// ```
    ///
    /// ### Notes
    ///
    /// - Unsafe operations only count towards the total for the innermost
    ///   enclosing `unsafe` block.
    /// - Each call to a macro expanding to unsafe operations count for one
    ///   unsafe operation.
    /// - Taking a raw pointer to a union field is always safe and will
    ///   not be considered unsafe by this lint, even when linting code written
    ///   with a specified Rust version of 1.91 or earlier (which required
    ///   using an `unsafe` block).
    #[clippy::version = "1.69.0"]
    pub MULTIPLE_UNSAFE_OPS_PER_BLOCK,
    restriction,
    "more than one unsafe operation per `unsafe` block"
}
declare_lint_pass!(MultipleUnsafeOpsPerBlock => [MULTIPLE_UNSAFE_OPS_PER_BLOCK]);

impl<'tcx> LateLintPass<'tcx> for MultipleUnsafeOpsPerBlock {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        if !matches!(block.rules, BlockCheckMode::UnsafeBlock(_)) || block.span.from_expansion() {
            return;
        }
        let unsafe_ops = UnsafeExprCollector::collect_unsafe_exprs(cx, block);
        if unsafe_ops.len() > 1 {
            span_lint_and_then(
                cx,
                MULTIPLE_UNSAFE_OPS_PER_BLOCK,
                block.span,
                format!(
                    "this `unsafe` block contains {} unsafe operations, expected only one",
                    unsafe_ops.len()
                ),
                |diag| {
                    for (msg, span) in unsafe_ops {
                        diag.span_note(span, msg);
                    }
                },
            );
        }
    }
}

struct UnsafeExprCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    typeck_results: &'tcx TypeckResults<'tcx>,
    unsafe_ops: FxHashMap<Span, &'static str>,
}

impl<'tcx> UnsafeExprCollector<'tcx> {
    fn collect_unsafe_exprs(cx: &LateContext<'tcx>, block: &'tcx hir::Block<'tcx>) -> Vec<(&'static str, Span)> {
        let mut collector = Self {
            tcx: cx.tcx,
            typeck_results: cx.typeck_results(),
            unsafe_ops: FxHashMap::default(),
        };
        collector.visit_block(block);
        #[allow(
            rustc::potential_query_instability,
            reason = "span ordering only needed inside the one expression being walked"
        )]
        let mut unsafe_ops = collector
            .unsafe_ops
            .into_iter()
            .map(|(span, msg)| (msg, span))
            .collect::<Vec<_>>();
        unsafe_ops.sort_unstable();
        unsafe_ops
    }
}

impl UnsafeExprCollector<'_> {
    fn insert_span(&mut self, span: Span, message: &'static str) {
        if span.from_expansion() {
            self.unsafe_ops.insert(
                span.source_callsite(),
                "this macro call expands into one or more unsafe operations",
            );
        } else {
            self.unsafe_ops.insert(span, message);
        }
    }
}

impl<'tcx> Visitor<'tcx> for UnsafeExprCollector<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        match expr.kind {
            // The `await` itself will desugar to two unsafe calls, but we should ignore those.
            // Instead, check the expression that is `await`ed
            _ if let Some(e) = desugar_await(expr) => {
                return self.visit_expr(e);
            },

            // Do not recurse inside an inner `unsafe` block, it will be checked on its own
            ExprKind::Block(block, _) if matches!(block.rules, BlockCheckMode::UnsafeBlock(_)) => return,

            ExprKind::InlineAsm(_) => self.insert_span(expr.span, "inline assembly used here"),

            ExprKind::AddrOf(BorrowKind::Raw, _, mut inner) => {
                while let ExprKind::Field(prefix, _) = inner.kind
                    && self.typeck_results.expr_adjustments(prefix).is_empty()
                {
                    inner = prefix;
                }
                return self.visit_expr(inner);
            },

            ExprKind::Field(e, _) => {
                if self.typeck_results.expr_ty(e).is_union() {
                    self.insert_span(expr.span, "union field access occurs here");
                }
            },

            ExprKind::Path(QPath::Resolved(
                _,
                hir::Path {
                    res:
                        Res::Def(
                            DefKind::Static {
                                mutability: Mutability::Mut,
                                ..
                            },
                            _,
                        ),
                    ..
                },
            )) => {
                self.insert_span(expr.span, "access of a mutable static occurs here");
            },

            ExprKind::Unary(UnOp::Deref, e) if self.typeck_results.expr_ty(e).is_raw_ptr() => {
                self.insert_span(expr.span, "raw pointer dereference occurs here");
            },

            ExprKind::Call(path_expr, _) => {
                let opt_sig = match *self.typeck_results.expr_ty_adjusted(path_expr).kind() {
                    ty::FnDef(id, _) => Some(self.tcx.fn_sig(id).skip_binder()),
                    ty::FnPtr(sig_tys, hdr) => Some(sig_tys.with(hdr)),
                    _ => None,
                };
                if opt_sig.is_some_and(|sig| sig.safety().is_unsafe()) {
                    self.insert_span(expr.span, "unsafe function call occurs here");
                }
            },

            ExprKind::MethodCall(..) => {
                let opt_sig = self
                    .typeck_results
                    .type_dependent_def_id(expr.hir_id)
                    .map(|def_id| self.tcx.fn_sig(def_id));
                if opt_sig.is_some_and(|sig| sig.skip_binder().safety().is_unsafe()) {
                    self.insert_span(expr.span, "unsafe method call occurs here");
                }
            },

            ExprKind::AssignOp(_, lhs, rhs) | ExprKind::Assign(lhs, rhs, _) => {
                if matches!(
                    lhs.kind,
                    ExprKind::Path(QPath::Resolved(
                        _,
                        hir::Path {
                            res: Res::Def(
                                DefKind::Static {
                                    mutability: Mutability::Mut,
                                    ..
                                },
                                _
                            ),
                            ..
                        }
                    ))
                ) {
                    self.insert_span(expr.span, "modification of a mutable static occurs here");
                    return self.visit_expr(rhs);
                }
            },

            _ => {},
        }

        walk_expr(self, expr);
    }

    fn visit_body(&mut self, body: &hir::Body<'tcx>) {
        let saved_typeck_results = self.typeck_results;
        self.typeck_results = self.tcx.typeck_body(body.id());
        walk_body(self, body);
        self.typeck_results = saved_typeck_results;
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }
}
