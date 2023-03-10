use clippy_utils::{
    diagnostics::span_lint_and_then,
    visitors::{for_each_expr_with_closures, Descend, Visitable},
};
use core::ops::ControlFlow::Continue;
use hir::{
    def::{DefKind, Res},
    BlockCheckMode, ExprKind, QPath, UnOp, Unsafety,
};
use rustc_ast::Mutability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `unsafe` blocks that contain more than one unsafe operation.
    ///
    /// ### Why is this bad?
    /// Combined with `undocumented_unsafe_blocks`,
    /// this lint ensures that each unsafe operation must be independently justified.
    /// Combined with `unused_unsafe`, this lint also ensures
    /// elimination of unnecessary unsafe blocks through refactoring.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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
    #[clippy::version = "1.68.0"]
    pub MULTIPLE_UNSAFE_OPS_PER_BLOCK,
    restriction,
    "more than one unsafe operation per `unsafe` block"
}
declare_lint_pass!(MultipleUnsafeOpsPerBlock => [MULTIPLE_UNSAFE_OPS_PER_BLOCK]);

impl<'tcx> LateLintPass<'tcx> for MultipleUnsafeOpsPerBlock {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        if !matches!(block.rules, BlockCheckMode::UnsafeBlock(_)) || in_external_macro(cx.tcx.sess, block.span) {
            return;
        }
        let mut unsafe_ops = vec![];
        collect_unsafe_exprs(cx, block, &mut unsafe_ops);
        if unsafe_ops.len() > 1 {
            span_lint_and_then(
                cx,
                MULTIPLE_UNSAFE_OPS_PER_BLOCK,
                block.span,
                &format!(
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

fn collect_unsafe_exprs<'tcx>(
    cx: &LateContext<'tcx>,
    node: impl Visitable<'tcx>,
    unsafe_ops: &mut Vec<(&'static str, Span)>,
) {
    for_each_expr_with_closures(cx, node, |expr| {
        match expr.kind {
            ExprKind::InlineAsm(_) => unsafe_ops.push(("inline assembly used here", expr.span)),

            ExprKind::Field(e, _) => {
                if cx.typeck_results().expr_ty(e).is_union() {
                    unsafe_ops.push(("union field access occurs here", expr.span));
                }
            },

            ExprKind::Path(QPath::Resolved(
                _,
                hir::Path {
                    res: Res::Def(DefKind::Static(Mutability::Mut), _),
                    ..
                },
            )) => {
                unsafe_ops.push(("access of a mutable static occurs here", expr.span));
            },

            ExprKind::Unary(UnOp::Deref, e) if cx.typeck_results().expr_ty_adjusted(e).is_unsafe_ptr() => {
                unsafe_ops.push(("raw pointer dereference occurs here", expr.span));
            },

            ExprKind::Call(path_expr, _) => {
                let sig = match *cx.typeck_results().expr_ty(path_expr).kind() {
                    ty::FnDef(id, _) => cx.tcx.fn_sig(id).skip_binder(),
                    ty::FnPtr(sig) => sig,
                    _ => return Continue(Descend::Yes),
                };
                if sig.unsafety() == Unsafety::Unsafe {
                    unsafe_ops.push(("unsafe function call occurs here", expr.span));
                }
            },

            ExprKind::MethodCall(..) => {
                if let Some(sig) = cx
                    .typeck_results()
                    .type_dependent_def_id(expr.hir_id)
                    .map(|def_id| cx.tcx.fn_sig(def_id))
                {
                    if sig.0.unsafety() == Unsafety::Unsafe {
                        unsafe_ops.push(("unsafe method call occurs here", expr.span));
                    }
                }
            },

            ExprKind::AssignOp(_, lhs, rhs) | ExprKind::Assign(lhs, rhs, _) => {
                if matches!(
                    lhs.kind,
                    ExprKind::Path(QPath::Resolved(
                        _,
                        hir::Path {
                            res: Res::Def(DefKind::Static(Mutability::Mut), _),
                            ..
                        }
                    ))
                ) {
                    unsafe_ops.push(("modification of a mutable static occurs here", expr.span));
                    collect_unsafe_exprs(cx, rhs, unsafe_ops);
                    return Continue(Descend::No);
                }
            },

            _ => {},
        };

        Continue::<(), _>(Descend::Yes)
    });
}
