use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::higher::{get_vec_init_kind, VecInitKind};
use clippy_utils::ty::{is_type_diagnostic_item, is_uninit_value_valid_for_ty};
use clippy_utils::{is_lint_allowed, path_to_local_id, peel_hir_expr_while, SpanlessEq};
use rustc_hir::{Block, Expr, ExprKind, HirId, PatKind, PathSegment, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

// TODO: add `ReadBuf` (RFC 2930) in "How to fix" once it is available in std
declare_clippy_lint! {
    /// ### What it does
    /// Checks for `set_len()` call that creates `Vec` with uninitialized elements.
    /// This is commonly caused by calling `set_len()` right after allocating or
    /// reserving a buffer with `new()`, `default()`, `with_capacity()`, or `reserve()`.
    ///
    /// ### Why is this bad?
    /// It creates a `Vec` with uninitialized data, which leads to
    /// undefined behavior with most safe operations. Notably, uninitialized
    /// `Vec<u8>` must not be used with generic `Read`.
    ///
    /// Moreover, calling `set_len()` on a `Vec` created with `new()` or `default()`
    /// creates out-of-bound values that lead to heap memory corruption when used.
    ///
    /// ### Known Problems
    /// This lint only checks directly adjacent statements.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let mut vec: Vec<u8> = Vec::with_capacity(1000);
    /// unsafe { vec.set_len(1000); }
    /// reader.read(&mut vec); // undefined behavior!
    /// ```
    ///
    /// ### How to fix?
    /// 1. Use an initialized buffer:
    ///    ```rust,ignore
    ///    let mut vec: Vec<u8> = vec![0; 1000];
    ///    reader.read(&mut vec);
    ///    ```
    /// 2. Wrap the content in `MaybeUninit`:
    ///    ```rust,ignore
    ///    let mut vec: Vec<MaybeUninit<T>> = Vec::with_capacity(1000);
    ///    vec.set_len(1000);  // `MaybeUninit` can be uninitialized
    ///    ```
    /// 3. If you are on 1.60.0 or later, `Vec::spare_capacity_mut()` is available:
    ///    ```rust,ignore
    ///    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    ///    let remaining = vec.spare_capacity_mut();  // `&mut [MaybeUninit<u8>]`
    ///    // perform initialization with `remaining`
    ///    vec.set_len(...);  // Safe to call `set_len()` on initialized part
    ///    ```
    #[clippy::version = "1.58.0"]
    pub UNINIT_VEC,
    correctness,
    "Vec with uninitialized data"
}

declare_lint_pass!(UninitVec => [UNINIT_VEC]);

// FIXME: update to a visitor-based implementation.
// Threads: https://github.com/rust-lang/rust-clippy/pull/7682#discussion_r710998368
impl<'tcx> LateLintPass<'tcx> for UninitVec {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        if !in_external_macro(cx.tcx.sess, block.span) {
            for w in block.stmts.windows(2) {
                if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = w[1].kind {
                    handle_uninit_vec_pair(cx, &w[0], expr);
                }
            }

            if let (Some(stmt), Some(expr)) = (block.stmts.last(), block.expr) {
                handle_uninit_vec_pair(cx, stmt, expr);
            }
        }
    }
}

fn handle_uninit_vec_pair<'tcx>(
    cx: &LateContext<'tcx>,
    maybe_init_or_reserve: &'tcx Stmt<'tcx>,
    maybe_set_len: &'tcx Expr<'tcx>,
) {
    if_chain! {
        if let Some(vec) = extract_init_or_reserve_target(cx, maybe_init_or_reserve);
        if let Some((set_len_self, call_span)) = extract_set_len_self(cx, maybe_set_len);
        if vec.location.eq_expr(cx, set_len_self);
        if let ty::Ref(_, vec_ty, _) = cx.typeck_results().expr_ty_adjusted(set_len_self).kind();
        if let ty::Adt(_, substs) = vec_ty.kind();
        // `#[allow(...)]` attribute can be set on enclosing unsafe block of `set_len()`
        if !is_lint_allowed(cx, UNINIT_VEC, maybe_set_len.hir_id);
        then {
            if vec.has_capacity() {
                // with_capacity / reserve -> set_len

                // Check T of Vec<T>
                if !is_uninit_value_valid_for_ty(cx, substs.type_at(0)) {
                    // FIXME: #7698, false positive of the internal lints
                    #[expect(clippy::collapsible_span_lint_calls)]
                    span_lint_and_then(
                        cx,
                        UNINIT_VEC,
                        vec![call_span, maybe_init_or_reserve.span],
                        "calling `set_len()` immediately after reserving a buffer creates uninitialized values",
                        |diag| {
                            diag.help("initialize the buffer or wrap the content in `MaybeUninit`");
                        },
                    );
                }
            } else {
                // new / default -> set_len
                span_lint(
                    cx,
                    UNINIT_VEC,
                    vec![call_span, maybe_init_or_reserve.span],
                    "calling `set_len()` on empty `Vec` creates out-of-bound values",
                );
            }
        }
    }
}

/// The target `Vec` that is initialized or reserved
#[derive(Clone, Copy)]
struct TargetVec<'tcx> {
    location: VecLocation<'tcx>,
    /// `None` if `reserve()`
    init_kind: Option<VecInitKind>,
}

impl TargetVec<'_> {
    pub fn has_capacity(self) -> bool {
        !matches!(self.init_kind, Some(VecInitKind::New | VecInitKind::Default))
    }
}

#[derive(Clone, Copy)]
enum VecLocation<'tcx> {
    Local(HirId),
    Expr(&'tcx Expr<'tcx>),
}

impl<'tcx> VecLocation<'tcx> {
    pub fn eq_expr(self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
        match self {
            VecLocation::Local(hir_id) => path_to_local_id(expr, hir_id),
            VecLocation::Expr(self_expr) => SpanlessEq::new(cx).eq_expr(self_expr, expr),
        }
    }
}

/// Finds the target location where the result of `Vec` initialization is stored
/// or `self` expression for `Vec::reserve()`.
fn extract_init_or_reserve_target<'tcx>(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'tcx>) -> Option<TargetVec<'tcx>> {
    match stmt.kind {
        StmtKind::Local(local) => {
            if_chain! {
                if let Some(init_expr) = local.init;
                if let PatKind::Binding(_, hir_id, _, None) = local.pat.kind;
                if let Some(init_kind) = get_vec_init_kind(cx, init_expr);
                then {
                    return Some(TargetVec {
                        location: VecLocation::Local(hir_id),
                        init_kind: Some(init_kind),
                    })
                }
            }
        },
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => match expr.kind {
            ExprKind::Assign(lhs, rhs, _span) => {
                if let Some(init_kind) = get_vec_init_kind(cx, rhs) {
                    return Some(TargetVec {
                        location: VecLocation::Expr(lhs),
                        init_kind: Some(init_kind),
                    });
                }
            },
            ExprKind::MethodCall(path, self_expr, [_], _) if is_reserve(cx, path, self_expr) => {
                return Some(TargetVec {
                    location: VecLocation::Expr(self_expr),
                    init_kind: None,
                });
            },
            _ => (),
        },
        StmtKind::Item(_) => (),
    }
    None
}

fn is_reserve(cx: &LateContext<'_>, path: &PathSegment<'_>, self_expr: &Expr<'_>) -> bool {
    is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(self_expr).peel_refs(), sym::Vec)
        && path.ident.name.as_str() == "reserve"
}

/// Returns self if the expression is `Vec::set_len()`
fn extract_set_len_self<'tcx>(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> Option<(&'tcx Expr<'tcx>, Span)> {
    // peel unsafe blocks in `unsafe { vec.set_len() }`
    let expr = peel_hir_expr_while(expr, |e| {
        if let ExprKind::Block(block, _) = e.kind {
            // Extract the first statement/expression
            match (block.stmts.get(0).map(|stmt| &stmt.kind), block.expr) {
                (None, Some(expr)) => Some(expr),
                (Some(StmtKind::Expr(expr) | StmtKind::Semi(expr)), _) => Some(expr),
                _ => None,
            }
        } else {
            None
        }
    });
    match expr.kind {
        ExprKind::MethodCall(path, self_expr, [_], _) => {
            let self_type = cx.typeck_results().expr_ty(self_expr).peel_refs();
            if is_type_diagnostic_item(cx, self_type, sym::Vec) && path.ident.name.as_str() == "set_len" {
                Some((self_expr, expr.span))
            } else {
                None
            }
        },
        _ => None,
    }
}
