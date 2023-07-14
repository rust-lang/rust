use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::higher::If;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::implements_trait;
use clippy_utils::visitors::is_const_evaluatable;
use clippy_utils::{
    eq_expr_value, in_constant, is_diag_trait_item, is_trait_method, path_res, path_to_local_id, peel_blocks,
    peel_blocks_with_stmt, MaybePath,
};
use itertools::Itertools;
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir::def::Res;
use rustc_hir::{Arm, BinOpKind, Block, Expr, ExprKind, Guard, HirId, PatKind, PathSegment, PrimTy, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;
use rustc_span::Span;
use std::ops::Deref;

declare_clippy_lint! {
    /// ### What it does
    /// Identifies good opportunities for a clamp function from std or core, and suggests using it.
    ///
    /// ### Why is this bad?
    /// clamp is much shorter, easier to read, and doesn't use any control flow.
    ///
    /// ### Known issue(s)
    /// If the clamped variable is NaN this suggestion will cause the code to propagate NaN
    /// rather than returning either `max` or `min`.
    ///
    /// `clamp` functions will panic if `max < min`, `max.is_nan()`, or `min.is_nan()`.
    /// Some may consider panicking in these situations to be desirable, but it also may
    /// introduce panicking where there wasn't any before.
    ///
    /// See also [the discussion in the
    /// PR](https://github.com/rust-lang/rust-clippy/pull/9484#issuecomment-1278922613).
    ///
    /// ### Examples
    /// ```rust
    /// # let (input, min, max) = (0, -2, 1);
    /// if input > max {
    ///     max
    /// } else if input < min {
    ///     min
    /// } else {
    ///     input
    /// }
    /// # ;
    /// ```
    ///
    /// ```rust
    /// # let (input, min, max) = (0, -2, 1);
    /// input.max(min).min(max)
    /// # ;
    /// ```
    ///
    /// ```rust
    /// # let (input, min, max) = (0, -2, 1);
    /// match input {
    ///     x if x > max => max,
    ///     x if x < min => min,
    ///     x => x,
    /// }
    /// # ;
    /// ```
    ///
    /// ```rust
    /// # let (input, min, max) = (0, -2, 1);
    /// let mut x = input;
    /// if x < min { x = min; }
    /// if x > max { x = max; }
    /// ```
    /// Use instead:
    /// ```rust
    /// # let (input, min, max) = (0, -2, 1);
    /// input.clamp(min, max)
    /// # ;
    /// ```
    #[clippy::version = "1.66.0"]
    pub MANUAL_CLAMP,
    nursery,
    "using a clamp pattern instead of the clamp function"
}
impl_lint_pass!(ManualClamp => [MANUAL_CLAMP]);

pub struct ManualClamp {
    msrv: Msrv,
}

impl ManualClamp {
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

#[derive(Debug)]
struct ClampSuggestion<'tcx> {
    params: InputMinMax<'tcx>,
    span: Span,
    make_assignment: Option<&'tcx Expr<'tcx>>,
    hir_with_ignore_attr: Option<HirId>,
}

#[derive(Debug)]
struct InputMinMax<'tcx> {
    input: &'tcx Expr<'tcx>,
    min: &'tcx Expr<'tcx>,
    max: &'tcx Expr<'tcx>,
    is_float: bool,
}

impl<'tcx> LateLintPass<'tcx> for ManualClamp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !self.msrv.meets(msrvs::CLAMP) {
            return;
        }
        if !expr.span.from_expansion() && !in_constant(cx, expr.hir_id) {
            let suggestion = is_if_elseif_else_pattern(cx, expr)
                .or_else(|| is_max_min_pattern(cx, expr))
                .or_else(|| is_call_max_min_pattern(cx, expr))
                .or_else(|| is_match_pattern(cx, expr))
                .or_else(|| is_if_elseif_pattern(cx, expr));
            if let Some(suggestion) = suggestion {
                emit_suggestion(cx, &suggestion);
            }
        }
    }

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if !self.msrv.meets(msrvs::CLAMP) || in_constant(cx, block.hir_id) {
            return;
        }
        for suggestion in is_two_if_pattern(cx, block) {
            emit_suggestion(cx, &suggestion);
        }
    }
    extract_msrv_attr!(LateContext);
}

fn emit_suggestion<'tcx>(cx: &LateContext<'tcx>, suggestion: &ClampSuggestion<'tcx>) {
    let ClampSuggestion {
        params: InputMinMax {
            input,
            min,
            max,
            is_float,
        },
        span,
        make_assignment,
        hir_with_ignore_attr,
    } = suggestion;
    let input = Sugg::hir(cx, input, "..").maybe_par();
    let min = Sugg::hir(cx, min, "..");
    let max = Sugg::hir(cx, max, "..");
    let semicolon = if make_assignment.is_some() { ";" } else { "" };
    let assignment = if let Some(assignment) = make_assignment {
        let assignment = Sugg::hir(cx, assignment, "..");
        format!("{assignment} = ")
    } else {
        String::new()
    };
    let suggestion = format!("{assignment}{input}.clamp({min}, {max}){semicolon}");
    let msg = "clamp-like pattern without using clamp function";
    let lint_builder = |d: &mut Diagnostic| {
        d.span_suggestion(*span, "replace with clamp", suggestion, Applicability::MaybeIncorrect);
        if *is_float {
            d.note("clamp will panic if max < min, min.is_nan(), or max.is_nan()")
                .note("clamp returns NaN if the input is NaN");
        } else {
            d.note("clamp will panic if max < min");
        }
    };
    if let Some(hir_id) = hir_with_ignore_attr {
        span_lint_hir_and_then(cx, MANUAL_CLAMP, *hir_id, *span, msg, lint_builder);
    } else {
        span_lint_and_then(cx, MANUAL_CLAMP, *span, msg, lint_builder);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum TypeClampability {
    Float,
    Ord,
}

impl TypeClampability {
    fn is_clampable<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<TypeClampability> {
        if ty.is_floating_point() {
            Some(TypeClampability::Float)
        } else if cx
            .tcx
            .get_diagnostic_item(sym::Ord)
            .map_or(false, |id| implements_trait(cx, ty, id, &[]))
        {
            Some(TypeClampability::Ord)
        } else {
            None
        }
    }

    fn is_float(self) -> bool {
        matches!(self, TypeClampability::Float)
    }
}

/// Targets patterns like
///
/// ```
/// # let (input, min, max) = (0, -3, 12);
///
/// if input < min {
///     min
/// } else if input > max {
///     max
/// } else {
///     input
/// }
/// # ;
/// ```
fn is_if_elseif_else_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<ClampSuggestion<'tcx>> {
    if let Some(If {
        cond,
        then,
        r#else: Some(else_if),
    }) = If::hir(expr)
    && let Some(If {
        cond: else_if_cond,
        then: else_if_then,
        r#else: Some(else_body),
    }) = If::hir(peel_blocks(else_if))
    {
        let params = is_clamp_meta_pattern(
            cx,
            &BinaryOp::new(peel_blocks(cond))?,
            &BinaryOp::new(peel_blocks(else_if_cond))?,
            peel_blocks(then),
            peel_blocks(else_if_then),
            None,
        )?;
        // Contents of the else should be the resolved input.
        if !eq_expr_value(cx, params.input, peel_blocks(else_body)) {
            return None;
        }
        Some(ClampSuggestion {
            params,
            span: expr.span,
            make_assignment: None,
            hir_with_ignore_attr: None,
        })
    } else {
        None
    }
}

/// Targets patterns like
///
/// ```
/// # let (input, min_value, max_value) = (0, -3, 12);
///
/// input.max(min_value).min(max_value)
/// # ;
/// ```
fn is_max_min_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<ClampSuggestion<'tcx>> {
    if let ExprKind::MethodCall(seg_second, receiver, [arg_second], _) = &expr.kind
        && (cx.typeck_results().expr_ty_adjusted(receiver).is_floating_point() || is_trait_method(cx, expr, sym::Ord))
        && let ExprKind::MethodCall(seg_first, input, [arg_first], _) = &receiver.kind
        && (cx.typeck_results().expr_ty_adjusted(input).is_floating_point() || is_trait_method(cx, receiver, sym::Ord))
    {
        let is_float = cx.typeck_results().expr_ty_adjusted(input).is_floating_point();
        let (min, max) = match (seg_first.ident.as_str(), seg_second.ident.as_str()) {
            ("min", "max") => (arg_second, arg_first),
            ("max", "min") => (arg_first, arg_second),
            _ => return None,
        };
        Some(ClampSuggestion {
            params: InputMinMax { input, min, max, is_float },
            span: expr.span,
            make_assignment: None,
            hir_with_ignore_attr: None,
        })
    } else {
        None
    }
}

/// Targets patterns like
///
/// ```
/// # let (input, min_value, max_value) = (0, -3, 12);
/// # use std::cmp::{max, min};
/// min(max(input, min_value), max_value)
/// # ;
/// ```
fn is_call_max_min_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<ClampSuggestion<'tcx>> {
    fn segment<'tcx>(cx: &LateContext<'_>, func: &Expr<'tcx>) -> Option<FunctionType<'tcx>> {
        match func.kind {
            ExprKind::Path(QPath::Resolved(None, path)) => {
                let id = path.res.opt_def_id()?;
                match cx.tcx.get_diagnostic_name(id) {
                    Some(sym::cmp_min) => Some(FunctionType::CmpMin),
                    Some(sym::cmp_max) => Some(FunctionType::CmpMax),
                    _ if is_diag_trait_item(cx, id, sym::Ord) => {
                        Some(FunctionType::OrdOrFloat(path.segments.last().expect("infallible")))
                    },
                    _ => None,
                }
            },
            ExprKind::Path(QPath::TypeRelative(ty, seg)) => {
                matches!(path_res(cx, ty), Res::PrimTy(PrimTy::Float(_))).then(|| FunctionType::OrdOrFloat(seg))
            },
            _ => None,
        }
    }

    enum FunctionType<'tcx> {
        CmpMin,
        CmpMax,
        OrdOrFloat(&'tcx PathSegment<'tcx>),
    }

    fn check<'tcx>(
        cx: &LateContext<'tcx>,
        outer_fn: &'tcx Expr<'tcx>,
        inner_call: &'tcx Expr<'tcx>,
        outer_arg: &'tcx Expr<'tcx>,
        span: Span,
    ) -> Option<ClampSuggestion<'tcx>> {
        if let ExprKind::Call(inner_fn, [first, second]) = &inner_call.kind
            && let Some(inner_seg) = segment(cx, inner_fn)
            && let Some(outer_seg) = segment(cx, outer_fn)
        {
            let (input, inner_arg) = match (is_const_evaluatable(cx, first), is_const_evaluatable(cx, second)) {
                (true, false) => (second, first),
                (false, true) => (first, second),
                _ => return None,
            };
            let is_float = cx.typeck_results().expr_ty_adjusted(input).is_floating_point();
            let (min, max) = match (inner_seg, outer_seg) {
                (FunctionType::CmpMin, FunctionType::CmpMax) => (outer_arg, inner_arg),
                (FunctionType::CmpMax, FunctionType::CmpMin) => (inner_arg, outer_arg),
                (FunctionType::OrdOrFloat(first_segment), FunctionType::OrdOrFloat(second_segment)) => {
                    match (first_segment.ident.as_str(), second_segment.ident.as_str()) {
                        ("min", "max") => (outer_arg, inner_arg),
                        ("max", "min") => (inner_arg, outer_arg),
                        _ => return None,
                    }
                }
                _ => return None,
            };
            Some(ClampSuggestion {
                params: InputMinMax { input, min, max, is_float },
                span,
                make_assignment: None,
                hir_with_ignore_attr: None,
            })
        } else {
            None
        }
    }

    if let ExprKind::Call(outer_fn, [first, second]) = &expr.kind {
        check(cx, outer_fn, first, second, expr.span).or_else(|| check(cx, outer_fn, second, first, expr.span))
    } else {
        None
    }
}

/// Targets patterns like
///
/// ```
/// # let (input, min, max) = (0, -3, 12);
///
/// match input {
///     input if input > max => max,
///     input if input < min => min,
///     input => input,
/// }
/// # ;
/// ```
fn is_match_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<ClampSuggestion<'tcx>> {
    if let ExprKind::Match(value, [first_arm, second_arm, last_arm], rustc_hir::MatchSource::Normal) = &expr.kind {
        // Find possible min/max branches
        let minmax_values = |a: &'tcx Arm<'tcx>| {
            if let PatKind::Binding(_, var_hir_id, _, None) = &a.pat.kind
            && let Some(Guard::If(e)) = a.guard {
                Some((e, var_hir_id, a.body))
            } else {
                None
            }
        };
        let (first, first_hir_id, first_expr) = minmax_values(first_arm)?;
        let (second, second_hir_id, second_expr) = minmax_values(second_arm)?;
        let first = BinaryOp::new(first)?;
        let second = BinaryOp::new(second)?;
        if let PatKind::Binding(_, binding, _, None) = &last_arm.pat.kind
            && path_to_local_id(peel_blocks_with_stmt(last_arm.body), *binding)
            && last_arm.guard.is_none()
        {
            // Proceed as normal
        } else {
            return None;
        }
        if let Some(params) = is_clamp_meta_pattern(
            cx,
            &first,
            &second,
            first_expr,
            second_expr,
            Some((*first_hir_id, *second_hir_id)),
        ) {
            return Some(ClampSuggestion {
                params: InputMinMax {
                    input: value,
                    min: params.min,
                    max: params.max,
                    is_float: params.is_float,
                },
                span: expr.span,
                make_assignment: None,
                hir_with_ignore_attr: None,
            });
        }
    }
    None
}

/// Targets patterns like
///
/// ```
/// # let (input, min, max) = (0, -3, 12);
///
/// let mut x = input;
/// if x < min { x = min; }
/// if x > max { x = max; }
/// ```
fn is_two_if_pattern<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) -> Vec<ClampSuggestion<'tcx>> {
    block_stmt_with_last(block)
        .tuple_windows()
        .filter_map(|(maybe_set_first, maybe_set_second)| {
            if let StmtKind::Expr(first_expr) = *maybe_set_first
                && let StmtKind::Expr(second_expr) = *maybe_set_second
                && let Some(If { cond: first_cond, then: first_then, r#else: None }) = If::hir(first_expr)
                && let Some(If { cond: second_cond, then: second_then, r#else: None }) = If::hir(second_expr)
                && let ExprKind::Assign(
                    maybe_input_first_path,
                    maybe_min_max_first,
                    _
                ) = peel_blocks_with_stmt(first_then).kind
                && let ExprKind::Assign(
                    maybe_input_second_path,
                    maybe_min_max_second,
                    _
                ) = peel_blocks_with_stmt(second_then).kind
                && eq_expr_value(cx, maybe_input_first_path, maybe_input_second_path)
                && let Some(first_bin) = BinaryOp::new(first_cond)
                && let Some(second_bin) = BinaryOp::new(second_cond)
                && let Some(input_min_max) = is_clamp_meta_pattern(
                    cx,
                    &first_bin,
                    &second_bin,
                    maybe_min_max_first,
                    maybe_min_max_second,
                    None
                )
            {
                Some(ClampSuggestion {
                    params: InputMinMax {
                        input: maybe_input_first_path,
                        min: input_min_max.min,
                        max: input_min_max.max,
                        is_float: input_min_max.is_float,
                    },
                    span: first_expr.span.to(second_expr.span),
                    make_assignment: Some(maybe_input_first_path),
                    hir_with_ignore_attr: Some(first_expr.hir_id()),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Targets patterns like
///
/// ```
/// # let (mut input, min, max) = (0, -3, 12);
///
/// if input < min {
///     input = min;
/// } else if input > max {
///     input = max;
/// }
/// ```
fn is_if_elseif_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<ClampSuggestion<'tcx>> {
    if let Some(If {
        cond,
        then,
        r#else: Some(else_if),
    }) = If::hir(expr)
        && let Some(If {
            cond: else_if_cond,
            then: else_if_then,
            r#else: None,
        }) = If::hir(peel_blocks(else_if))
        && let ExprKind::Assign(
            maybe_input_first_path,
            maybe_min_max_first,
            _
        ) = peel_blocks_with_stmt(then).kind
        && let ExprKind::Assign(
            maybe_input_second_path,
            maybe_min_max_second,
            _
        ) = peel_blocks_with_stmt(else_if_then).kind
    {
        let params = is_clamp_meta_pattern(
            cx,
            &BinaryOp::new(peel_blocks(cond))?,
            &BinaryOp::new(peel_blocks(else_if_cond))?,
            peel_blocks(maybe_min_max_first),
            peel_blocks(maybe_min_max_second),
            None,
        )?;
        if !eq_expr_value(cx, maybe_input_first_path, maybe_input_second_path) {
            return None;
        }
        Some(ClampSuggestion {
            params,
            span: expr.span,
            make_assignment: Some(maybe_input_first_path),
            hir_with_ignore_attr: None,
        })
    } else {
        None
    }
}

/// `ExprKind::Binary` but more narrowly typed
#[derive(Debug, Clone, Copy)]
struct BinaryOp<'tcx> {
    op: BinOpKind,
    left: &'tcx Expr<'tcx>,
    right: &'tcx Expr<'tcx>,
}

impl<'tcx> BinaryOp<'tcx> {
    fn new(e: &'tcx Expr<'tcx>) -> Option<BinaryOp<'tcx>> {
        match &e.kind {
            ExprKind::Binary(op, left, right) => Some(BinaryOp {
                op: op.node,
                left,
                right,
            }),
            _ => None,
        }
    }

    fn flip(&self) -> Self {
        Self {
            op: match self.op {
                BinOpKind::Le => BinOpKind::Ge,
                BinOpKind::Lt => BinOpKind::Gt,
                BinOpKind::Ge => BinOpKind::Le,
                BinOpKind::Gt => BinOpKind::Lt,
                other => other,
            },
            left: self.right,
            right: self.left,
        }
    }
}

/// The clamp meta pattern is a pattern shared between many (but not all) patterns.
/// In summary, this pattern consists of two if statements that meet many criteria,
/// - binary operators that are one of [`>`, `<`, `>=`, `<=`].
/// - Both binary statements must have a shared argument
///     - Which can appear on the left or right side of either statement
///     - The binary operators must define a finite range for the shared argument. To put this in
///       the terms of Rust `std` library, the following ranges are acceptable
///         - `Range`
///         - `RangeInclusive`
///       And all other range types are not accepted. For the purposes of `clamp` it's irrelevant
///       whether the range is inclusive or not, the output is the same.
/// - The result of each if statement must be equal to the argument unique to that if statement. The
///   result can not be the shared argument in either case.
fn is_clamp_meta_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    first_bin: &BinaryOp<'tcx>,
    second_bin: &BinaryOp<'tcx>,
    first_expr: &'tcx Expr<'tcx>,
    second_expr: &'tcx Expr<'tcx>,
    // This parameters is exclusively for the match pattern.
    // It exists because the variable bindings used in that pattern
    // refer to the variable bound in the match arm, not the variable
    // bound outside of it. Fortunately due to context we know this has to
    // be the input variable, not the min or max.
    input_hir_ids: Option<(HirId, HirId)>,
) -> Option<InputMinMax<'tcx>> {
    fn check<'tcx>(
        cx: &LateContext<'tcx>,
        first_bin: &BinaryOp<'tcx>,
        second_bin: &BinaryOp<'tcx>,
        first_expr: &'tcx Expr<'tcx>,
        second_expr: &'tcx Expr<'tcx>,
        input_hir_ids: Option<(HirId, HirId)>,
        is_float: bool,
    ) -> Option<InputMinMax<'tcx>> {
        match (&first_bin.op, &second_bin.op) {
            (BinOpKind::Ge | BinOpKind::Gt, BinOpKind::Le | BinOpKind::Lt) => {
                let (min, max) = (second_expr, first_expr);
                let refers_to_input = match input_hir_ids {
                    Some((first_hir_id, second_hir_id)) => {
                        path_to_local_id(peel_blocks(first_bin.left), first_hir_id)
                            && path_to_local_id(peel_blocks(second_bin.left), second_hir_id)
                    },
                    None => eq_expr_value(cx, first_bin.left, second_bin.left),
                };
                (refers_to_input
                    && eq_expr_value(cx, first_bin.right, first_expr)
                    && eq_expr_value(cx, second_bin.right, second_expr))
                .then_some(InputMinMax {
                    input: first_bin.left,
                    min,
                    max,
                    is_float,
                })
            },
            _ => None,
        }
    }
    // First filter out any expressions with side effects
    let exprs = [
        first_bin.left,
        first_bin.right,
        second_bin.left,
        second_bin.right,
        first_expr,
        second_expr,
    ];
    let clampability = TypeClampability::is_clampable(cx, cx.typeck_results().expr_ty(first_expr))?;
    let is_float = clampability.is_float();
    if exprs.iter().any(|e| peel_blocks(e).can_have_side_effects()) {
        return None;
    }
    if !(is_ord_op(first_bin.op) && is_ord_op(second_bin.op)) {
        return None;
    }
    let cases = [
        (*first_bin, *second_bin),
        (first_bin.flip(), second_bin.flip()),
        (first_bin.flip(), *second_bin),
        (*first_bin, second_bin.flip()),
    ];

    cases.into_iter().find_map(|(first, second)| {
        check(cx, &first, &second, first_expr, second_expr, input_hir_ids, is_float).or_else(|| {
            check(
                cx,
                &second,
                &first,
                second_expr,
                first_expr,
                input_hir_ids.map(|(l, r)| (r, l)),
                is_float,
            )
        })
    })
}

fn block_stmt_with_last<'tcx>(block: &'tcx Block<'tcx>) -> impl Iterator<Item = MaybeBorrowedStmtKind<'tcx>> {
    block
        .stmts
        .iter()
        .map(|s| MaybeBorrowedStmtKind::Borrowed(&s.kind))
        .chain(
            block
                .expr
                .as_ref()
                .map(|e| MaybeBorrowedStmtKind::Owned(StmtKind::Expr(e))),
        )
}

fn is_ord_op(op: BinOpKind) -> bool {
    matches!(op, BinOpKind::Ge | BinOpKind::Gt | BinOpKind::Le | BinOpKind::Lt)
}

/// Really similar to Cow, but doesn't have a `Clone` requirement.
#[derive(Debug)]
enum MaybeBorrowedStmtKind<'a> {
    Borrowed(&'a StmtKind<'a>),
    Owned(StmtKind<'a>),
}

impl<'a> Clone for MaybeBorrowedStmtKind<'a> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(t) => Self::Borrowed(t),
            Self::Owned(StmtKind::Expr(e)) => Self::Owned(StmtKind::Expr(e)),
            Self::Owned(_) => unreachable!("Owned should only ever contain a StmtKind::Expr."),
        }
    }
}

impl<'a> Deref for MaybeBorrowedStmtKind<'a> {
    type Target = StmtKind<'a>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(t) => t,
            Self::Owned(t) => t,
        }
    }
}
