use clippy_utils::ast_utils::{eq_id, is_useless_with_eq_exprs, IdentIter};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use core::ops::{Add, AddAssign};
use if_chain::if_chain;
use rustc_ast::ast::{BinOpKind, Expr, ExprKind, StmtKind};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unlikely usages of binary operators that are almost
    /// certainly typos and/or copy/paste errors, given the other usages
    /// of binary operators nearby.
    ///
    /// ### Why is this bad?
    /// They are probably bugs and if they aren't then they look like bugs
    /// and you should add a comment explaining why you are doing such an
    /// odd set of operations.
    ///
    /// ### Known problems
    /// There may be some false positives if you are trying to do something
    /// unusual that happens to look like a typo.
    ///
    /// ### Example
    /// ```rust
    /// struct Vec3 {
    ///     x: f64,
    ///     y: f64,
    ///     z: f64,
    /// }
    ///
    /// impl Eq for Vec3 {}
    ///
    /// impl PartialEq for Vec3 {
    ///     fn eq(&self, other: &Self) -> bool {
    ///         // This should trigger the lint because `self.x` is compared to `other.y`
    ///         self.x == other.y && self.y == other.y && self.z == other.z
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # struct Vec3 {
    /// #     x: f64,
    /// #     y: f64,
    /// #     z: f64,
    /// # }
    /// // same as above except:
    /// impl PartialEq for Vec3 {
    ///     fn eq(&self, other: &Self) -> bool {
    ///         // Note we now compare other.x to self.x
    ///         self.x == other.x && self.y == other.y && self.z == other.z
    ///     }
    /// }
    /// ```
    pub SUSPICIOUS_OPERATION_GROUPINGS,
    nursery,
    "groupings of binary operations that look suspiciously like typos"
}

declare_lint_pass!(SuspiciousOperationGroupings => [SUSPICIOUS_OPERATION_GROUPINGS]);

impl EarlyLintPass for SuspiciousOperationGroupings {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if expr.span.from_expansion() {
            return;
        }

        if let Some(binops) = extract_related_binops(&expr.kind) {
            check_binops(cx, &binops.iter().collect::<Vec<_>>());

            let mut op_types = Vec::with_capacity(binops.len());
            // We could use a hashmap, etc. to avoid being O(n*m) here, but
            // we want the lints to be emitted in a consistent order. Besides,
            // m, (the number of distinct `BinOpKind`s in `binops`)
            // will often be small, and does have an upper limit.
            binops.iter().map(|b| b.op).for_each(|op| {
                if !op_types.contains(&op) {
                    op_types.push(op);
                }
            });

            for op_type in op_types {
                let ops: Vec<_> = binops.iter().filter(|b| b.op == op_type).collect();

                check_binops(cx, &ops);
            }
        }
    }
}

fn check_binops(cx: &EarlyContext<'_>, binops: &[&BinaryOp<'_>]) {
    let binop_count = binops.len();
    if binop_count < 2 {
        // Single binary operation expressions would likely be false
        // positives.
        return;
    }

    let mut one_ident_difference_count = 0;
    let mut no_difference_info = None;
    let mut double_difference_info = None;
    let mut expected_ident_loc = None;

    let mut paired_identifiers = FxHashSet::default();

    for (i, BinaryOp { left, right, op, .. }) in binops.iter().enumerate() {
        match ident_difference_expr(left, right) {
            IdentDifference::NoDifference => {
                if is_useless_with_eq_exprs(*op) {
                    // The `eq_op` lint should catch this in this case.
                    return;
                }

                no_difference_info = Some(i);
            },
            IdentDifference::Single(ident_loc) => {
                one_ident_difference_count += 1;
                if let Some(previous_expected) = expected_ident_loc {
                    if previous_expected != ident_loc {
                        // This expression doesn't match the form we're
                        // looking for.
                        return;
                    }
                } else {
                    expected_ident_loc = Some(ident_loc);
                }

                // If there was only a single difference, all other idents
                // must have been the same, and thus were paired.
                for id in skip_index(IdentIter::from(*left), ident_loc.index) {
                    paired_identifiers.insert(id);
                }
            },
            IdentDifference::Double(ident_loc1, ident_loc2) => {
                double_difference_info = Some((i, ident_loc1, ident_loc2));
            },
            IdentDifference::Multiple | IdentDifference::NonIdent => {
                // It's too hard to know whether this is a bug or not.
                return;
            },
        }
    }

    let mut applicability = Applicability::MachineApplicable;

    if let Some(expected_loc) = expected_ident_loc {
        match (no_difference_info, double_difference_info) {
            (Some(i), None) => attempt_to_emit_no_difference_lint(cx, binops, i, expected_loc),
            (None, Some((double_difference_index, ident_loc1, ident_loc2))) => {
                if_chain! {
                    if one_ident_difference_count == binop_count - 1;
                    if let Some(binop) = binops.get(double_difference_index);
                    then {
                        let changed_loc = if ident_loc1 == expected_loc {
                            ident_loc2
                        } else if ident_loc2 == expected_loc {
                            ident_loc1
                        } else {
                            // This expression doesn't match the form we're
                            // looking for.
                            return;
                        };

                        if let Some(sugg) = ident_swap_sugg(
                            cx,
                            &paired_identifiers,
                            binop,
                            changed_loc,
                            &mut applicability,
                        ) {
                            emit_suggestion(
                                cx,
                                binop.span,
                                sugg,
                                applicability,
                            );
                        }
                    }
                }
            },
            _ => {},
        }
    }
}

fn attempt_to_emit_no_difference_lint(
    cx: &EarlyContext<'_>,
    binops: &[&BinaryOp<'_>],
    i: usize,
    expected_loc: IdentLocation,
) {
    if let Some(binop) = binops.get(i).copied() {
        // We need to try and figure out which identifier we should
        // suggest using instead. Since there could be multiple
        // replacement candidates in a given expression, and we're
        // just taking the first one, we may get some bad lint
        // messages.
        let mut applicability = Applicability::MaybeIncorrect;

        // We assume that the correct ident is one used elsewhere in
        // the other binops, in a place that there was a single
        // difference between idents before.
        let old_left_ident = get_ident(binop.left, expected_loc);
        let old_right_ident = get_ident(binop.right, expected_loc);

        for b in skip_index(binops.iter(), i) {
            if_chain! {
                if let (Some(old_ident), Some(new_ident)) =
                (old_left_ident, get_ident(b.left, expected_loc));
                if old_ident != new_ident;
                if let Some(sugg) = suggestion_with_swapped_ident(
                    cx,
                    binop.left,
                    expected_loc,
                    new_ident,
                    &mut applicability,
                );
                then {
                    emit_suggestion(
                        cx,
                        binop.span,
                        replace_left_sugg(cx, binop, &sugg, &mut applicability),
                        applicability,
                    );
                    return;
                }
            }

            if_chain! {
                if let (Some(old_ident), Some(new_ident)) =
                    (old_right_ident, get_ident(b.right, expected_loc));
                if old_ident != new_ident;
                if let Some(sugg) = suggestion_with_swapped_ident(
                    cx,
                    binop.right,
                    expected_loc,
                    new_ident,
                    &mut applicability,
                );
                then {
                    emit_suggestion(
                        cx,
                        binop.span,
                        replace_right_sugg(cx, binop, &sugg, &mut applicability),
                        applicability,
                    );
                    return;
                }
            }
        }
    }
}

fn emit_suggestion(cx: &EarlyContext<'_>, span: Span, sugg: String, applicability: Applicability) {
    span_lint_and_sugg(
        cx,
        SUSPICIOUS_OPERATION_GROUPINGS,
        span,
        "this sequence of operators looks suspiciously like a bug",
        "did you mean",
        sugg,
        applicability,
    );
}

fn ident_swap_sugg(
    cx: &EarlyContext<'_>,
    paired_identifiers: &FxHashSet<Ident>,
    binop: &BinaryOp<'_>,
    location: IdentLocation,
    applicability: &mut Applicability,
) -> Option<String> {
    let left_ident = get_ident(binop.left, location)?;
    let right_ident = get_ident(binop.right, location)?;

    let sugg = match (
        paired_identifiers.contains(&left_ident),
        paired_identifiers.contains(&right_ident),
    ) {
        (true, true) | (false, false) => {
            // We don't have a good guess of what ident should be
            // used instead, in these cases.
            *applicability = Applicability::MaybeIncorrect;

            // We arbitraily choose one side to suggest changing,
            // since we don't have a better guess. If the user
            // ends up duplicating a clause, the `logic_bug` lint
            // should catch it.

            let right_suggestion = suggestion_with_swapped_ident(cx, binop.right, location, left_ident, applicability)?;

            replace_right_sugg(cx, binop, &right_suggestion, applicability)
        },
        (false, true) => {
            // We haven't seen a pair involving the left one, so
            // it's probably what is wanted.

            let right_suggestion = suggestion_with_swapped_ident(cx, binop.right, location, left_ident, applicability)?;

            replace_right_sugg(cx, binop, &right_suggestion, applicability)
        },
        (true, false) => {
            // We haven't seen a pair involving the right one, so
            // it's probably what is wanted.
            let left_suggestion = suggestion_with_swapped_ident(cx, binop.left, location, right_ident, applicability)?;

            replace_left_sugg(cx, binop, &left_suggestion, applicability)
        },
    };

    Some(sugg)
}

fn replace_left_sugg(
    cx: &EarlyContext<'_>,
    binop: &BinaryOp<'_>,
    left_suggestion: &str,
    applicability: &mut Applicability,
) -> String {
    format!(
        "{} {} {}",
        left_suggestion,
        binop.op.to_string(),
        snippet_with_applicability(cx, binop.right.span, "..", applicability),
    )
}

fn replace_right_sugg(
    cx: &EarlyContext<'_>,
    binop: &BinaryOp<'_>,
    right_suggestion: &str,
    applicability: &mut Applicability,
) -> String {
    format!(
        "{} {} {}",
        snippet_with_applicability(cx, binop.left.span, "..", applicability),
        binop.op.to_string(),
        right_suggestion,
    )
}

#[derive(Clone, Debug)]
struct BinaryOp<'exprs> {
    op: BinOpKind,
    span: Span,
    left: &'exprs Expr,
    right: &'exprs Expr,
}

impl BinaryOp<'exprs> {
    fn new(op: BinOpKind, span: Span, (left, right): (&'exprs Expr, &'exprs Expr)) -> Self {
        Self { op, span, left, right }
    }
}

fn strip_non_ident_wrappers(expr: &Expr) -> &Expr {
    let mut output = expr;
    loop {
        output = match &output.kind {
            ExprKind::Paren(ref inner) | ExprKind::Unary(_, ref inner) => inner,
            _ => {
                return output;
            },
        };
    }
}

fn extract_related_binops(kind: &ExprKind) -> Option<Vec<BinaryOp<'_>>> {
    append_opt_vecs(chained_binops(kind), if_statment_binops(kind))
}

fn if_statment_binops(kind: &ExprKind) -> Option<Vec<BinaryOp<'_>>> {
    match kind {
        ExprKind::If(ref condition, _, _) => chained_binops(&condition.kind),
        ExprKind::Paren(ref e) => if_statment_binops(&e.kind),
        ExprKind::Block(ref block, _) => {
            let mut output = None;
            for stmt in &block.stmts {
                match stmt.kind {
                    StmtKind::Expr(ref e) | StmtKind::Semi(ref e) => {
                        output = append_opt_vecs(output, if_statment_binops(&e.kind));
                    },
                    _ => {},
                }
            }
            output
        },
        _ => None,
    }
}

fn append_opt_vecs<A>(target_opt: Option<Vec<A>>, source_opt: Option<Vec<A>>) -> Option<Vec<A>> {
    match (target_opt, source_opt) {
        (Some(mut target), Some(mut source)) => {
            target.reserve(source.len());
            for op in source.drain(..) {
                target.push(op);
            }
            Some(target)
        },
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

fn chained_binops(kind: &ExprKind) -> Option<Vec<BinaryOp<'_>>> {
    match kind {
        ExprKind::Binary(_, left_outer, right_outer) => chained_binops_helper(left_outer, right_outer),
        ExprKind::Paren(ref e) | ExprKind::Unary(_, ref e) => chained_binops(&e.kind),
        _ => None,
    }
}

fn chained_binops_helper(left_outer: &'expr Expr, right_outer: &'expr Expr) -> Option<Vec<BinaryOp<'expr>>> {
    match (&left_outer.kind, &right_outer.kind) {
        (
            ExprKind::Paren(ref left_e) | ExprKind::Unary(_, ref left_e),
            ExprKind::Paren(ref right_e) | ExprKind::Unary(_, ref right_e),
        ) => chained_binops_helper(left_e, right_e),
        (ExprKind::Paren(ref left_e) | ExprKind::Unary(_, ref left_e), _) => chained_binops_helper(left_e, right_outer),
        (_, ExprKind::Paren(ref right_e) | ExprKind::Unary(_, ref right_e)) => {
            chained_binops_helper(left_outer, right_e)
        },
        (
            ExprKind::Binary(Spanned { node: left_op, .. }, ref left_left, ref left_right),
            ExprKind::Binary(Spanned { node: right_op, .. }, ref right_left, ref right_right),
        ) => match (
            chained_binops_helper(left_left, left_right),
            chained_binops_helper(right_left, right_right),
        ) {
            (Some(mut left_ops), Some(mut right_ops)) => {
                left_ops.reserve(right_ops.len());
                for op in right_ops.drain(..) {
                    left_ops.push(op);
                }
                Some(left_ops)
            },
            (Some(mut left_ops), _) => {
                left_ops.push(BinaryOp::new(*right_op, right_outer.span, (right_left, right_right)));
                Some(left_ops)
            },
            (_, Some(mut right_ops)) => {
                right_ops.insert(0, BinaryOp::new(*left_op, left_outer.span, (left_left, left_right)));
                Some(right_ops)
            },
            (None, None) => Some(vec![
                BinaryOp::new(*left_op, left_outer.span, (left_left, left_right)),
                BinaryOp::new(*right_op, right_outer.span, (right_left, right_right)),
            ]),
        },
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct IdentLocation {
    index: usize,
}

impl Add for IdentLocation {
    type Output = IdentLocation;

    fn add(self, other: Self) -> Self::Output {
        Self {
            index: self.index + other.index,
        }
    }
}

impl AddAssign for IdentLocation {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

#[derive(Clone, Copy, Debug)]
enum IdentDifference {
    NoDifference,
    Single(IdentLocation),
    Double(IdentLocation, IdentLocation),
    Multiple,
    NonIdent,
}

impl Add for IdentDifference {
    type Output = IdentDifference;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Self::NoDifference, output) | (output, Self::NoDifference) => output,
            (Self::Multiple, _)
            | (_, Self::Multiple)
            | (Self::Double(_, _), Self::Single(_))
            | (Self::Single(_) | Self::Double(_, _), Self::Double(_, _)) => Self::Multiple,
            (Self::NonIdent, _) | (_, Self::NonIdent) => Self::NonIdent,
            (Self::Single(il1), Self::Single(il2)) => Self::Double(il1, il2),
        }
    }
}

impl AddAssign for IdentDifference {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl IdentDifference {
    /// Returns true if learning about more differences will not change the value
    /// of this `IdentDifference`, and false otherwise.
    fn is_complete(&self) -> bool {
        match self {
            Self::NoDifference | Self::Single(_) | Self::Double(_, _) => false,
            Self::Multiple | Self::NonIdent => true,
        }
    }
}

fn ident_difference_expr(left: &Expr, right: &Expr) -> IdentDifference {
    ident_difference_expr_with_base_location(left, right, IdentLocation::default()).0
}

fn ident_difference_expr_with_base_location(
    left: &Expr,
    right: &Expr,
    mut base: IdentLocation,
) -> (IdentDifference, IdentLocation) {
    // Ideally, this function should not use IdentIter because it should return
    // early if the expressions have any non-ident differences. We want that early
    // return because if without that restriction the lint would lead to false
    // positives.
    //
    // But, we cannot (easily?) use a `rustc_ast::visit::Visitor`, since we need
    // the two expressions to be walked in lockstep. And without a `Visitor`, we'd
    // have to do all the AST traversal ourselves, which is a lot of work, since to
    // do it properly we'd need to be able to handle more or less every possible
    // AST node since `Item`s can be written inside `Expr`s.
    //
    // In practice, it seems likely that expressions, above a certain size, that
    // happen to use the exact same idents in the exact same order, and which are
    // not structured the same, would be rare. Therefore it seems likely that if
    // we do only the first layer of matching ourselves and eventually fallback on
    // IdentIter, then the output of this function will be almost always be correct
    // in practice.
    //
    // If it turns out that problematic cases are more prelavent than we assume,
    // then we should be able to change this function to do the correct traversal,
    // without needing to change the rest of the code.

    #![allow(clippy::enum_glob_use)]
    use ExprKind::*;

    match (
        &strip_non_ident_wrappers(left).kind,
        &strip_non_ident_wrappers(right).kind,
    ) {
        (Yield(_), Yield(_))
        | (Try(_), Try(_))
        | (Paren(_), Paren(_))
        | (Repeat(_, _), Repeat(_, _))
        | (Struct(_), Struct(_))
        | (MacCall(_), MacCall(_))
        | (LlvmInlineAsm(_), LlvmInlineAsm(_))
        | (InlineAsm(_), InlineAsm(_))
        | (Ret(_), Ret(_))
        | (Continue(_), Continue(_))
        | (Break(_, _), Break(_, _))
        | (AddrOf(_, _, _), AddrOf(_, _, _))
        | (Path(_, _), Path(_, _))
        | (Range(_, _, _), Range(_, _, _))
        | (Index(_, _), Index(_, _))
        | (Field(_, _), Field(_, _))
        | (AssignOp(_, _, _), AssignOp(_, _, _))
        | (Assign(_, _, _), Assign(_, _, _))
        | (TryBlock(_), TryBlock(_))
        | (Await(_), Await(_))
        | (Async(_, _, _), Async(_, _, _))
        | (Block(_, _), Block(_, _))
        | (Closure(_, _, _, _, _, _), Closure(_, _, _, _, _, _))
        | (Match(_, _), Match(_, _))
        | (Loop(_, _), Loop(_, _))
        | (ForLoop(_, _, _, _), ForLoop(_, _, _, _))
        | (While(_, _, _), While(_, _, _))
        | (If(_, _, _), If(_, _, _))
        | (Let(_, _, _), Let(_, _, _))
        | (Type(_, _), Type(_, _))
        | (Cast(_, _), Cast(_, _))
        | (Lit(_), Lit(_))
        | (Unary(_, _), Unary(_, _))
        | (Binary(_, _, _), Binary(_, _, _))
        | (Tup(_), Tup(_))
        | (MethodCall(_, _, _), MethodCall(_, _, _))
        | (Call(_, _), Call(_, _))
        | (ConstBlock(_), ConstBlock(_))
        | (Array(_), Array(_))
        | (Box(_), Box(_)) => {
            // keep going
        },
        _ => {
            return (IdentDifference::NonIdent, base);
        },
    }

    let mut difference = IdentDifference::NoDifference;

    for (left_attr, right_attr) in left.attrs.iter().zip(right.attrs.iter()) {
        let (new_difference, new_base) =
            ident_difference_via_ident_iter_with_base_location(left_attr, right_attr, base);
        base = new_base;
        difference += new_difference;
        if difference.is_complete() {
            return (difference, base);
        }
    }

    let (new_difference, new_base) = ident_difference_via_ident_iter_with_base_location(left, right, base);
    base = new_base;
    difference += new_difference;

    (difference, base)
}

fn ident_difference_via_ident_iter_with_base_location<Iterable: Into<IdentIter>>(
    left: Iterable,
    right: Iterable,
    mut base: IdentLocation,
) -> (IdentDifference, IdentLocation) {
    // See the note in `ident_difference_expr_with_base_location` about `IdentIter`
    let mut difference = IdentDifference::NoDifference;

    let mut left_iterator = left.into();
    let mut right_iterator = right.into();

    loop {
        match (left_iterator.next(), right_iterator.next()) {
            (Some(left_ident), Some(right_ident)) => {
                if !eq_id(left_ident, right_ident) {
                    difference += IdentDifference::Single(base);
                    if difference.is_complete() {
                        return (difference, base);
                    }
                }
            },
            (Some(_), None) | (None, Some(_)) => {
                return (IdentDifference::NonIdent, base);
            },
            (None, None) => {
                return (difference, base);
            },
        }
        base += IdentLocation { index: 1 };
    }
}

fn get_ident(expr: &Expr, location: IdentLocation) -> Option<Ident> {
    IdentIter::from(expr).nth(location.index)
}

fn suggestion_with_swapped_ident(
    cx: &EarlyContext<'_>,
    expr: &Expr,
    location: IdentLocation,
    new_ident: Ident,
    applicability: &mut Applicability,
) -> Option<String> {
    get_ident(expr, location).and_then(|current_ident| {
        if eq_id(current_ident, new_ident) {
            // We never want to suggest a non-change
            return None;
        }

        Some(format!(
            "{}{}{}",
            snippet_with_applicability(cx, expr.span.with_hi(current_ident.span.lo()), "..", applicability),
            new_ident,
            snippet_with_applicability(cx, expr.span.with_lo(current_ident.span.hi()), "..", applicability),
        ))
    })
}

fn skip_index<A, Iter>(iter: Iter, index: usize) -> impl Iterator<Item = A>
where
    Iter: Iterator<Item = A>,
{
    iter.enumerate()
        .filter_map(move |(i, a)| if i == index { None } else { Some(a) })
}
