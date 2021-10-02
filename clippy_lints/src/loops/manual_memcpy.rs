use super::{get_span_of_entire_for_loop, IncrementVisitor, InitializeVisitor, MANUAL_MEMCPY};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_enclosing_block, higher, path_to_local, sugg};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir::intravisit::walk_block;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, HirId, Pat, PatKind, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;
use std::iter::Iterator;

/// Checks for for loops that sequentially copy items from one slice-like
/// object to another.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
) -> bool {
    if let Some(higher::Range {
        start: Some(start),
        end: Some(end),
        limits,
    }) = higher::Range::hir(arg)
    {
        // the var must be a single name
        if let PatKind::Binding(_, canonical_id, _, _) = pat.kind {
            let mut starts = vec![Start {
                id: canonical_id,
                kind: StartKind::Range,
            }];

            // This is one of few ways to return different iterators
            // derived from: https://stackoverflow.com/questions/29760668/conditionally-iterate-over-one-of-several-possible-iterators/52064434#52064434
            let mut iter_a = None;
            let mut iter_b = None;

            if let ExprKind::Block(block, _) = body.kind {
                if let Some(loop_counters) = get_loop_counters(cx, block, expr) {
                    starts.extend(loop_counters);
                }
                iter_a = Some(get_assignments(block, &starts));
            } else {
                iter_b = Some(get_assignment(body));
            }

            let assignments = iter_a.into_iter().flatten().chain(iter_b.into_iter());

            let big_sugg = assignments
                // The only statements in the for loops can be indexed assignments from
                // indexed retrievals (except increments of loop counters).
                .map(|o| {
                    o.and_then(|(lhs, rhs)| {
                        let rhs = fetch_cloned_expr(rhs);
                        if_chain! {
                            if let ExprKind::Index(base_left, idx_left) = lhs.kind;
                            if let ExprKind::Index(base_right, idx_right) = rhs.kind;
                            if is_slice_like(cx, cx.typeck_results().expr_ty(base_left));
                            if is_slice_like(cx, cx.typeck_results().expr_ty(base_right));
                            if let Some((start_left, offset_left)) = get_details_from_idx(cx, idx_left, &starts);
                            if let Some((start_right, offset_right)) = get_details_from_idx(cx, idx_right, &starts);

                            // Source and destination must be different
                            if path_to_local(base_left) != path_to_local(base_right);
                            then {
                                Some((IndexExpr { base: base_left, idx: start_left, idx_offset: offset_left },
                                    IndexExpr { base: base_right, idx: start_right, idx_offset: offset_right }))
                            } else {
                                None
                            }
                        }
                    })
                })
                .map(|o| o.map(|(dst, src)| build_manual_memcpy_suggestion(cx, start, end, limits, &dst, &src)))
                .collect::<Option<Vec<_>>>()
                .filter(|v| !v.is_empty())
                .map(|v| v.join("\n    "));

            if let Some(big_sugg) = big_sugg {
                span_lint_and_sugg(
                    cx,
                    MANUAL_MEMCPY,
                    get_span_of_entire_for_loop(expr),
                    "it looks like you're manually copying between slices",
                    "try replacing the loop by",
                    big_sugg,
                    Applicability::Unspecified,
                );
                return true;
            }
        }
    }
    false
}

fn build_manual_memcpy_suggestion<'tcx>(
    cx: &LateContext<'tcx>,
    start: &Expr<'_>,
    end: &Expr<'_>,
    limits: ast::RangeLimits,
    dst: &IndexExpr<'_>,
    src: &IndexExpr<'_>,
) -> String {
    fn print_offset(offset: MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        if offset.as_str() == "0" {
            sugg::EMPTY.into()
        } else {
            offset
        }
    }

    let print_limit = |end: &Expr<'_>, end_str: &str, base: &Expr<'_>, sugg: MinifyingSugg<'static>| {
        if_chain! {
            if let ExprKind::MethodCall(method, _, len_args, _) = end.kind;
            if method.ident.name == sym::len;
            if len_args.len() == 1;
            if let Some(arg) = len_args.get(0);
            if path_to_local(arg) == path_to_local(base);
            then {
                if sugg.as_str() == end_str {
                    sugg::EMPTY.into()
                } else {
                    sugg
                }
            } else {
                match limits {
                    ast::RangeLimits::Closed => {
                        sugg + &sugg::ONE.into()
                    },
                    ast::RangeLimits::HalfOpen => sugg,
                }
            }
        }
    };

    let start_str = Sugg::hir(cx, start, "").into();
    let end_str: MinifyingSugg<'_> = Sugg::hir(cx, end, "").into();

    let print_offset_and_limit = |idx_expr: &IndexExpr<'_>| match idx_expr.idx {
        StartKind::Range => (
            print_offset(apply_offset(&start_str, &idx_expr.idx_offset)).into_sugg(),
            print_limit(
                end,
                end_str.as_str(),
                idx_expr.base,
                apply_offset(&end_str, &idx_expr.idx_offset),
            )
            .into_sugg(),
        ),
        StartKind::Counter { initializer } => {
            let counter_start = Sugg::hir(cx, initializer, "").into();
            (
                print_offset(apply_offset(&counter_start, &idx_expr.idx_offset)).into_sugg(),
                print_limit(
                    end,
                    end_str.as_str(),
                    idx_expr.base,
                    apply_offset(&end_str, &idx_expr.idx_offset) + &counter_start - &start_str,
                )
                .into_sugg(),
            )
        },
    };

    let (dst_offset, dst_limit) = print_offset_and_limit(dst);
    let (src_offset, src_limit) = print_offset_and_limit(src);

    let dst_base_str = snippet(cx, dst.base.span, "???");
    let src_base_str = snippet(cx, src.base.span, "???");

    let dst = if dst_offset == sugg::EMPTY && dst_limit == sugg::EMPTY {
        dst_base_str
    } else {
        format!(
            "{}[{}..{}]",
            dst_base_str,
            dst_offset.maybe_par(),
            dst_limit.maybe_par()
        )
        .into()
    };

    format!(
        "{}.clone_from_slice(&{}[{}..{}]);",
        dst,
        src_base_str,
        src_offset.maybe_par(),
        src_limit.maybe_par()
    )
}

/// a wrapper of `Sugg`. Besides what `Sugg` do, this removes unnecessary `0`;
/// and also, it avoids subtracting a variable from the same one by replacing it with `0`.
/// it exists for the convenience of the overloaded operators while normal functions can do the
/// same.
#[derive(Clone)]
struct MinifyingSugg<'a>(Sugg<'a>);

impl<'a> MinifyingSugg<'a> {
    fn as_str(&self) -> &str {
        // HACK: Don't sync to Clippy! Required because something with the `or_patterns` feature
        // changed and this would now require parentheses.
        match &self.0 {
            Sugg::NonParen(s) | Sugg::MaybeParen(s) | Sugg::BinOp(_, s) => s.as_ref(),
        }
    }

    fn into_sugg(self) -> Sugg<'a> {
        self.0
    }
}

impl<'a> From<Sugg<'a>> for MinifyingSugg<'a> {
    fn from(sugg: Sugg<'a>) -> Self {
        Self(sugg)
    }
}

impl std::ops::Add for &MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn add(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            ("0", _) => rhs.clone(),
            (_, "0") => self.clone(),
            (_, _) => (&self.0 + &rhs.0).into(),
        }
    }
}

impl std::ops::Sub for &MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn sub(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            (_, "0") => self.clone(),
            ("0", _) => (-rhs.0.clone()).into(),
            (x, y) if x == y => sugg::ZERO.into(),
            (_, _) => (&self.0 - &rhs.0).into(),
        }
    }
}

impl std::ops::Add<&MinifyingSugg<'static>> for MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn add(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            ("0", _) => rhs.clone(),
            (_, "0") => self,
            (_, _) => (self.0 + &rhs.0).into(),
        }
    }
}

impl std::ops::Sub<&MinifyingSugg<'static>> for MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn sub(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            (_, "0") => self,
            ("0", _) => (-rhs.0.clone()).into(),
            (x, y) if x == y => sugg::ZERO.into(),
            (_, _) => (self.0 - &rhs.0).into(),
        }
    }
}

/// a wrapper around `MinifyingSugg`, which carries an operator like currying
/// so that the suggested code become more efficient (e.g. `foo + -bar` `foo - bar`).
struct Offset {
    value: MinifyingSugg<'static>,
    sign: OffsetSign,
}

#[derive(Clone, Copy)]
enum OffsetSign {
    Positive,
    Negative,
}

impl Offset {
    fn negative(value: Sugg<'static>) -> Self {
        Self {
            value: value.into(),
            sign: OffsetSign::Negative,
        }
    }

    fn positive(value: Sugg<'static>) -> Self {
        Self {
            value: value.into(),
            sign: OffsetSign::Positive,
        }
    }

    fn empty() -> Self {
        Self::positive(sugg::ZERO)
    }
}

fn apply_offset(lhs: &MinifyingSugg<'static>, rhs: &Offset) -> MinifyingSugg<'static> {
    match rhs.sign {
        OffsetSign::Positive => lhs + &rhs.value,
        OffsetSign::Negative => lhs - &rhs.value,
    }
}

#[derive(Debug, Clone, Copy)]
enum StartKind<'hir> {
    Range,
    Counter { initializer: &'hir Expr<'hir> },
}

struct IndexExpr<'hir> {
    base: &'hir Expr<'hir>,
    idx: StartKind<'hir>,
    idx_offset: Offset,
}

struct Start<'hir> {
    id: HirId,
    kind: StartKind<'hir>,
}

fn is_slice_like<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'_>) -> bool {
    let is_slice = match ty.kind() {
        ty::Ref(_, subty, _) => is_slice_like(cx, subty),
        ty::Slice(..) | ty::Array(..) => true,
        _ => false,
    };

    is_slice || is_type_diagnostic_item(cx, ty, sym::Vec) || is_type_diagnostic_item(cx, ty, sym::VecDeque)
}

fn fetch_cloned_expr<'tcx>(expr: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    if_chain! {
        if let ExprKind::MethodCall(method, _, args, _) = expr.kind;
        if method.ident.name == sym::clone;
        if args.len() == 1;
        if let Some(arg) = args.get(0);
        then { arg } else { expr }
    }
}

fn get_details_from_idx<'tcx>(
    cx: &LateContext<'tcx>,
    idx: &Expr<'_>,
    starts: &[Start<'tcx>],
) -> Option<(StartKind<'tcx>, Offset)> {
    fn get_start<'tcx>(e: &Expr<'_>, starts: &[Start<'tcx>]) -> Option<StartKind<'tcx>> {
        let id = path_to_local(e)?;
        starts.iter().find(|start| start.id == id).map(|start| start.kind)
    }

    fn get_offset<'tcx>(cx: &LateContext<'tcx>, e: &Expr<'_>, starts: &[Start<'tcx>]) -> Option<Sugg<'static>> {
        match &e.kind {
            ExprKind::Lit(l) => match l.node {
                ast::LitKind::Int(x, _ty) => Some(Sugg::NonParen(x.to_string().into())),
                _ => None,
            },
            ExprKind::Path(..) if get_start(e, starts).is_none() => Some(Sugg::hir(cx, e, "???")),
            _ => None,
        }
    }

    match idx.kind {
        ExprKind::Binary(op, lhs, rhs) => match op.node {
            BinOpKind::Add => {
                let offset_opt = get_start(lhs, starts)
                    .and_then(|s| get_offset(cx, rhs, starts).map(|o| (s, o)))
                    .or_else(|| get_start(rhs, starts).and_then(|s| get_offset(cx, lhs, starts).map(|o| (s, o))));

                offset_opt.map(|(s, o)| (s, Offset::positive(o)))
            },
            BinOpKind::Sub => {
                get_start(lhs, starts).and_then(|s| get_offset(cx, rhs, starts).map(|o| (s, Offset::negative(o))))
            },
            _ => None,
        },
        ExprKind::Path(..) => get_start(idx, starts).map(|s| (s, Offset::empty())),
        _ => None,
    }
}

fn get_assignment<'tcx>(e: &'tcx Expr<'tcx>) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if let ExprKind::Assign(lhs, rhs, _) = e.kind {
        Some((lhs, rhs))
    } else {
        None
    }
}

/// Get assignments from the given block.
/// The returned iterator yields `None` if no assignment expressions are there,
/// filtering out the increments of the given whitelisted loop counters;
/// because its job is to make sure there's nothing other than assignments and the increments.
fn get_assignments<'a, 'tcx>(
    Block { stmts, expr, .. }: &'tcx Block<'tcx>,
    loop_counters: &'a [Start<'tcx>],
) -> impl Iterator<Item = Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)>> + 'a {
    // As the `filter` and `map` below do different things, I think putting together
    // just increases complexity. (cc #3188 and #4193)
    stmts
        .iter()
        .filter_map(move |stmt| match stmt.kind {
            StmtKind::Local(..) | StmtKind::Item(..) => None,
            StmtKind::Expr(e) | StmtKind::Semi(e) => Some(e),
        })
        .chain((*expr).into_iter())
        .filter(move |e| {
            if let ExprKind::AssignOp(_, place, _) = e.kind {
                path_to_local(place).map_or(false, |id| {
                    !loop_counters
                        .iter()
                        // skip the first item which should be `StartKind::Range`
                        // this makes it possible to use the slice with `StartKind::Range` in the same iterator loop.
                        .skip(1)
                        .any(|counter| counter.id == id)
                })
            } else {
                true
            }
        })
        .map(get_assignment)
}

fn get_loop_counters<'a, 'tcx>(
    cx: &'a LateContext<'tcx>,
    body: &'tcx Block<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<impl Iterator<Item = Start<'tcx>> + 'a> {
    // Look for variables that are incremented once per loop iteration.
    let mut increment_visitor = IncrementVisitor::new(cx);
    walk_block(&mut increment_visitor, body);

    // For each candidate, check the parent block to see if
    // it's initialized to zero at the start of the loop.
    get_enclosing_block(cx, expr.hir_id).and_then(|block| {
        increment_visitor
            .into_results()
            .filter_map(move |var_id| {
                let mut initialize_visitor = InitializeVisitor::new(cx, expr, var_id);
                walk_block(&mut initialize_visitor, block);

                initialize_visitor.get_result().map(|(_, initializer)| Start {
                    id: var_id,
                    kind: StartKind::Counter { initializer },
                })
            })
            .into()
    })
}
