use super::NEEDLESS_RANGE_LOOP;
use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::ty::has_iter_method;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{contains_name, higher, is_integer_const, sugg, SpanlessEq};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{BinOpKind, BorrowKind, Closure, Expr, ExprKind, HirId, Mutability, Pat, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::middle::region;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::{sym, Symbol};
use std::iter::{self, Iterator};
use std::mem;

/// Checks for looping over a range and then indexing a sequence with it.
/// The iteratee must be a range literal.
#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
) {
    if let Some(higher::Range {
        start: Some(start),
        ref end,
        limits,
    }) = higher::Range::hir(arg)
    {
        // the var must be a single name
        if let PatKind::Binding(_, canonical_id, ident, _) = pat.kind {
            let mut visitor = VarVisitor {
                cx,
                var: canonical_id,
                indexed_mut: FxHashSet::default(),
                indexed_indirectly: FxHashMap::default(),
                indexed_directly: FxHashMap::default(),
                referenced: FxHashSet::default(),
                nonindex: false,
                prefer_mutable: false,
            };
            walk_expr(&mut visitor, body);

            // linting condition: we only indexed one variable, and indexed it directly
            if visitor.indexed_indirectly.is_empty() && visitor.indexed_directly.len() == 1 {
                let (indexed, (indexed_extent, indexed_ty)) = visitor
                    .indexed_directly
                    .into_iter()
                    .next()
                    .expect("already checked that we have exactly 1 element");

                // ensure that the indexed variable was declared before the loop, see #601
                if let Some(indexed_extent) = indexed_extent {
                    let parent_def_id = cx.tcx.hir().get_parent_item(expr.hir_id);
                    let region_scope_tree = cx.tcx.region_scope_tree(parent_def_id);
                    let pat_extent = region_scope_tree.var_scope(pat.hir_id.local_id).unwrap();
                    if region_scope_tree.is_subscope_of(indexed_extent, pat_extent) {
                        return;
                    }
                }

                // don't lint if the container that is indexed does not have .iter() method
                let has_iter = has_iter_method(cx, indexed_ty);
                if has_iter.is_none() {
                    return;
                }

                // don't lint if the container that is indexed into is also used without
                // indexing
                if visitor.referenced.contains(&indexed) {
                    return;
                }

                let starts_at_zero = is_integer_const(cx, start, 0);

                let skip = if starts_at_zero {
                    String::new()
                } else if visitor.indexed_mut.contains(&indexed) && contains_name(indexed, start, cx) {
                    return;
                } else {
                    format!(".skip({})", snippet(cx, start.span, ".."))
                };

                let mut end_is_start_plus_val = false;

                let take = if let Some(end) = *end {
                    let mut take_expr = end;

                    if let ExprKind::Binary(ref op, left, right) = end.kind {
                        if op.node == BinOpKind::Add {
                            let start_equal_left = SpanlessEq::new(cx).eq_expr(start, left);
                            let start_equal_right = SpanlessEq::new(cx).eq_expr(start, right);

                            if start_equal_left {
                                take_expr = right;
                            } else if start_equal_right {
                                take_expr = left;
                            }

                            end_is_start_plus_val = start_equal_left | start_equal_right;
                        }
                    }

                    if is_len_call(end, indexed) || is_end_eq_array_len(cx, end, limits, indexed_ty) {
                        String::new()
                    } else if visitor.indexed_mut.contains(&indexed) && contains_name(indexed, take_expr, cx) {
                        return;
                    } else {
                        match limits {
                            ast::RangeLimits::Closed => {
                                let take_expr = sugg::Sugg::hir(cx, take_expr, "<count>");
                                format!(".take({})", take_expr + sugg::ONE)
                            },
                            ast::RangeLimits::HalfOpen => {
                                format!(".take({})", snippet(cx, take_expr.span, ".."))
                            },
                        }
                    }
                } else {
                    String::new()
                };

                let (ref_mut, method) = if visitor.indexed_mut.contains(&indexed) {
                    ("mut ", "iter_mut")
                } else {
                    ("", "iter")
                };

                let take_is_empty = take.is_empty();
                let mut method_1 = take;
                let mut method_2 = skip;

                if end_is_start_plus_val {
                    mem::swap(&mut method_1, &mut method_2);
                }

                if visitor.nonindex {
                    span_lint_and_then(
                        cx,
                        NEEDLESS_RANGE_LOOP,
                        arg.span,
                        &format!("the loop variable `{}` is used to index `{indexed}`", ident.name),
                        |diag| {
                            multispan_sugg(
                                diag,
                                "consider using an iterator",
                                vec![
                                    (pat.span, format!("({}, <item>)", ident.name)),
                                    (
                                        arg.span,
                                        format!("{indexed}.{method}().enumerate(){method_1}{method_2}"),
                                    ),
                                ],
                            );
                        },
                    );
                } else {
                    let repl = if starts_at_zero && take_is_empty {
                        format!("&{ref_mut}{indexed}")
                    } else {
                        format!("{indexed}.{method}(){method_1}{method_2}")
                    };

                    span_lint_and_then(
                        cx,
                        NEEDLESS_RANGE_LOOP,
                        arg.span,
                        &format!("the loop variable `{}` is only used to index `{indexed}`", ident.name),
                        |diag| {
                            multispan_sugg(
                                diag,
                                "consider using an iterator",
                                vec![(pat.span, "<item>".to_string()), (arg.span, repl)],
                            );
                        },
                    );
                }
            }
        }
    }
}

fn is_len_call(expr: &Expr<'_>, var: Symbol) -> bool {
    if_chain! {
        if let ExprKind::MethodCall(method, recv, [], _) = expr.kind;
        if method.ident.name == sym::len;
        if let ExprKind::Path(QPath::Resolved(_, path)) = recv.kind;
        if path.segments.len() == 1;
        if path.segments[0].ident.name == var;
        then {
            return true;
        }
    }

    false
}

fn is_end_eq_array_len<'tcx>(
    cx: &LateContext<'tcx>,
    end: &Expr<'_>,
    limits: ast::RangeLimits,
    indexed_ty: Ty<'tcx>,
) -> bool {
    if_chain! {
        if let ExprKind::Lit(ref lit) = end.kind;
        if let ast::LitKind::Int(end_int, _) = lit.node;
        if let ty::Array(_, arr_len_const) = indexed_ty.kind();
        if let Some(arr_len) = arr_len_const.try_eval_usize(cx.tcx, cx.param_env);
        then {
            return match limits {
                ast::RangeLimits::Closed => end_int + 1 >= arr_len.into(),
                ast::RangeLimits::HalfOpen => end_int >= arr_len.into(),
            };
        }
    }

    false
}

struct VarVisitor<'a, 'tcx> {
    /// context reference
    cx: &'a LateContext<'tcx>,
    /// var name to look for as index
    var: HirId,
    /// indexed variables that are used mutably
    indexed_mut: FxHashSet<Symbol>,
    /// indirectly indexed variables (`v[(i + 4) % N]`), the extend is `None` for global
    indexed_indirectly: FxHashMap<Symbol, Option<region::Scope>>,
    /// subset of `indexed` of vars that are indexed directly: `v[i]`
    /// this will not contain cases like `v[calc_index(i)]` or `v[(i + 4) % N]`
    indexed_directly: FxHashMap<Symbol, (Option<region::Scope>, Ty<'tcx>)>,
    /// Any names that are used outside an index operation.
    /// Used to detect things like `&mut vec` used together with `vec[i]`
    referenced: FxHashSet<Symbol>,
    /// has the loop variable been used in expressions other than the index of
    /// an index op?
    nonindex: bool,
    /// Whether we are inside the `$` in `&mut $` or `$ = foo` or `$.bar`, where bar
    /// takes `&mut self`
    prefer_mutable: bool,
}

impl<'a, 'tcx> VarVisitor<'a, 'tcx> {
    fn check(&mut self, idx: &'tcx Expr<'_>, seqexpr: &'tcx Expr<'_>, expr: &'tcx Expr<'_>) -> bool {
        if_chain! {
            // the indexed container is referenced by a name
            if let ExprKind::Path(ref seqpath) = seqexpr.kind;
            if let QPath::Resolved(None, seqvar) = *seqpath;
            if seqvar.segments.len() == 1;
            if is_local_used(self.cx, idx, self.var);
            then {
                if self.prefer_mutable {
                    self.indexed_mut.insert(seqvar.segments[0].ident.name);
                }
                let index_used_directly = matches!(idx.kind, ExprKind::Path(_));
                let res = self.cx.qpath_res(seqpath, seqexpr.hir_id);
                match res {
                    Res::Local(hir_id) => {
                        let parent_def_id = self.cx.tcx.hir().get_parent_item(expr.hir_id);
                        let extent = self
                            .cx
                            .tcx
                            .region_scope_tree(parent_def_id)
                            .var_scope(hir_id.local_id)
                            .unwrap();
                        if index_used_directly {
                            self.indexed_directly.insert(
                                seqvar.segments[0].ident.name,
                                (Some(extent), self.cx.typeck_results().node_type(seqexpr.hir_id)),
                            );
                        } else {
                            self.indexed_indirectly
                                .insert(seqvar.segments[0].ident.name, Some(extent));
                        }
                        return false; // no need to walk further *on the variable*
                    },
                    Res::Def(DefKind::Static(_) | DefKind::Const, ..) => {
                        if index_used_directly {
                            self.indexed_directly.insert(
                                seqvar.segments[0].ident.name,
                                (None, self.cx.typeck_results().node_type(seqexpr.hir_id)),
                            );
                        } else {
                            self.indexed_indirectly.insert(seqvar.segments[0].ident.name, None);
                        }
                        return false; // no need to walk further *on the variable*
                    },
                    _ => (),
                }
            }
        }
        true
    }
}

impl<'a, 'tcx> Visitor<'tcx> for VarVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if_chain! {
            // a range index op
            if let ExprKind::MethodCall(meth, args_0, [args_1, ..], _) = &expr.kind;
            if let Some(trait_id) = self
                .cx
                .typeck_results()
                .type_dependent_def_id(expr.hir_id)
                .and_then(|def_id| self.cx.tcx.trait_of_item(def_id));
            if (meth.ident.name == sym::index && self.cx.tcx.lang_items().index_trait() == Some(trait_id))
                || (meth.ident.name == sym::index_mut && self.cx.tcx.lang_items().index_mut_trait() == Some(trait_id));
            if !self.check(args_1, args_0, expr);
            then {
                return;
            }
        }

        if_chain! {
            // an index op
            if let ExprKind::Index(seqexpr, idx) = expr.kind;
            if !self.check(idx, seqexpr, expr);
            then {
                return;
            }
        }

        if_chain! {
            // directly using a variable
            if let ExprKind::Path(QPath::Resolved(None, path)) = expr.kind;
            if let Res::Local(local_id) = path.res;
            then {
                if local_id == self.var {
                    self.nonindex = true;
                } else {
                    // not the correct variable, but still a variable
                    self.referenced.insert(path.segments[0].ident.name);
                }
            }
        }

        let old = self.prefer_mutable;
        match expr.kind {
            ExprKind::AssignOp(_, lhs, rhs) | ExprKind::Assign(lhs, rhs, _) => {
                self.prefer_mutable = true;
                self.visit_expr(lhs);
                self.prefer_mutable = false;
                self.visit_expr(rhs);
            },
            ExprKind::AddrOf(BorrowKind::Ref, mutbl, expr) => {
                if mutbl == Mutability::Mut {
                    self.prefer_mutable = true;
                }
                self.visit_expr(expr);
            },
            ExprKind::Call(f, args) => {
                self.visit_expr(f);
                for expr in args {
                    let ty = self.cx.typeck_results().expr_ty_adjusted(expr);
                    self.prefer_mutable = false;
                    if let ty::Ref(_, _, mutbl) = *ty.kind() {
                        if mutbl == Mutability::Mut {
                            self.prefer_mutable = true;
                        }
                    }
                    self.visit_expr(expr);
                }
            },
            ExprKind::MethodCall(_, receiver, args, _) => {
                let def_id = self.cx.typeck_results().type_dependent_def_id(expr.hir_id).unwrap();
                for (ty, expr) in iter::zip(
                    self.cx.tcx.fn_sig(def_id).inputs().skip_binder(),
                    std::iter::once(receiver).chain(args.iter()),
                ) {
                    self.prefer_mutable = false;
                    if let ty::Ref(_, _, mutbl) = *ty.kind() {
                        if mutbl == Mutability::Mut {
                            self.prefer_mutable = true;
                        }
                    }
                    self.visit_expr(expr);
                }
            },
            ExprKind::Closure(&Closure { body, .. }) => {
                let body = self.cx.tcx.hir().body(body);
                self.visit_expr(body.value);
            },
            _ => walk_expr(self, expr),
        }
        self.prefer_mutable = old;
    }
}
