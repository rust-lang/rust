use super::WHILE_LET_ON_ITERATOR;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{
    get_enclosing_loop_or_closure, is_refutable, is_trait_method, match_def_path, paths, visitors::is_res_used,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, ErasedMap, NestedVisitorMap, Visitor};
use rustc_hir::{def::Res, Expr, ExprKind, HirId, Local, Mutability, PatKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_span::{symbol::sym, Span, Symbol};

pub(super) fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    let (scrutinee_expr, iter_expr, some_pat, loop_expr) = if_chain! {
        if let Some(higher::WhileLet { if_then, let_pat, let_expr }) = higher::WhileLet::hir(expr);
        // check for `Some(..)` pattern
        if let PatKind::TupleStruct(QPath::Resolved(None, pat_path), some_pat, _) = let_pat.kind;
        if let Res::Def(_, pat_did) = pat_path.res;
        if match_def_path(cx, pat_did, &paths::OPTION_SOME);
        // check for call to `Iterator::next`
        if let ExprKind::MethodCall(method_name, _, [iter_expr], _) = let_expr.kind;
        if method_name.ident.name == sym::next;
        if is_trait_method(cx, let_expr, sym::Iterator);
        if let Some(iter_expr_struct) = try_parse_iter_expr(cx, iter_expr);
        // get the loop containing the match expression
        if !uses_iter(cx, &iter_expr_struct, if_then);
        then {
            (let_expr, iter_expr_struct, some_pat, expr)
        } else {
            return;
        }
    };

    let mut applicability = Applicability::MachineApplicable;
    let loop_var = if let Some(some_pat) = some_pat.first() {
        if is_refutable(cx, some_pat) {
            // Refutable patterns don't work with for loops.
            return;
        }
        snippet_with_applicability(cx, some_pat.span, "..", &mut applicability)
    } else {
        "_".into()
    };

    // If the iterator is a field or the iterator is accessed after the loop is complete it needs to be
    // borrowed mutably. TODO: If the struct can be partially moved from and the struct isn't used
    // afterwards a mutable borrow of a field isn't necessary.
    let ref_mut = if !iter_expr.fields.is_empty() || needs_mutable_borrow(cx, &iter_expr, loop_expr) {
        if cx.typeck_results().node_type(iter_expr.hir_id).ref_mutability() == Some(Mutability::Mut) {
            // Reborrow for mutable references. It may not be possible to get a mutable reference here.
            "&mut *"
        } else {
            "&mut "
        }
    } else {
        ""
    };

    let iterator = snippet_with_applicability(cx, iter_expr.span, "_", &mut applicability);
    span_lint_and_sugg(
        cx,
        WHILE_LET_ON_ITERATOR,
        expr.span.with_hi(scrutinee_expr.span.hi()),
        "this loop could be written as a `for` loop",
        "try",
        format!("for {} in {}{}", loop_var, ref_mut, iterator),
        applicability,
    );
}

#[derive(Debug)]
struct IterExpr {
    /// The span of the whole expression, not just the path and fields stored here.
    span: Span,
    /// The HIR id of the whole expression, not just the path and fields stored here.
    hir_id: HirId,
    /// The fields used, in order of child to parent.
    fields: Vec<Symbol>,
    /// The path being used.
    path: Res,
}

/// Parses any expression to find out which field of which variable is used. Will return `None` if
/// the expression might have side effects.
fn try_parse_iter_expr(cx: &LateContext<'_>, mut e: &Expr<'_>) -> Option<IterExpr> {
    let span = e.span;
    let hir_id = e.hir_id;
    let mut fields = Vec::new();
    loop {
        match e.kind {
            ExprKind::Path(ref path) => {
                break Some(IterExpr {
                    span,
                    hir_id,
                    fields,
                    path: cx.qpath_res(path, e.hir_id),
                });
            },
            ExprKind::Field(base, name) => {
                fields.push(name.name);
                e = base;
            },
            // Dereferencing a pointer has no side effects and doesn't affect which field is being used.
            ExprKind::Unary(UnOp::Deref, base) if cx.typeck_results().expr_ty(base).is_ref() => e = base,

            // Shouldn't have side effects, but there's no way to trace which field is used. So forget which fields have
            // already been seen.
            ExprKind::Index(base, idx) if !idx.can_have_side_effects() => {
                fields.clear();
                e = base;
            },
            ExprKind::Unary(UnOp::Deref, base) => {
                fields.clear();
                e = base;
            },

            // No effect and doesn't affect which field is being used.
            ExprKind::DropTemps(base) | ExprKind::AddrOf(_, _, base) | ExprKind::Type(base, _) => e = base,
            _ => break None,
        }
    }
}

fn is_expr_same_field(cx: &LateContext<'_>, mut e: &Expr<'_>, mut fields: &[Symbol], path_res: Res) -> bool {
    loop {
        match (&e.kind, fields) {
            (&ExprKind::Field(base, name), [head_field, tail_fields @ ..]) if name.name == *head_field => {
                e = base;
                fields = tail_fields;
            },
            (ExprKind::Path(path), []) => {
                break cx.qpath_res(path, e.hir_id) == path_res;
            },
            (&(ExprKind::DropTemps(base) | ExprKind::AddrOf(_, _, base) | ExprKind::Type(base, _)), _) => e = base,
            _ => break false,
        }
    }
}

/// Checks if the given expression is the same field as, is a child of, or is the parent of the
/// given field. Used to check if the expression can be used while the given field is borrowed
/// mutably. e.g. if checking for `x.y`, then `x.y`, `x.y.z`, and `x` will all return true, but
/// `x.z`, and `y` will return false.
fn is_expr_same_child_or_parent_field(cx: &LateContext<'_>, expr: &Expr<'_>, fields: &[Symbol], path_res: Res) -> bool {
    match expr.kind {
        ExprKind::Field(base, name) => {
            if let Some((head_field, tail_fields)) = fields.split_first() {
                if name.name == *head_field && is_expr_same_field(cx, base, tail_fields, path_res) {
                    return true;
                }
                // Check if the expression is a parent field
                let mut fields_iter = tail_fields.iter();
                while let Some(field) = fields_iter.next() {
                    if *field == name.name && is_expr_same_field(cx, base, fields_iter.as_slice(), path_res) {
                        return true;
                    }
                }
            }

            // Check if the expression is a child field.
            let mut e = base;
            loop {
                match e.kind {
                    ExprKind::Field(..) if is_expr_same_field(cx, e, fields, path_res) => break true,
                    ExprKind::Field(base, _) | ExprKind::DropTemps(base) | ExprKind::Type(base, _) => e = base,
                    ExprKind::Path(ref path) if fields.is_empty() => {
                        break cx.qpath_res(path, e.hir_id) == path_res;
                    },
                    _ => break false,
                }
            }
        },
        // If the path matches, this is either an exact match, or the expression is a parent of the field.
        ExprKind::Path(ref path) => cx.qpath_res(path, expr.hir_id) == path_res,
        ExprKind::DropTemps(base) | ExprKind::Type(base, _) | ExprKind::AddrOf(_, _, base) => {
            is_expr_same_child_or_parent_field(cx, base, fields, path_res)
        },
        _ => false,
    }
}

/// Strips off all field and path expressions. This will return true if a field or path has been
/// skipped. Used to skip them after failing to check for equality.
fn skip_fields_and_path(expr: &'tcx Expr<'_>) -> (Option<&'tcx Expr<'tcx>>, bool) {
    let mut e = expr;
    let e = loop {
        match e.kind {
            ExprKind::Field(base, _) | ExprKind::DropTemps(base) | ExprKind::Type(base, _) => e = base,
            ExprKind::Path(_) => return (None, true),
            _ => break e,
        }
    };
    (Some(e), e.hir_id != expr.hir_id)
}

/// Checks if the given expression uses the iterator.
fn uses_iter(cx: &LateContext<'tcx>, iter_expr: &IterExpr, container: &'tcx Expr<'_>) -> bool {
    struct V<'a, 'b, 'tcx> {
        cx: &'a LateContext<'tcx>,
        iter_expr: &'b IterExpr,
        uses_iter: bool,
    }
    impl Visitor<'tcx> for V<'_, '_, 'tcx> {
        type Map = ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.uses_iter {
                // return
            } else if is_expr_same_child_or_parent_field(self.cx, e, &self.iter_expr.fields, self.iter_expr.path) {
                self.uses_iter = true;
            } else if let (e, true) = skip_fields_and_path(e) {
                if let Some(e) = e {
                    self.visit_expr(e);
                }
            } else if let ExprKind::Closure(_, _, id, _, _) = e.kind {
                if is_res_used(self.cx, self.iter_expr.path, id) {
                    self.uses_iter = true;
                }
            } else {
                walk_expr(self, e);
            }
        }
    }

    let mut v = V {
        cx,
        iter_expr,
        uses_iter: false,
    };
    v.visit_expr(container);
    v.uses_iter
}

#[allow(clippy::too_many_lines)]
fn needs_mutable_borrow(cx: &LateContext<'tcx>, iter_expr: &IterExpr, loop_expr: &'tcx Expr<'_>) -> bool {
    struct AfterLoopVisitor<'a, 'b, 'tcx> {
        cx: &'a LateContext<'tcx>,
        iter_expr: &'b IterExpr,
        loop_id: HirId,
        after_loop: bool,
        used_iter: bool,
    }
    impl Visitor<'tcx> for AfterLoopVisitor<'_, '_, 'tcx> {
        type Map = ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.used_iter {
                return;
            }
            if self.after_loop {
                if is_expr_same_child_or_parent_field(self.cx, e, &self.iter_expr.fields, self.iter_expr.path) {
                    self.used_iter = true;
                } else if let (e, true) = skip_fields_and_path(e) {
                    if let Some(e) = e {
                        self.visit_expr(e);
                    }
                } else if let ExprKind::Closure(_, _, id, _, _) = e.kind {
                    self.used_iter = is_res_used(self.cx, self.iter_expr.path, id);
                } else {
                    walk_expr(self, e);
                }
            } else if self.loop_id == e.hir_id {
                self.after_loop = true;
            } else {
                walk_expr(self, e);
            }
        }
    }

    struct NestedLoopVisitor<'a, 'b, 'tcx> {
        cx: &'a LateContext<'tcx>,
        iter_expr: &'b IterExpr,
        local_id: HirId,
        loop_id: HirId,
        after_loop: bool,
        found_local: bool,
        used_after: bool,
    }
    impl Visitor<'tcx> for NestedLoopVisitor<'a, 'b, 'tcx> {
        type Map = ErasedMap<'tcx>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_local(&mut self, l: &'tcx Local<'_>) {
            if !self.after_loop {
                l.pat.each_binding_or_first(&mut |_, id, _, _| {
                    if id == self.local_id {
                        self.found_local = true;
                    }
                });
            }
            if let Some(e) = l.init {
                self.visit_expr(e);
            }
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.used_after {
                return;
            }
            if self.after_loop {
                if is_expr_same_child_or_parent_field(self.cx, e, &self.iter_expr.fields, self.iter_expr.path) {
                    self.used_after = true;
                } else if let (e, true) = skip_fields_and_path(e) {
                    if let Some(e) = e {
                        self.visit_expr(e);
                    }
                } else if let ExprKind::Closure(_, _, id, _, _) = e.kind {
                    self.used_after = is_res_used(self.cx, self.iter_expr.path, id);
                } else {
                    walk_expr(self, e);
                }
            } else if e.hir_id == self.loop_id {
                self.after_loop = true;
            } else {
                walk_expr(self, e);
            }
        }
    }

    if let Some(e) = get_enclosing_loop_or_closure(cx.tcx, loop_expr) {
        // The iterator expression will be used on the next iteration (for loops), or on the next call (for
        // closures) unless it is declared within the enclosing expression. TODO: Check for closures
        // used where an `FnOnce` type is expected.
        let local_id = match iter_expr.path {
            Res::Local(id) => id,
            _ => return true,
        };
        let mut v = NestedLoopVisitor {
            cx,
            iter_expr,
            local_id,
            loop_id: loop_expr.hir_id,
            after_loop: false,
            found_local: false,
            used_after: false,
        };
        v.visit_expr(e);
        v.used_after || !v.found_local
    } else {
        let mut v = AfterLoopVisitor {
            cx,
            iter_expr,
            loop_id: loop_expr.hir_id,
            after_loop: false,
            used_iter: false,
        };
        v.visit_expr(&cx.tcx.hir().body(cx.enclosing_body.unwrap()).value);
        v.used_iter
    }
}
