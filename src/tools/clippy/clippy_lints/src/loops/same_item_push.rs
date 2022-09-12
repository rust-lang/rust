use super::SAME_ITEM_PUSH;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::path_to_local;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{BindingAnnotation, Block, Expr, ExprKind, HirId, Mutability, Node, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use std::iter::Iterator;

/// Detects for loop pushing the same item into a Vec
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    _: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    _: &'tcx Expr<'_>,
) {
    fn emit_lint(cx: &LateContext<'_>, vec: &Expr<'_>, pushed_item: &Expr<'_>) {
        let vec_str = snippet_with_macro_callsite(cx, vec.span, "");
        let item_str = snippet_with_macro_callsite(cx, pushed_item.span, "");

        span_lint_and_help(
            cx,
            SAME_ITEM_PUSH,
            vec.span,
            "it looks like the same item is being pushed into this Vec",
            None,
            &format!(
                "try using vec![{};SIZE] or {}.resize(NEW_SIZE, {})",
                item_str, vec_str, item_str
            ),
        );
    }

    if !matches!(pat.kind, PatKind::Wild) {
        return;
    }

    // Determine whether it is safe to lint the body
    let mut same_item_push_visitor = SameItemPushVisitor::new(cx);
    walk_expr(&mut same_item_push_visitor, body);
    if_chain! {
        if same_item_push_visitor.should_lint();
        if let Some((vec, pushed_item)) = same_item_push_visitor.vec_push;
        let vec_ty = cx.typeck_results().expr_ty(vec);
        let ty = vec_ty.walk().nth(1).unwrap().expect_ty();
        if cx
            .tcx
            .lang_items()
            .clone_trait()
            .map_or(false, |id| implements_trait(cx, ty, id, &[]));
        then {
            // Make sure that the push does not involve possibly mutating values
            match pushed_item.kind {
                ExprKind::Path(ref qpath) => {
                    match cx.qpath_res(qpath, pushed_item.hir_id) {
                        // immutable bindings that are initialized with literal or constant
                        Res::Local(hir_id) => {
                            let node = cx.tcx.hir().get(hir_id);
                            if_chain! {
                                if let Node::Pat(pat) = node;
                                if let PatKind::Binding(bind_ann, ..) = pat.kind;
                                if !matches!(bind_ann, BindingAnnotation(_, Mutability::Mut));
                                let parent_node = cx.tcx.hir().get_parent_node(hir_id);
                                if let Some(Node::Local(parent_let_expr)) = cx.tcx.hir().find(parent_node);
                                if let Some(init) = parent_let_expr.init;
                                then {
                                    match init.kind {
                                        // immutable bindings that are initialized with literal
                                        ExprKind::Lit(..) => emit_lint(cx, vec, pushed_item),
                                        // immutable bindings that are initialized with constant
                                        ExprKind::Path(ref path) => {
                                            if let Res::Def(DefKind::Const, ..) = cx.qpath_res(path, init.hir_id) {
                                                emit_lint(cx, vec, pushed_item);
                                            }
                                        }
                                        _ => {},
                                    }
                                }
                            }
                        },
                        // constant
                        Res::Def(DefKind::Const, ..) => emit_lint(cx, vec, pushed_item),
                        _ => {},
                    }
                },
                ExprKind::Lit(..) => emit_lint(cx, vec, pushed_item),
                _ => {},
            }
        }
    }
}

// Scans the body of the for loop and determines whether lint should be given
struct SameItemPushVisitor<'a, 'tcx> {
    non_deterministic_expr: bool,
    multiple_pushes: bool,
    // this field holds the last vec push operation visited, which should be the only push seen
    vec_push: Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)>,
    cx: &'a LateContext<'tcx>,
    used_locals: FxHashSet<HirId>,
}

impl<'a, 'tcx> SameItemPushVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            non_deterministic_expr: false,
            multiple_pushes: false,
            vec_push: None,
            cx,
            used_locals: FxHashSet::default(),
        }
    }

    fn should_lint(&self) -> bool {
        if_chain! {
            if !self.non_deterministic_expr;
            if !self.multiple_pushes;
            if let Some((vec, _)) = self.vec_push;
            if let Some(hir_id) = path_to_local(vec);
            then {
                !self.used_locals.contains(&hir_id)
            } else {
                false
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for SameItemPushVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match &expr.kind {
            // Non-determinism may occur ... don't give a lint
            ExprKind::Loop(..) | ExprKind::Match(..) | ExprKind::If(..) => self.non_deterministic_expr = true,
            ExprKind::Block(block, _) => self.visit_block(block),
            _ => {
                if let Some(hir_id) = path_to_local(expr) {
                    self.used_locals.insert(hir_id);
                }
                walk_expr(self, expr);
            },
        }
    }

    fn visit_block(&mut self, b: &'tcx Block<'_>) {
        for stmt in b.stmts.iter() {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, s: &'tcx Stmt<'_>) {
        let vec_push_option = get_vec_push(self.cx, s);
        if vec_push_option.is_none() {
            // Current statement is not a push so visit inside
            match &s.kind {
                StmtKind::Expr(expr) | StmtKind::Semi(expr) => self.visit_expr(expr),
                _ => {},
            }
        } else {
            // Current statement is a push ...check whether another
            // push had been previously done
            if self.vec_push.is_none() {
                self.vec_push = vec_push_option;
            } else {
                // There are multiple pushes ... don't lint
                self.multiple_pushes = true;
            }
        }
    }
}

// Given some statement, determine if that statement is a push on a Vec. If it is, return
// the Vec being pushed into and the item being pushed
fn get_vec_push<'tcx>(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if_chain! {
            // Extract method being called
            if let StmtKind::Semi(semi_stmt) = &stmt.kind;
            if let ExprKind::MethodCall(path, self_expr, args, _) = &semi_stmt.kind;
            // Figure out the parameters for the method call
            if let Some(pushed_item) = args.get(0);
            // Check that the method being called is push() on a Vec
            if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(self_expr), sym::Vec);
            if path.ident.name.as_str() == "push";
            then {
                return Some((self_expr, pushed_item))
            }
    }
    None
}
