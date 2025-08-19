use super::SAME_ITEM_PUSH;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::Msrv;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::{msrvs, path_to_local, std_or_core, sym};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{BindingMode, Block, Expr, ExprKind, HirId, Mutability, Node, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_span::SyntaxContext;

/// Detects for loop pushing the same item into a Vec
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    _: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    _: &'tcx Expr<'_>,
    msrv: Msrv,
) {
    fn emit_lint(cx: &LateContext<'_>, vec: &Expr<'_>, pushed_item: &Expr<'_>, ctxt: SyntaxContext, msrv: Msrv) {
        let mut app = Applicability::Unspecified;
        let vec_str = snippet_with_context(cx, vec.span, ctxt, "", &mut app).0;
        let item_str = snippet_with_context(cx, pushed_item.span, ctxt, "", &mut app).0;

        let secondary_help = if msrv.meets(cx, msrvs::REPEAT_N)
            && let Some(std_or_core) = std_or_core(cx)
        {
            format!("or `{vec_str}.extend({std_or_core}::iter::repeat_n({item_str}, SIZE))`")
        } else {
            format!("or `{vec_str}.resize(NEW_SIZE, {item_str})`")
        };

        span_lint_and_then(
            cx,
            SAME_ITEM_PUSH,
            vec.span,
            "it looks like the same item is being pushed into this `Vec`",
            |diag| {
                diag.help(format!("consider using `vec![{item_str};SIZE]`"))
                    .help(secondary_help);
            },
        );
    }

    if !matches!(pat.kind, PatKind::Wild) {
        return;
    }

    // Determine whether it is safe to lint the body
    let mut same_item_push_visitor = SameItemPushVisitor::new(cx);
    walk_expr(&mut same_item_push_visitor, body);
    if same_item_push_visitor.should_lint()
        && let Some((vec, pushed_item, ctxt)) = same_item_push_visitor.vec_push
        && let vec_ty = cx.typeck_results().expr_ty(vec)
        && let ty = vec_ty.walk().nth(1).unwrap().expect_ty()
        && cx
            .tcx
            .lang_items()
            .clone_trait()
            .is_some_and(|id| implements_trait(cx, ty, id, &[]))
    {
        // Make sure that the push does not involve possibly mutating values
        match pushed_item.kind {
            ExprKind::Path(ref qpath) => {
                match cx.qpath_res(qpath, pushed_item.hir_id) {
                    // immutable bindings that are initialized with literal or constant
                    Res::Local(hir_id) => {
                        let node = cx.tcx.hir_node(hir_id);
                        if let Node::Pat(pat) = node
                            && let PatKind::Binding(bind_ann, ..) = pat.kind
                            && !matches!(bind_ann, BindingMode(_, Mutability::Mut))
                            && let Node::LetStmt(parent_let_expr) = cx.tcx.parent_hir_node(hir_id)
                            && let Some(init) = parent_let_expr.init
                        {
                            match init.kind {
                                // immutable bindings that are initialized with literal
                                ExprKind::Lit(..) => emit_lint(cx, vec, pushed_item, ctxt, msrv),
                                // immutable bindings that are initialized with constant
                                ExprKind::Path(ref path) => {
                                    if let Res::Def(DefKind::Const, ..) = cx.qpath_res(path, init.hir_id) {
                                        emit_lint(cx, vec, pushed_item, ctxt, msrv);
                                    }
                                },
                                _ => {},
                            }
                        }
                    },
                    // constant
                    Res::Def(DefKind::Const, ..) => emit_lint(cx, vec, pushed_item, ctxt, msrv),
                    _ => {},
                }
            },
            ExprKind::Lit(..) => emit_lint(cx, vec, pushed_item, ctxt, msrv),
            _ => {},
        }
    }
}

// Scans the body of the for loop and determines whether lint should be given
struct SameItemPushVisitor<'a, 'tcx> {
    non_deterministic_expr: bool,
    multiple_pushes: bool,
    // this field holds the last vec push operation visited, which should be the only push seen
    vec_push: Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, SyntaxContext)>,
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
        if !self.non_deterministic_expr
            && !self.multiple_pushes
            && let Some((vec, _, _)) = self.vec_push
            && let Some(hir_id) = path_to_local(vec)
        {
            !self.used_locals.contains(&hir_id)
        } else {
            false
        }
    }
}

impl<'tcx> Visitor<'tcx> for SameItemPushVisitor<'_, 'tcx> {
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
        for stmt in b.stmts {
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
        }
        // Current statement is a push ...check whether another
        // push had been previously done
        else if self.vec_push.is_none() {
            self.vec_push = vec_push_option;
        } else {
            // There are multiple pushes ... don't lint
            self.multiple_pushes = true;
        }
    }
}

// Given some statement, determine if that statement is a push on a Vec. If it is, return
// the Vec being pushed into and the item being pushed
fn get_vec_push<'tcx>(
    cx: &LateContext<'tcx>,
    stmt: &'tcx Stmt<'_>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, SyntaxContext)> {
    if let StmtKind::Semi(semi_stmt) = &stmt.kind
            // Extract method being called and figure out the parameters for the method call
            && let ExprKind::MethodCall(path, self_expr, [pushed_item], _) = &semi_stmt.kind
            // Check that the method being called is push() on a Vec
            && path.ident.name == sym::push
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(self_expr), sym::Vec)
    {
        return Some((self_expr, pushed_item, semi_stmt.span.ctxt()));
    }
    None
}
