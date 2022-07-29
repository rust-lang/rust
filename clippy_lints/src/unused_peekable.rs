use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::{match_type, peel_mid_ty_refs_is_mutable};
use clippy_utils::{fn_def_id, path_to_local_id, paths, peel_ref_operators};
use rustc_ast::Mutability;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{Block, Expr, ExprKind, HirId, Local, Node, PatKind, PathSegment, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the creation of a `peekable` iterator that is never `.peek()`ed
    ///
    /// ### Why is this bad?
    /// Creating a peekable iterator without using any of its methods is likely a mistake,
    /// or just a leftover after a refactor.
    ///
    /// ### Example
    /// ```rust
    /// let collection = vec![1, 2, 3];
    /// let iter = collection.iter().peekable();
    ///
    /// for item in iter {
    ///     // ...
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let collection = vec![1, 2, 3];
    /// let iter = collection.iter();
    ///
    /// for item in iter {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "1.64.0"]
    pub UNUSED_PEEKABLE,
    suspicious,
    "creating a peekable iterator without using any of its methods"
}

#[derive(Default)]
pub struct UnusedPeekable {
    visited: Vec<HirId>,
}

impl_lint_pass!(UnusedPeekable => [UNUSED_PEEKABLE]);

impl<'tcx> LateLintPass<'tcx> for UnusedPeekable {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &Block<'tcx>) {
        // Don't lint `Peekable`s returned from a block
        if let Some(expr) = block.expr
            && let Some(ty) = cx.typeck_results().expr_ty_opt(peel_ref_operators(cx, expr))
            && match_type(cx, ty, &paths::PEEKABLE)
        {
            return;
        }

        for (idx, stmt) in block.stmts.iter().enumerate() {
            if !stmt.span.from_expansion()
                && let StmtKind::Local(local) = stmt.kind
                && !self.visited.contains(&local.pat.hir_id)
                && let PatKind::Binding(_, _, ident, _) = local.pat.kind
                && let Some(init) = local.init
                && !init.span.from_expansion()
                && let Some(ty) = cx.typeck_results().expr_ty_opt(init)
                && let (ty, _, Mutability::Mut) = peel_mid_ty_refs_is_mutable(ty)
                && match_type(cx, ty, &paths::PEEKABLE)
            {
                let mut vis = PeekableVisitor::new(cx, local.pat.hir_id);

                if idx + 1 == block.stmts.len() && block.expr.is_none() {
                    return;
                }

                for stmt in &block.stmts[idx..] {
                    vis.visit_stmt(stmt);
                }

                if let Some(expr) = block.expr {
                    vis.visit_expr(expr);
                }

                if !vis.found_peek_call {
                    span_lint_and_help(
                        cx,
                        UNUSED_PEEKABLE,
                        ident.span,
                        "`peek` never called on `Peekable` iterator",
                        None,
                        "consider removing the call to `peekable`"
                   );
                }
            }
        }
    }
}

struct PeekableVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    expected_hir_id: HirId,
    found_peek_call: bool,
}

impl<'a, 'tcx> PeekableVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, expected_hir_id: HirId) -> Self {
        Self {
            cx,
            expected_hir_id,
            found_peek_call: false,
        }
    }
}

impl<'tcx> Visitor<'_> for PeekableVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, ex: &'_ Expr<'_>) {
        if path_to_local_id(ex, self.expected_hir_id) {
            for (_, node) in self.cx.tcx.hir().parent_iter(ex.hir_id) {
                match node {
                    Node::Expr(expr) => {
                        match expr.kind {
                            // some_function(peekable)
                            //
                            // If the Peekable is passed to a function, stop
                            ExprKind::Call(_, args) => {
                                if let Some(func_did) = fn_def_id(self.cx, expr)
                                    && let Ok(into_iter_did) = self
                                        .cx
                                        .tcx
                                        .lang_items()
                                        .require(LangItem::IntoIterIntoIter)
                                    && func_did == into_iter_did
                                {
                                    // Probably a for loop desugar, stop searching
                                    return;
                                }

                                for arg in args.iter().map(|arg| peel_ref_operators(self.cx, arg)) {
                                    if let ExprKind::Path(_) = arg.kind
                                        && let Some(ty) = self
                                            .cx
                                            .typeck_results()
                                            .expr_ty_opt(arg)
                                            .map(Ty::peel_refs)
                                        && match_type(self.cx, ty, &paths::PEEKABLE)
                                    {
                                        self.found_peek_call = true;
                                        return;
                                    }
                                }
                            },
                            // Peekable::peek()
                            ExprKind::MethodCall(PathSegment { ident: method_name, .. }, [arg, ..], _) => {
                                let arg = peel_ref_operators(self.cx, arg);
                                let method_name = method_name.name.as_str();

                                if (method_name == "peek"
                                    || method_name == "peek_mut"
                                    || method_name == "next_if"
                                    || method_name == "next_if_eq")
                                    && let ExprKind::Path(_) = arg.kind
                                    && let Some(ty) = self.cx.typeck_results().expr_ty_opt(arg).map(Ty::peel_refs)
                                    && match_type(self.cx, ty, &paths::PEEKABLE)
                                {
                                    self.found_peek_call = true;
                                    return;
                                }
                            },
                            // Don't bother if moved into a struct
                            ExprKind::Struct(..) => {
                                self.found_peek_call = true;
                                return;
                            },
                            _ => {},
                        }
                    },
                    Node::Local(Local { init: Some(init), .. }) => {
                        if let Some(ty) = self.cx.typeck_results().expr_ty_opt(init)
                            && let (ty, _, Mutability::Mut) = peel_mid_ty_refs_is_mutable(ty)
                            && match_type(self.cx, ty, &paths::PEEKABLE)
                        {
                            self.found_peek_call = true;
                            return;
                        }

                        break;
                    },
                    Node::Stmt(stmt) => match stmt.kind {
                        StmtKind::Expr(_) | StmtKind::Semi(_) => {},
                        _ => {
                            self.found_peek_call = true;
                            return;
                        },
                    },
                    Node::Block(_) => {},
                    _ => {
                        break;
                    },
                }
            }
        }

        walk_expr(self, ex);
    }
}
