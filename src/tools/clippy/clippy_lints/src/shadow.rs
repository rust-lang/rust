use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::path_to_local_id;
use clippy_utils::source::snippet;
use clippy_utils::visitors::{Descend, Visitable, for_each_expr};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::hir_id::ItemLocalId;
use rustc_hir::{Block, Body, BodyOwnerKind, Expr, ExprKind, HirId, LetExpr, Node, Pat, PatKind, QPath, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that shadow other bindings already in
    /// scope, while just changing reference level or mutability.
    ///
    /// ### Why restrict this?
    /// To require that what are formally distinct variables be given distinct names.
    ///
    /// See also `shadow_reuse` and `shadow_unrelated` for other restrictions on shadowing.
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 1;
    /// let x = &x;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = 1;
    /// let y = &x; // use different variable name
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHADOW_SAME,
    restriction,
    "rebinding a name to itself, e.g., `let mut x = &mut x`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that shadow other bindings already in
    /// scope, while reusing the original value.
    ///
    /// ### Why restrict this?
    /// Some argue that name shadowing like this hurts readability,
    /// because a value may be bound to different things depending on position in
    /// the code.
    ///
    /// See also `shadow_same` and `shadow_unrelated` for other restrictions on shadowing.
    ///
    /// ### Example
    /// ```no_run
    /// let x = 2;
    /// let x = x + 1;
    /// ```
    /// use different variable name:
    /// ```no_run
    /// let x = 2;
    /// let y = x + 1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHADOW_REUSE,
    restriction,
    "rebinding a name to an expression that re-uses the original value, e.g., `let x = x + 1`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that shadow other bindings already in
    /// scope, either without an initialization or with one that does not even use
    /// the original value.
    ///
    /// ### Why restrict this?
    /// Shadowing a binding with a closely related one is part of idiomatic Rust,
    /// but shadowing a binding by accident with an unrelated one may indicate a mistake.
    ///
    /// Additionally, name shadowing in general can hurt readability, especially in
    /// large code bases, because it is easy to lose track of the active binding at
    /// any place in the code. If linting against all shadowing is desired, you may wish
    /// to use the `shadow_same` and `shadow_reuse` lints as well.
    ///
    /// ### Example
    /// ```no_run
    /// # let y = 1;
    /// # let z = 2;
    /// let x = y;
    /// let x = z; // shadows the earlier binding
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let y = 1;
    /// # let z = 2;
    /// let x = y;
    /// let w = z; // use different variable name
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHADOW_UNRELATED,
    restriction,
    "rebinding a name without even using the original value"
}

#[derive(Default)]
pub(crate) struct Shadow {
    bindings: Vec<(FxHashMap<Symbol, Vec<ItemLocalId>>, LocalDefId)>,
}

impl_lint_pass!(Shadow => [SHADOW_SAME, SHADOW_REUSE, SHADOW_UNRELATED]);

impl<'tcx> LateLintPass<'tcx> for Shadow {
    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        let PatKind::Binding(_, id, ident, _) = pat.kind else {
            return;
        };

        if pat.span.desugaring_kind().is_some() || pat.span.from_expansion() {
            return;
        }

        if ident.span.from_expansion() || ident.span.is_dummy() {
            return;
        }

        let HirId { owner, local_id } = id;
        // get (or insert) the list of items for this owner and symbol
        let (ref mut data, scope_owner) = *self.bindings.last_mut().unwrap();
        let items_with_name = data.entry(ident.name).or_default();

        // check other bindings with the same name, most recently seen first
        for &prev in items_with_name.iter().rev() {
            if prev == local_id {
                // repeated binding in an `Or` pattern
                return;
            }

            if is_shadow(cx, scope_owner, prev, local_id) {
                let prev_hir_id = HirId { owner, local_id: prev };
                lint_shadow(cx, pat, prev_hir_id, ident.span);
                // only lint against the "nearest" shadowed binding
                break;
            }
        }
        // store the binding
        items_with_name.push(local_id);
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &Body<'_>) {
        let owner_id = cx.tcx.hir_body_owner_def_id(body.id());
        if !matches!(cx.tcx.hir_body_owner_kind(owner_id), BodyOwnerKind::Closure) {
            self.bindings.push((FxHashMap::default(), owner_id));
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &Body<'_>) {
        if !matches!(
            cx.tcx.hir_body_owner_kind(cx.tcx.hir_body_owner_def_id(body.id())),
            BodyOwnerKind::Closure
        ) {
            self.bindings.pop();
        }
    }
}

fn is_shadow(cx: &LateContext<'_>, owner: LocalDefId, first: ItemLocalId, second: ItemLocalId) -> bool {
    let scope_tree = cx.tcx.region_scope_tree(owner.to_def_id());
    if let Some(first_scope) = scope_tree.var_scope(first) {
        if let Some(second_scope) = scope_tree.var_scope(second) {
            return scope_tree.is_subscope_of(second_scope, first_scope);
        }
    }

    false
}

/// Checks if the given local is used, except for in child expression of `except`.
///
/// This is a version of [`is_local_used`](clippy_utils::visitors::is_local_used), used to
/// implement the fix for <https://github.com/rust-lang/rust-clippy/issues/10780>.
pub fn is_local_used_except<'tcx>(
    cx: &LateContext<'tcx>,
    visitable: impl Visitable<'tcx>,
    id: HirId,
    except: Option<HirId>,
) -> bool {
    for_each_expr(cx, visitable, |e| {
        if except.is_some_and(|it| it == e.hir_id) {
            ControlFlow::Continue(Descend::No)
        } else if path_to_local_id(e, id) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(Descend::Yes)
        }
    })
    .is_some()
}

fn lint_shadow(cx: &LateContext<'_>, pat: &Pat<'_>, shadowed: HirId, span: Span) {
    let (lint, msg) = match find_init(cx, pat.hir_id) {
        Some((expr, _)) if is_self_shadow(cx, pat, expr, shadowed) => {
            let msg = format!(
                "`{}` is shadowed by itself in `{}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, expr.span, "..")
            );
            (SHADOW_SAME, msg)
        },
        Some((expr, except)) if is_local_used_except(cx, expr, shadowed, except) => {
            let msg = format!("`{}` is shadowed", snippet(cx, pat.span, "_"));
            (SHADOW_REUSE, msg)
        },
        _ => {
            let msg = format!("`{}` shadows a previous, unrelated binding", snippet(cx, pat.span, "_"));
            (SHADOW_UNRELATED, msg)
        },
    };
    span_lint_and_then(cx, lint, span, msg, |diag| {
        diag.span_note(cx.tcx.hir_span(shadowed), "previous binding is here");
    });
}

/// Returns true if the expression is a simple transformation of a local binding such as `&x`
fn is_self_shadow(cx: &LateContext<'_>, pat: &Pat<'_>, mut expr: &Expr<'_>, hir_id: HirId) -> bool {
    let is_direct_binding = cx
        .tcx
        .hir_parent_iter(pat.hir_id)
        .map_while(|(_id, node)| match node {
            Node::Pat(pat) => Some(pat),
            _ => None,
        })
        .all(|pat| matches!(pat.kind, PatKind::Ref(..) | PatKind::Or(_)));
    if !is_direct_binding {
        return false;
    }
    loop {
        expr = match expr.kind {
            ExprKind::AddrOf(_, _, e)
            | ExprKind::Block(
                &Block {
                    stmts: [],
                    expr: Some(e),
                    ..
                },
                _,
            )
            | ExprKind::Unary(UnOp::Deref, e) => e,
            ExprKind::Path(QPath::Resolved(None, path)) => break path.res == Res::Local(hir_id),
            _ => break false,
        }
    }
}

/// Finds the "init" expression for a pattern: `let <pat> = <init>;` (or `if let`) or
/// `match <init> { .., <pat> => .., .. }`
///
/// For closure arguments passed to a method call, returns the method call, and the `HirId` of the
/// closure (which will later be skipped). This is for <https://github.com/rust-lang/rust-clippy/issues/10780>
fn find_init<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<(&'tcx Expr<'tcx>, Option<HirId>)> {
    for (hir_id, node) in cx.tcx.hir_parent_iter(hir_id) {
        let init = match node {
            Node::Arm(_) | Node::Pat(_) | Node::PatField(_) | Node::Param(_) => continue,
            Node::Expr(expr) => match expr.kind {
                ExprKind::Match(e, _, _) | ExprKind::Let(&LetExpr { init: e, .. }) => Some((e, None)),
                // If we're a closure argument, then a parent call is also an associated item.
                ExprKind::Closure(_) => {
                    if let Some((_, node)) = cx.tcx.hir_parent_iter(hir_id).next() {
                        match node {
                            Node::Expr(expr) => match expr.kind {
                                ExprKind::MethodCall(_, _, _, _) | ExprKind::Call(_, _) => Some((expr, Some(hir_id))),
                                _ => None,
                            },
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                _ => None,
            },
            Node::LetStmt(local) => local.init.map(|init| (init, None)),
            _ => None,
        };
        return init;
    }
    None
}
