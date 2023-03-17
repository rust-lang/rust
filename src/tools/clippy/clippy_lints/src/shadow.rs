use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::source::snippet;
use clippy_utils::visitors::is_local_used;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::hir_id::ItemLocalId;
use rustc_hir::{Block, Body, BodyOwnerKind, Expr, ExprKind, HirId, Let, Node, Pat, PatKind, QPath, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bindings that shadow other bindings already in
    /// scope, while just changing reference level or mutability.
    ///
    /// ### Why is this bad?
    /// Not much, in fact it's a very common pattern in Rust
    /// code. Still, some may opt to avoid it in their code base, they can set this
    /// lint to `Warn`.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// let x = &x;
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
    /// ### Why is this bad?
    /// Not too much, in fact it's a common pattern in Rust
    /// code. Still, some argue that name shadowing like this hurts readability,
    /// because a value may be bound to different things depending on position in
    /// the code.
    ///
    /// ### Example
    /// ```rust
    /// let x = 2;
    /// let x = x + 1;
    /// ```
    /// use different variable name:
    /// ```rust
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
    /// ### Why is this bad?
    /// Name shadowing can hurt readability, especially in
    /// large code bases, because it is easy to lose track of the active binding at
    /// any place in the code. This can be alleviated by either giving more specific
    /// names to bindings or introducing more scopes to contain the bindings.
    ///
    /// ### Example
    /// ```rust
    /// # let y = 1;
    /// # let z = 2;
    /// let x = y;
    /// let x = z; // shadows the earlier binding
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
        let PatKind::Binding(_, id, ident, _) = pat.kind else { return };

        if pat.span.desugaring_kind().is_some() {
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
        let hir = cx.tcx.hir();
        let owner_id = hir.body_owner_def_id(body.id());
        if !matches!(hir.body_owner_kind(owner_id), BodyOwnerKind::Closure) {
            self.bindings.push((FxHashMap::default(), owner_id));
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &Body<'_>) {
        let hir = cx.tcx.hir();
        if !matches!(
            hir.body_owner_kind(hir.body_owner_def_id(body.id())),
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

fn lint_shadow(cx: &LateContext<'_>, pat: &Pat<'_>, shadowed: HirId, span: Span) {
    let (lint, msg) = match find_init(cx, pat.hir_id) {
        Some(expr) if is_self_shadow(cx, pat, expr, shadowed) => {
            let msg = format!(
                "`{}` is shadowed by itself in `{}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, expr.span, "..")
            );
            (SHADOW_SAME, msg)
        },
        Some(expr) if is_local_used(cx, expr, shadowed) => {
            let msg = format!("`{}` is shadowed", snippet(cx, pat.span, "_"));
            (SHADOW_REUSE, msg)
        },
        _ => {
            let msg = format!("`{}` shadows a previous, unrelated binding", snippet(cx, pat.span, "_"));
            (SHADOW_UNRELATED, msg)
        },
    };
    span_lint_and_note(
        cx,
        lint,
        span,
        &msg,
        Some(cx.tcx.hir().span(shadowed)),
        "previous binding is here",
    );
}

/// Returns true if the expression is a simple transformation of a local binding such as `&x`
fn is_self_shadow(cx: &LateContext<'_>, pat: &Pat<'_>, mut expr: &Expr<'_>, hir_id: HirId) -> bool {
    let hir = cx.tcx.hir();
    let is_direct_binding = hir
        .parent_iter(pat.hir_id)
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
fn find_init<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Option<&'tcx Expr<'tcx>> {
    for (_, node) in cx.tcx.hir().parent_iter(hir_id) {
        let init = match node {
            Node::Arm(_) | Node::Pat(_) => continue,
            Node::Expr(expr) => match expr.kind {
                ExprKind::Match(e, _, _) | ExprKind::Let(&Let { init: e, .. }) => Some(e),
                _ => None,
            },
            Node::Local(local) => local.init,
            _ => None,
        };
        return init;
    }
    None
}
