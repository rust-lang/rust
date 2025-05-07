use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::{get_type_diagnostic_name, is_type_lang_item};
use clippy_utils::visitors::{Visitable, for_each_expr};
use clippy_utils::{get_enclosing_block, path_to_local_id};
use core::ops::ControlFlow;
use rustc_hir::{Body, ExprKind, HirId, LangItem, LetStmt, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for collections that are never queried.
    ///
    /// ### Why is this bad?
    /// Putting effort into constructing a collection but then never querying it might indicate that
    /// the author forgot to do whatever they intended to do with the collection. Example: Clone
    /// a vector, sort it for iteration, but then mistakenly iterate the original vector
    /// instead.
    ///
    /// ### Example
    /// ```no_run
    /// # let samples = vec![3, 1, 2];
    /// let mut sorted_samples = samples.clone();
    /// sorted_samples.sort();
    /// for sample in &samples { // Oops, meant to use `sorted_samples`.
    ///     println!("{sample}");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let samples = vec![3, 1, 2];
    /// let mut sorted_samples = samples.clone();
    /// sorted_samples.sort();
    /// for sample in &sorted_samples {
    ///     println!("{sample}");
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub COLLECTION_IS_NEVER_READ,
    nursery,
    "a collection is never queried"
}
declare_lint_pass!(CollectionIsNeverRead => [COLLECTION_IS_NEVER_READ]);

impl<'tcx> LateLintPass<'tcx> for CollectionIsNeverRead {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'tcx>) {
        // Look for local variables whose type is a container. Search surrounding block for read access.
        if let PatKind::Binding(_, local_id, _, _) = local.pat.kind
            && match_acceptable_type(cx, local)
            && let Some(enclosing_block) = get_enclosing_block(cx, local.hir_id)
            && has_no_read_access(cx, local_id, enclosing_block)
        {
            span_lint(cx, COLLECTION_IS_NEVER_READ, local.span, "collection is never read");
        }
    }
}

fn match_acceptable_type(cx: &LateContext<'_>, local: &LetStmt<'_>) -> bool {
    let ty = cx.typeck_results().pat_ty(local.pat);
    matches!(
        get_type_diagnostic_name(cx, ty),
        Some(
            sym::BTreeMap
                | sym::BTreeSet
                | sym::BinaryHeap
                | sym::HashMap
                | sym::HashSet
                | sym::LinkedList
                | sym::Option
                | sym::Vec
                | sym::VecDeque
        )
    ) || is_type_lang_item(cx, ty, LangItem::String)
}

fn has_no_read_access<'tcx, T: Visitable<'tcx>>(cx: &LateContext<'tcx>, id: HirId, block: T) -> bool {
    let mut has_access = false;
    let mut has_read_access = false;

    // Inspect all expressions and sub-expressions in the block.
    for_each_expr(cx, block, |expr| {
        // Ignore expressions that are not simply `id`.
        if !path_to_local_id(expr, id) {
            return ControlFlow::Continue(());
        }

        // `id` is being accessed. Investigate if it's a read access.
        has_access = true;

        // `id` appearing in the left-hand side of an assignment is not a read access:
        //
        // id = ...; // Not reading `id`.
        if let Node::Expr(parent) = cx.tcx.parent_hir_node(expr.hir_id)
            && let ExprKind::Assign(lhs, ..) = parent.kind
            && path_to_local_id(lhs, id)
        {
            return ControlFlow::Continue(());
        }

        // Look for method call with receiver `id`. It might be a non-read access:
        //
        // id.foo(args)
        //
        // Only assuming this for "official" methods defined on the type. For methods defined in extension
        // traits (identified as local, based on the orphan rule), pessimistically assume that they might
        // have side effects, so consider them a read.
        if let Node::Expr(parent) = cx.tcx.parent_hir_node(expr.hir_id)
            && let ExprKind::MethodCall(_, receiver, args, _) = parent.kind
            && path_to_local_id(receiver, id)
            && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(parent.hir_id)
            && !method_def_id.is_local()
        {
            // If this "official" method takes closures,
            // it has read access if one of the closures has read access.
            //
            // items.retain(|item| send_item(item).is_ok());
            let is_read_in_closure_arg = args.iter().any(|arg| {
                if let ExprKind::Closure(closure) = arg.kind
                    // To keep things simple, we only check the first param to see if its read.
                    && let Body { params: [param, ..], value } = cx.tcx.hir_body(closure.body)
                {
                    !has_no_read_access(cx, param.hir_id, *value)
                } else {
                    false
                }
            });
            if is_read_in_closure_arg {
                has_read_access = true;
                return ControlFlow::Break(());
            }

            // The method call is a statement, so the return value is not used. That's not a read access:
            //
            // id.foo(args);
            if let Node::Stmt(..) = cx.tcx.parent_hir_node(parent.hir_id) {
                return ControlFlow::Continue(());
            }

            // The method call is not a statement, so its return value is used somehow but its type is the
            // unit type, so this is not a real read access. Examples:
            //
            // let y = x.clear();
            // println!("{:?}", x.clear());
            if cx.typeck_results().expr_ty(parent).is_unit() {
                return ControlFlow::Continue(());
            }
        }

        // Any other access to `id` is a read access. Stop searching.
        has_read_access = true;
        ControlFlow::Break(())
    });

    // Ignore collections that have no access at all. Other lints should catch them.
    has_access && !has_read_access
}
