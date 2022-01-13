use clippy_utils::diagnostics::{span_lint_and_note, span_lint_and_sugg};
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::ty::{has_drop, is_copy};
use clippy_utils::{any_parent_is_automatically_derived, contains_name, match_def_path, paths};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Block, Expr, ExprKind, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for literal calls to `Default::default()`.
    ///
    /// ### Why is this bad?
    /// It's more clear to the reader to use the name of the type whose default is
    /// being gotten than the generic `Default`.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let s: String = Default::default();
    ///
    /// // Good
    /// let s = String::default();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DEFAULT_TRAIT_ACCESS,
    pedantic,
    "checks for literal calls to `Default::default()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for immediate reassignment of fields initialized
    /// with Default::default().
    ///
    /// ### Why is this bad?
    ///It's more idiomatic to use the [functional update syntax](https://doc.rust-lang.org/reference/expressions/struct-expr.html#functional-update-syntax).
    ///
    /// ### Known problems
    /// Assignments to patterns that are of tuple type are not linted.
    ///
    /// ### Example
    /// Bad:
    /// ```
    /// # #[derive(Default)]
    /// # struct A { i: i32 }
    /// let mut a: A = Default::default();
    /// a.i = 42;
    /// ```
    /// Use instead:
    /// ```
    /// # #[derive(Default)]
    /// # struct A { i: i32 }
    /// let a = A {
    ///     i: 42,
    ///     .. Default::default()
    /// };
    /// ```
    #[clippy::version = "1.49.0"]
    pub FIELD_REASSIGN_WITH_DEFAULT,
    style,
    "binding initialized with Default should have its fields set in the initializer"
}

#[derive(Default)]
pub struct Default {
    // Spans linted by `field_reassign_with_default`.
    reassigned_linted: FxHashSet<Span>,
}

impl_lint_pass!(Default => [DEFAULT_TRAIT_ACCESS, FIELD_REASSIGN_WITH_DEFAULT]);

impl<'tcx> LateLintPass<'tcx> for Default {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion();
            // Avoid cases already linted by `field_reassign_with_default`
            if !self.reassigned_linted.contains(&expr.span);
            if let ExprKind::Call(path, ..) = expr.kind;
            if !any_parent_is_automatically_derived(cx.tcx, expr.hir_id);
            if let ExprKind::Path(ref qpath) = path.kind;
            if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            // Detect and ignore <Foo as Default>::default() because these calls do explicitly name the type.
            if let QPath::Resolved(None, _path) = qpath;
            let expr_ty = cx.typeck_results().expr_ty(expr);
            if let ty::Adt(def, ..) = expr_ty.kind();
            then {
                // TODO: Work out a way to put "whatever the imported way of referencing
                // this type in this file" rather than a fully-qualified type.
                let replacement = format!("{}::default()", cx.tcx.def_path_str(def.did));
                span_lint_and_sugg(
                    cx,
                    DEFAULT_TRAIT_ACCESS,
                    expr.span,
                    &format!("calling `{}` is more clear than this expression", replacement),
                    "try",
                    replacement,
                    Applicability::Unspecified, // First resolve the TODO above
                );
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &Block<'tcx>) {
        // start from the `let mut _ = _::default();` and look at all the following
        // statements, see if they re-assign the fields of the binding
        let stmts_head = match block.stmts {
            // Skip the last statement since there cannot possibly be any following statements that re-assign fields.
            [head @ .., _] if !head.is_empty() => head,
            _ => return,
        };
        for (stmt_idx, stmt) in stmts_head.iter().enumerate() {
            // find all binding statements like `let mut _ = T::default()` where `T::default()` is the
            // `default` method of the `Default` trait, and store statement index in current block being
            // checked and the name of the bound variable
            let (local, variant, binding_name, binding_type, span) = if_chain! {
                // only take `let ...` statements
                if let StmtKind::Local(local) = stmt.kind;
                if let Some(expr) = local.init;
                if !any_parent_is_automatically_derived(cx.tcx, expr.hir_id);
                if !expr.span.from_expansion();
                // only take bindings to identifiers
                if let PatKind::Binding(_, binding_id, ident, _) = local.pat.kind;
                // only when assigning `... = Default::default()`
                if is_expr_default(expr, cx);
                let binding_type = cx.typeck_results().node_type(binding_id);
                if let Some(adt) = binding_type.ty_adt_def();
                if adt.is_struct();
                let variant = adt.non_enum_variant();
                if adt.did.is_local() || !variant.is_field_list_non_exhaustive();
                let module_did = cx.tcx.parent_module(stmt.hir_id).to_def_id();
                if variant
                    .fields
                    .iter()
                    .all(|field| field.vis.is_accessible_from(module_did, cx.tcx));
                let all_fields_are_copy = variant
                    .fields
                    .iter()
                    .all(|field| {
                        is_copy(cx, cx.tcx.type_of(field.did))
                    });
                if !has_drop(cx, binding_type) || all_fields_are_copy;
                then {
                    (local, variant, ident.name, binding_type, expr.span)
                } else {
                    continue;
                }
            };

            // find all "later statement"'s where the fields of the binding set as
            // Default::default() get reassigned, unless the reassignment refers to the original binding
            let mut first_assign = None;
            let mut assigned_fields = Vec::new();
            let mut cancel_lint = false;
            for consecutive_statement in &block.stmts[stmt_idx + 1..] {
                // find out if and which field was set by this `consecutive_statement`
                if let Some((field_ident, assign_rhs)) = field_reassigned_by_stmt(consecutive_statement, binding_name) {
                    // interrupt and cancel lint if assign_rhs references the original binding
                    if contains_name(binding_name, assign_rhs) {
                        cancel_lint = true;
                        break;
                    }

                    // if the field was previously assigned, replace the assignment, otherwise insert the assignment
                    if let Some(prev) = assigned_fields
                        .iter_mut()
                        .find(|(field_name, _)| field_name == &field_ident.name)
                    {
                        *prev = (field_ident.name, assign_rhs);
                    } else {
                        assigned_fields.push((field_ident.name, assign_rhs));
                    }

                    // also set first instance of error for help message
                    if first_assign.is_none() {
                        first_assign = Some(consecutive_statement);
                    }
                }
                // interrupt if no field was assigned, since we only want to look at consecutive statements
                else {
                    break;
                }
            }

            // if there are incorrectly assigned fields, do a span_lint_and_note to suggest
            // construction using `Ty { fields, ..Default::default() }`
            if !assigned_fields.is_empty() && !cancel_lint {
                // if all fields of the struct are not assigned, add `.. Default::default()` to the suggestion.
                let ext_with_default = !variant
                    .fields
                    .iter()
                    .all(|field| assigned_fields.iter().any(|(a, _)| a == &field.name));

                let field_list = assigned_fields
                    .into_iter()
                    .map(|(field, rhs)| {
                        // extract and store the assigned value for help message
                        let value_snippet = snippet_with_macro_callsite(cx, rhs.span, "..");
                        format!("{}: {}", field, value_snippet)
                    })
                    .collect::<Vec<String>>()
                    .join(", ");

                // give correct suggestion if generics are involved (see #6944)
                let binding_type = if_chain! {
                    if let ty::Adt(adt_def, substs) = binding_type.kind();
                    if !substs.is_empty();
                    then {
                        let adt_def_ty_name = cx.tcx.item_name(adt_def.did);
                        let generic_args = substs.iter().collect::<Vec<_>>();
                        let tys_str = generic_args
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{}::<{}>", adt_def_ty_name, &tys_str)
                    } else {
                        binding_type.to_string()
                    }
                };

                let sugg = if ext_with_default {
                    if field_list.is_empty() {
                        format!("{}::default()", binding_type)
                    } else {
                        format!("{} {{ {}, ..Default::default() }}", binding_type, field_list)
                    }
                } else {
                    format!("{} {{ {} }}", binding_type, field_list)
                };

                // span lint once per statement that binds default
                span_lint_and_note(
                    cx,
                    FIELD_REASSIGN_WITH_DEFAULT,
                    first_assign.unwrap().span,
                    "field assignment outside of initializer for an instance created with Default::default()",
                    Some(local.span),
                    &format!(
                        "consider initializing the variable with `{}` and removing relevant reassignments",
                        sugg
                    ),
                );
                self.reassigned_linted.insert(span);
            }
        }
    }
}

/// Checks if the given expression is the `default` method belonging to the `Default` trait.
fn is_expr_default<'tcx>(expr: &'tcx Expr<'tcx>, cx: &LateContext<'tcx>) -> bool {
    if_chain! {
        if let ExprKind::Call(fn_expr, _) = &expr.kind;
        if let ExprKind::Path(qpath) = &fn_expr.kind;
        if let Res::Def(_, def_id) = cx.qpath_res(qpath, fn_expr.hir_id);
        then {
            // right hand side of assignment is `Default::default`
            match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD)
        } else {
            false
        }
    }
}

/// Returns the reassigned field and the assigning expression (right-hand side of assign).
fn field_reassigned_by_stmt<'tcx>(this: &Stmt<'tcx>, binding_name: Symbol) -> Option<(Ident, &'tcx Expr<'tcx>)> {
    if_chain! {
        // only take assignments
        if let StmtKind::Semi(later_expr) = this.kind;
        if let ExprKind::Assign(assign_lhs, assign_rhs, _) = later_expr.kind;
        // only take assignments to fields where the left-hand side field is a field of
        // the same binding as the previous statement
        if let ExprKind::Field(binding, field_ident) = assign_lhs.kind;
        if let ExprKind::Path(QPath::Resolved(_, path)) = binding.kind;
        if let Some(second_binding_name) = path.segments.last();
        if second_binding_name.ident.name == binding_name;
        then {
            Some((field_ident, assign_rhs))
        } else {
            None
        }
    }
}
