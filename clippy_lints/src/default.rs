use crate::utils::{any_parent_is_automatically_derived, contains_name, match_def_path, paths, qpath_res, snippet};
use crate::utils::{span_lint_and_note, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Block, Expr, ExprKind, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Adt, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for literal calls to `Default::default()`.
    ///
    /// **Why is this bad?** It's more clear to the reader to use the name of the type whose default is
    /// being gotten than the generic `Default`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad
    /// let s: String = Default::default();
    ///
    /// // Good
    /// let s = String::default();
    /// ```
    pub DEFAULT_TRAIT_ACCESS,
    pedantic,
    "checks for literal calls to `Default::default()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for immediate reassignment of fields initialized
    /// with Default::default().
    ///
    /// **Why is this bad?**It's more idiomatic to use the [functional update syntax](https://doc.rust-lang.org/reference/expressions/struct-expr.html#functional-update-syntax).
    ///
    /// **Known problems:** Assignments to patterns that are of tuple type are not linted.
    ///
    /// **Example:**
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

impl LateLintPass<'_> for Default {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            // Avoid cases already linted by `field_reassign_with_default`
            if !self.reassigned_linted.contains(&expr.span);
            if let ExprKind::Call(ref path, ..) = expr.kind;
            if !any_parent_is_automatically_derived(cx.tcx, expr.hir_id);
            if let ExprKind::Path(ref qpath) = path.kind;
            if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            // Detect and ignore <Foo as Default>::default() because these calls do explicitly name the type.
            if let QPath::Resolved(None, _path) = qpath;
            then {
                let expr_ty = cx.typeck_results().expr_ty(expr);
                if let ty::Adt(def, ..) = expr_ty.kind() {
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
    }

    fn check_block<'tcx>(&mut self, cx: &LateContext<'tcx>, block: &Block<'tcx>) {
        // find all binding statements like `let mut _ = T::default()` where `T::default()` is the
        // `default` method of the `Default` trait, and store statement index in current block being
        // checked and the name of the bound variable
        let binding_statements_using_default = enumerate_bindings_using_default(cx, block);

        // start from the `let mut _ = _::default();` and look at all the following
        // statements, see if they re-assign the fields of the binding
        for (stmt_idx, binding_name, binding_type, span) in binding_statements_using_default {
            // the last statement of a block cannot trigger the lint
            if stmt_idx == block.stmts.len() - 1 {
                break;
            }

            // find all "later statement"'s where the fields of the binding set as
            // Default::default() get reassigned, unless the reassignment refers to the original binding
            let mut first_assign = None;
            let mut assigned_fields = Vec::new();
            let mut cancel_lint = false;
            for consecutive_statement in &block.stmts[stmt_idx + 1..] {
                // interrupt if the statement is a let binding (`Local`) that shadows the original
                // binding
                if stmt_shadows_binding(consecutive_statement, binding_name) {
                    break;
                }
                // find out if and which field was set by this `consecutive_statement`
                else if let Some((field_ident, assign_rhs)) =
                    field_reassigned_by_stmt(consecutive_statement, binding_name)
                {
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
                // interrupt also if no field was assigned, since we only want to look at consecutive statements
                else {
                    break;
                }
            }

            // if there are incorrectly assigned fields, do a span_lint_and_note to suggest
            // construction using `Ty { fields, ..Default::default() }`
            if !assigned_fields.is_empty() && !cancel_lint {
                // take the original assignment as span
                let stmt = &block.stmts[stmt_idx];

                if let StmtKind::Local(preceding_local) = &stmt.kind {
                    // filter out fields like `= Default::default()`, because the FRU already covers them
                    let assigned_fields = assigned_fields
                        .into_iter()
                        .filter(|(_, rhs)| !is_expr_default(rhs, cx))
                        .collect::<Vec<(Symbol, &Expr<'_>)>>();

                    // if all fields of the struct are not assigned, add `.. Default::default()` to the suggestion.
                    let ext_with_default = !fields_of_type(binding_type)
                        .iter()
                        .all(|field| assigned_fields.iter().any(|(a, _)| a == &field.name));

                    let field_list = assigned_fields
                        .into_iter()
                        .map(|(field, rhs)| {
                            // extract and store the assigned value for help message
                            let value_snippet = snippet(cx, rhs.span, "..");
                            format!("{}: {}", field, value_snippet)
                        })
                        .collect::<Vec<String>>()
                        .join(", ");

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
                        Some(preceding_local.span),
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
}

/// Checks if the given expression is the `default` method belonging to the `Default` trait.
fn is_expr_default<'tcx>(expr: &'tcx Expr<'tcx>, cx: &LateContext<'tcx>) -> bool {
    if_chain! {
        if let ExprKind::Call(ref fn_expr, _) = &expr.kind;
        if let ExprKind::Path(qpath) = &fn_expr.kind;
        if let Res::Def(_, def_id) = qpath_res(cx, qpath, fn_expr.hir_id);
        then {
            // right hand side of assignment is `Default::default`
            match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD)
        } else {
            false
        }
    }
}

/// Returns the block indices, identifiers and types of bindings set as `Default::default()`, except
/// for when the pattern type is a tuple.
fn enumerate_bindings_using_default<'tcx>(
    cx: &LateContext<'tcx>,
    block: &Block<'tcx>,
) -> Vec<(usize, Symbol, Ty<'tcx>, Span)> {
    block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(idx, stmt)| {
            if_chain! {
                // only take `let ...` statements
                if let StmtKind::Local(ref local) = stmt.kind;
                // only take bindings to identifiers
                if let PatKind::Binding(_, _, ident, _) = local.pat.kind;
                // that are not tuples
                let ty = cx.typeck_results().pat_ty(local.pat);
                if !matches!(ty.kind(), ty::Tuple(_));
                // only when assigning `... = Default::default()`
                if let Some(ref expr) = local.init;
                if is_expr_default(expr, cx);
                then {
                    Some((idx, ident.name, ty, expr.span))
                } else {
                    None
                }
            }
        })
        .collect()
}

fn stmt_shadows_binding(this: &Stmt<'_>, shadowed: Symbol) -> bool {
    if let StmtKind::Local(local) = &this.kind {
        if let PatKind::Binding(_, _, ident, _) = local.pat.kind {
            return ident.name == shadowed;
        }
    }
    false
}

/// Returns the reassigned field and the assigning expression (right-hand side of assign).
fn field_reassigned_by_stmt<'tcx>(this: &Stmt<'tcx>, binding_name: Symbol) -> Option<(Ident, &'tcx Expr<'tcx>)> {
    if_chain! {
        // only take assignments
        if let StmtKind::Semi(ref later_expr) = this.kind;
        if let ExprKind::Assign(ref assign_lhs, ref assign_rhs, _) = later_expr.kind;
        // only take assignments to fields where the left-hand side field is a field of
        // the same binding as the previous statement
        if let ExprKind::Field(ref binding, field_ident) = assign_lhs.kind;
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

/// Returns the vec of fields for a struct and an empty vec for non-struct ADTs.
fn fields_of_type(ty: Ty<'_>) -> Vec<Ident> {
    if let Adt(adt, _) = ty.kind() {
        if adt.is_struct() {
            let variant = &adt.non_enum_variant();
            return variant.fields.iter().map(|f| f.ident).collect();
        }
    }
    vec![]
}
