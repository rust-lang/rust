use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::fulfill_or_allowed;
use clippy_utils::source::snippet;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Symbol;
use std::fmt::{self, Write as _};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for struct constructors where all fields are shorthand and
    /// the order of the field init shorthand in the constructor is inconsistent
    /// with the order in the struct definition.
    ///
    /// ### Why is this bad?
    /// Since the order of fields in a constructor doesn't affect the
    /// resulted instance as the below example indicates,
    ///
    /// ```no_run
    /// #[derive(Debug, PartialEq, Eq)]
    /// struct Foo {
    ///     x: i32,
    ///     y: i32,
    /// }
    /// let x = 1;
    /// let y = 2;
    ///
    /// // This assertion never fails:
    /// assert_eq!(Foo { x, y }, Foo { y, x });
    /// ```
    ///
    /// inconsistent order can be confusing and decreases readability and consistency.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo {
    ///     x: i32,
    ///     y: i32,
    /// }
    /// let x = 1;
    /// let y = 2;
    ///
    /// Foo { y, x };
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # struct Foo {
    /// #     x: i32,
    /// #     y: i32,
    /// # }
    /// # let x = 1;
    /// # let y = 2;
    /// Foo { x, y };
    /// ```
    #[clippy::version = "1.52.0"]
    pub INCONSISTENT_STRUCT_CONSTRUCTOR,
    pedantic,
    "the order of the field init shorthand is inconsistent with the order in the struct definition"
}

declare_lint_pass!(InconsistentStructConstructor => [INCONSISTENT_STRUCT_CONSTRUCTOR]);

impl<'tcx> LateLintPass<'tcx> for InconsistentStructConstructor {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let ExprKind::Struct(qpath, fields, base) = expr.kind
            && fields.iter().all(|f| f.is_shorthand)
            && !expr.span.from_expansion()
            && let ty = cx.typeck_results().expr_ty(expr)
            && let Some(adt_def) = ty.ty_adt_def()
            && adt_def.is_struct()
            && let Some(local_def_id) = adt_def.did().as_local()
            && let ty_hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id)
            && let Some(variant) = adt_def.variants().iter().next()
        {
            let mut def_order_map = FxHashMap::default();
            for (idx, field) in variant.fields.iter().enumerate() {
                def_order_map.insert(field.name, idx);
            }

            if is_consistent_order(fields, &def_order_map) {
                return;
            }

            let mut ordered_fields: Vec<_> = fields.iter().map(|f| f.ident.name).collect();
            ordered_fields.sort_unstable_by_key(|id| def_order_map[id]);

            let mut fields_snippet = String::new();
            let (last_ident, idents) = ordered_fields.split_last().unwrap();
            for ident in idents {
                let _: fmt::Result = write!(fields_snippet, "{ident}, ");
            }
            fields_snippet.push_str(&last_ident.to_string());

            let base_snippet = if let Some(base) = base {
                format!(", ..{}", snippet(cx, base.span, ".."))
            } else {
                String::new()
            };

            let sugg = format!(
                "{} {{ {fields_snippet}{base_snippet} }}",
                snippet(cx, qpath.span(), ".."),
            );

            if !fulfill_or_allowed(cx, INCONSISTENT_STRUCT_CONSTRUCTOR, Some(ty_hir_id)) {
                span_lint_and_sugg(
                    cx,
                    INCONSISTENT_STRUCT_CONSTRUCTOR,
                    expr.span,
                    "struct constructor field order is inconsistent with struct definition field order",
                    "try",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

// Check whether the order of the fields in the constructor is consistent with the order in the
// definition.
fn is_consistent_order<'tcx>(fields: &'tcx [hir::ExprField<'tcx>], def_order_map: &FxHashMap<Symbol, usize>) -> bool {
    let mut cur_idx = usize::MIN;
    for f in fields {
        let next_idx = def_order_map[&f.ident.name];
        if cur_idx > next_idx {
            return false;
        }
        cur_idx = next_idx;
    }

    true
}
