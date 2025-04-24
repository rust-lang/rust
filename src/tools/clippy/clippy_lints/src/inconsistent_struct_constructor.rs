use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::fulfill_or_allowed;
use clippy_utils::source::snippet;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for struct constructors where the order of the field
    /// init in the constructor is inconsistent with the order in the
    /// struct definition.
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
    "the order of the field init is inconsistent with the order in the struct definition"
}

pub struct InconsistentStructConstructor {
    check_inconsistent_struct_field_initializers: bool,
}

impl InconsistentStructConstructor {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            check_inconsistent_struct_field_initializers: conf.check_inconsistent_struct_field_initializers,
        }
    }
}

impl_lint_pass!(InconsistentStructConstructor => [INCONSISTENT_STRUCT_CONSTRUCTOR]);

impl<'tcx> LateLintPass<'tcx> for InconsistentStructConstructor {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        let ExprKind::Struct(_, fields, _) = expr.kind else {
            return;
        };
        let all_fields_are_shorthand = fields.iter().all(|f| f.is_shorthand);
        let applicability = if all_fields_are_shorthand {
            Applicability::MachineApplicable
        } else if self.check_inconsistent_struct_field_initializers {
            Applicability::MaybeIncorrect
        } else {
            return;
        };
        if !expr.span.from_expansion()
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

            let span = field_with_attrs_span(cx.tcx, fields.first().unwrap())
                .with_hi(field_with_attrs_span(cx.tcx, fields.last().unwrap()).hi());

            if !fulfill_or_allowed(cx, INCONSISTENT_STRUCT_CONSTRUCTOR, Some(ty_hir_id)) {
                span_lint_and_then(
                    cx,
                    INCONSISTENT_STRUCT_CONSTRUCTOR,
                    span,
                    "struct constructor field order is inconsistent with struct definition field order",
                    |diag| {
                        let msg = if all_fields_are_shorthand {
                            "try"
                        } else {
                            "if the field evaluation order doesn't matter, try"
                        };
                        let sugg = suggestion(cx, fields, &def_order_map);
                        diag.span_suggestion(span, msg, sugg, applicability);
                    },
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

fn suggestion<'tcx>(
    cx: &LateContext<'_>,
    fields: &'tcx [hir::ExprField<'tcx>],
    def_order_map: &FxHashMap<Symbol, usize>,
) -> String {
    let ws = fields
        .windows(2)
        .map(|w| {
            let w0_span = field_with_attrs_span(cx.tcx, &w[0]);
            let w1_span = field_with_attrs_span(cx.tcx, &w[1]);
            let span = w0_span.between(w1_span);
            snippet(cx, span, " ")
        })
        .collect::<Vec<_>>();

    let mut fields = fields.to_vec();
    fields.sort_unstable_by_key(|field| def_order_map[&field.ident.name]);
    let field_snippets = fields
        .iter()
        .map(|field| snippet(cx, field_with_attrs_span(cx.tcx, field), ".."))
        .collect::<Vec<_>>();

    assert_eq!(field_snippets.len(), ws.len() + 1);

    let mut sugg = String::new();
    for i in 0..field_snippets.len() {
        sugg += &field_snippets[i];
        if i < ws.len() {
            sugg += &ws[i];
        }
    }
    sugg
}

fn field_with_attrs_span(tcx: TyCtxt<'_>, field: &hir::ExprField<'_>) -> Span {
    if let Some(attr) = tcx.hir_attrs(field.hir_id).first() {
        field.span.with_lo(attr.span().lo())
    } else {
        field.span
    }
}
