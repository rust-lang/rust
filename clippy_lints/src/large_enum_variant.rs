//! lint when there is a large size difference between variants on an enum

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for large size differences between variants on
    /// `enum`s.
    ///
    /// ### Why is this bad?
    /// Enum size is bounded by the largest variant. Having a
    /// large variant can penalize the memory layout of that enum.
    ///
    /// ### Known problems
    /// This lint obviously cannot take the distribution of
    /// variants in your running program into account. It is possible that the
    /// smaller variants make up less than 1% of all instances, in which case
    /// the overhead is negligible and the boxing is counter-productive. Always
    /// measure the change this lint suggests.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// enum Test {
    ///     A(i32),
    ///     B([i32; 8000]),
    /// }
    ///
    /// // Possibly better
    /// enum Test2 {
    ///     A(i32),
    ///     B(Box<[i32; 8000]>),
    /// }
    /// ```
    pub LARGE_ENUM_VARIANT,
    perf,
    "large size difference between variants on an enum"
}

#[derive(Copy, Clone)]
pub struct LargeEnumVariant {
    maximum_size_difference_allowed: u64,
}

impl LargeEnumVariant {
    #[must_use]
    pub fn new(maximum_size_difference_allowed: u64) -> Self {
        Self {
            maximum_size_difference_allowed,
        }
    }
}

impl_lint_pass!(LargeEnumVariant => [LARGE_ENUM_VARIANT]);

impl<'tcx> LateLintPass<'tcx> for LargeEnumVariant {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if in_external_macro(cx.tcx.sess, item.span) {
            return;
        }
        if let ItemKind::Enum(ref def, _) = item.kind {
            let ty = cx.tcx.type_of(item.def_id);
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");

            let mut largest_variant: Option<(_, _)> = None;
            let mut second_variant: Option<(_, _)> = None;

            for (i, variant) in adt.variants.iter().enumerate() {
                let size: u64 = variant
                    .fields
                    .iter()
                    .filter_map(|f| {
                        let ty = cx.tcx.type_of(f.did);
                        // don't count generics by filtering out everything
                        // that does not have a layout
                        cx.layout_of(ty).ok().map(|l| l.size.bytes())
                    })
                    .sum();

                let grouped = (size, (i, variant));

                if grouped.0 >= largest_variant.map_or(0, |x| x.0) {
                    second_variant = largest_variant;
                    largest_variant = Some(grouped);
                }
            }

            if let (Some(largest), Some(second)) = (largest_variant, second_variant) {
                let difference = largest.0 - second.0;

                if difference > self.maximum_size_difference_allowed {
                    let (i, variant) = largest.1;

                    let help_text = "consider boxing the large fields to reduce the total size of the enum";
                    span_lint_and_then(
                        cx,
                        LARGE_ENUM_VARIANT,
                        def.variants[i].span,
                        "large size difference between variants",
                        |diag| {
                            diag.span_label(
                                def.variants[(largest.1).0].span,
                                &format!("this variant is {} bytes", largest.0),
                            );
                            diag.span_note(
                                def.variants[(second.1).0].span,
                                &format!("and the second-largest variant is {} bytes:", second.0),
                            );
                            if variant.fields.len() == 1 {
                                let span = match def.variants[i].data {
                                    VariantData::Struct(fields, ..) | VariantData::Tuple(fields, ..) => {
                                        fields[0].ty.span
                                    },
                                    VariantData::Unit(..) => unreachable!(),
                                };
                                if let Some(snip) = snippet_opt(cx, span) {
                                    diag.span_suggestion(
                                        span,
                                        help_text,
                                        format!("Box<{}>", snip),
                                        Applicability::MaybeIncorrect,
                                    );
                                    return;
                                }
                            }
                            diag.span_help(def.variants[i].span, help_text);
                        },
                    );
                }
            }
        }
    }
}
