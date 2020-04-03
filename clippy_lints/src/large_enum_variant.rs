//! lint when there is a large size difference between variants on an enum

use crate::utils::{snippet_opt, span_lint_and_then};
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_target::abi::LayoutOf;

declare_clippy_lint! {
    /// **What it does:** Checks for large size differences between variants on
    /// `enum`s.
    ///
    /// **Why is this bad?** Enum size is bounded by the largest variant. Having a
    /// large variant
    /// can penalize the memory layout of that enum.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// enum Test {
    ///     A(i32),
    ///     B([i32; 8000]),
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LargeEnumVariant {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, item: &Item<'_>) {
        let did = cx.tcx.hir().local_def_id(item.hir_id);
        if let ItemKind::Enum(ref def, _) = item.kind {
            let ty = cx.tcx.type_of(did);
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");

            let mut smallest_variant: Option<(_, _)> = None;
            let mut largest_variant: Option<(_, _)> = None;

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

                update_if(&mut smallest_variant, grouped, |a, b| b.0 <= a.0);
                update_if(&mut largest_variant, grouped, |a, b| b.0 >= a.0);
            }

            if let (Some(smallest), Some(largest)) = (smallest_variant, largest_variant) {
                let difference = largest.0 - smallest.0;

                if difference > self.maximum_size_difference_allowed {
                    let (i, variant) = largest.1;

                    span_lint_and_then(
                        cx,
                        LARGE_ENUM_VARIANT,
                        def.variants[i].span,
                        "large size difference between variants",
                        |db| {
                            if variant.fields.len() == 1 {
                                let span = match def.variants[i].data {
                                    VariantData::Struct(ref fields, ..) | VariantData::Tuple(ref fields, ..) => {
                                        fields[0].ty.span
                                    },
                                    VariantData::Unit(..) => unreachable!(),
                                };
                                if let Some(snip) = snippet_opt(cx, span) {
                                    db.span_suggestion(
                                        span,
                                        "consider boxing the large fields to reduce the total size of the \
                                         enum",
                                        format!("Box<{}>", snip),
                                        Applicability::MaybeIncorrect,
                                    );
                                    return;
                                }
                            }
                            db.span_help(
                                def.variants[i].span,
                                "consider boxing the large fields to reduce the total size of the enum",
                            );
                        },
                    );
                }
            }
        }
    }
}

fn update_if<T, F>(old: &mut Option<T>, new: T, f: F)
where
    F: Fn(&T, &T) -> bool,
{
    if let Some(ref mut val) = *old {
        if f(val, &new) {
            *val = new;
        }
    } else {
        *old = Some(new);
    }
}
