//! lint when there is a large size difference between variants on an enum

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

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

struct FieldInfo {
    ind: usize,
    size: u64,
}

struct VariantInfo {
    ind: usize,
    size: u64,
    fields_size: Vec<FieldInfo>,
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
            if adt.variants.len() <= 1 {
                return;
            }
            let mut variants_size: Vec<VariantInfo> = adt
                .variants
                .iter()
                .enumerate()
                .map(|(i, variant)| {
                    let mut fields_size = Vec::new();
                    let size: u64 = variant
                        .fields
                        .iter()
                        .enumerate()
                        .filter_map(|(i, f)| {
                            let ty = cx.tcx.type_of(f.did);
                            // don't count generics by filtering out everything
                            // that does not have a layout
                            cx.layout_of(ty).ok().map(|l| {
                                let size = l.size.bytes();
                                fields_size.push(FieldInfo { ind: i, size });
                                size
                            })
                        })
                        .sum();
                    VariantInfo {
                        ind: i,
                        size,
                        fields_size,
                    }
                })
                .collect();

            variants_size.sort_by(|a, b| (b.size.cmp(&a.size)));

            let mut difference = variants_size[0].size - variants_size[1].size;
            if difference > self.maximum_size_difference_allowed {
                let help_text = "consider boxing the large fields to reduce the total size of the enum";
                span_lint_and_then(
                    cx,
                    LARGE_ENUM_VARIANT,
                    def.variants[variants_size[0].ind].span,
                    "large size difference between variants",
                    |diag| {
                        diag.span_label(
                            def.variants[variants_size[0].ind].span,
                            &format!("this variant is {} bytes", variants_size[0].size),
                        );
                        diag.span_note(
                            def.variants[variants_size[1].ind].span,
                            &format!("and the second-largest variant is {} bytes:", variants_size[1].size),
                        );

                        let fields = def.variants[variants_size[0].ind].data.fields();
                        variants_size[0].fields_size.sort_by(|a, b| (a.size.cmp(&b.size)));
                        let mut applicability = Applicability::MaybeIncorrect;
                        let sugg: Vec<(Span, String)> = variants_size[0]
                            .fields_size
                            .iter()
                            .rev()
                            .map_while(|val| {
                                if difference > self.maximum_size_difference_allowed {
                                    difference = difference.saturating_sub(val.size);
                                    Some((
                                        fields[val.ind].ty.span,
                                        format!(
                                            "Box<{}>",
                                            snippet_with_applicability(
                                                cx,
                                                fields[val.ind].ty.span,
                                                "..",
                                                &mut applicability
                                            )
                                            .into_owned()
                                        ),
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !sugg.is_empty() {
                            diag.multipart_suggestion(help_text, sugg, Applicability::MaybeIncorrect);
                            return;
                        }

                        diag.span_help(def.variants[variants_size[0].ind].span, help_text);
                    },
                );
            }
        }
    }
}
