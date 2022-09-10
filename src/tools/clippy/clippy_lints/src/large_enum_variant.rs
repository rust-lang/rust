//! lint when there is a large size difference between variants on an enum

use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{diagnostics::span_lint_and_then, ty::approx_ty_size, ty::is_copy};
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{Adt, AdtDef, GenericArg, List, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for large size differences between variants on
    /// `enum`s.
    ///
    /// ### Why is this bad?
    /// Enum size is bounded by the largest variant. Having one
    /// large variant can penalize the memory layout of that enum.
    ///
    /// ### Known problems
    /// This lint obviously cannot take the distribution of
    /// variants in your running program into account. It is possible that the
    /// smaller variants make up less than 1% of all instances, in which case
    /// the overhead is negligible and the boxing is counter-productive. Always
    /// measure the change this lint suggests.
    ///
    /// For types that implement `Copy`, the suggestion to `Box` a variant's
    /// data would require removing the trait impl. The types can of course
    /// still be `Clone`, but that is worse ergonomically. Depending on the
    /// use case it may be possible to store the large data in an auxiliary
    /// structure (e.g. Arena or ECS).
    ///
    /// The lint will ignore the impact of generic types to the type layout by
    /// assuming every type parameter is zero-sized. Depending on your use case,
    /// this may lead to a false positive.
    ///
    /// ### Example
    /// ```rust
    /// enum Test {
    ///     A(i32),
    ///     B([i32; 8000]),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// // Possibly better
    /// enum Test2 {
    ///     A(i32),
    ///     B(Box<[i32; 8000]>),
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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

fn variants_size<'tcx>(
    cx: &LateContext<'tcx>,
    adt: AdtDef<'tcx>,
    subst: &'tcx List<GenericArg<'tcx>>,
) -> Vec<VariantInfo> {
    let mut variants_size = adt
        .variants()
        .iter()
        .enumerate()
        .map(|(i, variant)| {
            let mut fields_size = variant
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| FieldInfo {
                    ind: i,
                    size: approx_ty_size(cx, f.ty(cx.tcx, subst)),
                })
                .collect::<Vec<_>>();
            fields_size.sort_by(|a, b| (a.size.cmp(&b.size)));

            VariantInfo {
                ind: i,
                size: fields_size.iter().map(|info| info.size).sum(),
                fields_size,
            }
        })
        .collect::<Vec<_>>();
    variants_size.sort_by(|a, b| (b.size.cmp(&a.size)));
    variants_size
}

impl_lint_pass!(LargeEnumVariant => [LARGE_ENUM_VARIANT]);

impl<'tcx> LateLintPass<'tcx> for LargeEnumVariant {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &Item<'tcx>) {
        if in_external_macro(cx.tcx.sess, item.span) {
            return;
        }
        if let ItemKind::Enum(ref def, _) = item.kind {
            let ty = cx.tcx.type_of(item.def_id);
            let (adt, subst) = match ty.kind() {
                Adt(adt, subst) => (adt, subst),
                _ => panic!("already checked whether this is an enum"),
            };
            if adt.variants().len() <= 1 {
                return;
            }
            let variants_size = variants_size(cx, *adt, subst);

            let mut difference = variants_size[0].size - variants_size[1].size;
            if difference > self.maximum_size_difference_allowed {
                let help_text = "consider boxing the large fields to reduce the total size of the enum";
                span_lint_and_then(
                    cx,
                    LARGE_ENUM_VARIANT,
                    item.span,
                    "large size difference between variants",
                    |diag| {
                        diag.span_label(
                            item.span,
                            format!("the entire enum is at least {} bytes", approx_ty_size(cx, ty)),
                        );
                        diag.span_label(
                            def.variants[variants_size[0].ind].span,
                            format!("the largest variant contains at least {} bytes", variants_size[0].size),
                        );
                        diag.span_label(
                            def.variants[variants_size[1].ind].span,
                            &if variants_size[1].fields_size.is_empty() {
                                "the second-largest variant carries no data at all".to_owned()
                            } else {
                                format!(
                                    "the second-largest variant contains at least {} bytes",
                                    variants_size[1].size
                                )
                            },
                        );

                        let fields = def.variants[variants_size[0].ind].data.fields();
                        let mut applicability = Applicability::MaybeIncorrect;
                        if is_copy(cx, ty) || maybe_copy(cx, ty) {
                            diag.span_note(
                                item.ident.span,
                                "boxing a variant would require the type no longer be `Copy`",
                            );
                        } else {
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
                        }
                        diag.span_help(def.variants[variants_size[0].ind].span, help_text);
                    },
                );
            }
        }
    }
}

fn maybe_copy<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    if let Adt(_def, substs) = ty.kind()
        && substs.types().next().is_some()
        && let Some(copy_trait) = cx.tcx.lang_items().copy_trait()
    {
        return cx.tcx.non_blanket_impls_for_ty(copy_trait, ty).next().is_some();
    }
    false
}
