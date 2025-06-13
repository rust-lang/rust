use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{AdtVariantInfo, approx_ty_size, is_copy};
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

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
    /// ```no_run
    /// enum Test {
    ///     A(i32),
    ///     B([i32; 8000]),
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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

pub struct LargeEnumVariant {
    maximum_size_difference_allowed: u64,
}

impl LargeEnumVariant {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            maximum_size_difference_allowed: conf.enum_variant_size_threshold,
        }
    }
}

impl_lint_pass!(LargeEnumVariant => [LARGE_ENUM_VARIANT]);

impl<'tcx> LateLintPass<'tcx> for LargeEnumVariant {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &Item<'tcx>) {
        if let ItemKind::Enum(ident, _, ref def) = item.kind
            && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
            && let ty::Adt(adt, subst) = ty.kind()
            && adt.variants().len() > 1
            && !item.span.in_external_macro(cx.tcx.sess.source_map())
        {
            let variants_size = AdtVariantInfo::new(cx, *adt, subst);

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
                            if variants_size[1].fields_size.is_empty() {
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
                                ident.span,
                                "boxing a variant would require the type no longer be `Copy`",
                            );
                        } else {
                            let sugg: Vec<(Span, String)> = variants_size[0]
                                .fields_size
                                .iter()
                                .rev()
                                .map_while(|&(ind, size)| {
                                    if difference > self.maximum_size_difference_allowed {
                                        difference = difference.saturating_sub(size);
                                        Some((
                                            fields[ind].ty.span,
                                            format!(
                                                "Box<{}>",
                                                snippet_with_applicability(
                                                    cx,
                                                    fields[ind].ty.span,
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
    if let ty::Adt(_def, args) = ty.kind()
        && args.types().next().is_some()
        && let Some(copy_trait) = cx.tcx.lang_items().copy_trait()
    {
        return cx.tcx.non_blanket_impls_for_ty(copy_trait, ty).next().is_some();
    }
    false
}
