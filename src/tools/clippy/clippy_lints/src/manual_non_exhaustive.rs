use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::is_doc_hidden;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_indent;
use itertools::Itertools;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{Expr, ExprKind, Item, ItemKind, QPath, TyKind, VariantData, find_attr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of the non-exhaustive pattern.
    ///
    /// ### Why is this bad?
    /// Using the #[non_exhaustive] attribute expresses better the intent
    /// and allows possible optimizations when applied to enums.
    ///
    /// ### Example
    /// ```no_run
    /// struct S {
    ///     pub a: i32,
    ///     pub b: i32,
    ///     _c: (),
    /// }
    ///
    /// enum E {
    ///     A,
    ///     B,
    ///     #[doc(hidden)]
    ///     _C,
    /// }
    ///
    /// struct T(pub i32, pub i32, ());
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[non_exhaustive]
    /// struct S {
    ///     pub a: i32,
    ///     pub b: i32,
    /// }
    ///
    /// #[non_exhaustive]
    /// enum E {
    ///     A,
    ///     B,
    /// }
    ///
    /// #[non_exhaustive]
    /// struct T(pub i32, pub i32);
    /// ```
    #[clippy::version = "1.45.0"]
    pub MANUAL_NON_EXHAUSTIVE,
    style,
    "manual implementations of the non-exhaustive pattern can be simplified using #[non_exhaustive]"
}

pub struct ManualNonExhaustive {
    msrv: Msrv,
    constructed_enum_variants: FxHashSet<LocalDefId>,
    potential_enums: Vec<(LocalDefId, LocalDefId, Span, Span)>,
}

impl ManualNonExhaustive {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            constructed_enum_variants: FxHashSet::default(),
            potential_enums: Vec::new(),
        }
    }
}

impl_lint_pass!(ManualNonExhaustive => [MANUAL_NON_EXHAUSTIVE]);

impl<'tcx> LateLintPass<'tcx> for ManualNonExhaustive {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if !cx.effective_visibilities.is_exported(item.owner_id.def_id) || !self.msrv.meets(cx, msrvs::NON_EXHAUSTIVE) {
            return;
        }

        match item.kind {
            ItemKind::Enum(_, _, def) if def.variants.len() > 1 => {
                let iter = def.variants.iter().filter_map(|v| {
                    (matches!(v.data, VariantData::Unit(_, _)) && is_doc_hidden(cx.tcx.hir_attrs(v.hir_id)))
                        .then_some((v.def_id, v.span))
                });
                if let Ok((id, span)) = iter.exactly_one()
                    && !find_attr!(cx.tcx.hir_attrs(item.hir_id()), AttributeKind::NonExhaustive(..))
                {
                    self.potential_enums.push((item.owner_id.def_id, id, item.span, span));
                }
            },
            ItemKind::Struct(_, _, variant_data) => {
                let fields = variant_data.fields();
                let private_fields = fields
                    .iter()
                    .filter(|field| !cx.effective_visibilities.is_exported(field.def_id));
                if fields.len() > 1
                    && let Ok(field) = private_fields.exactly_one()
                    && let TyKind::Tup([]) = field.ty.kind
                {
                    span_lint_and_then(
                        cx,
                        MANUAL_NON_EXHAUSTIVE,
                        item.span,
                        "this seems like a manual implementation of the non-exhaustive pattern",
                        |diag| {
                            if let Some(non_exhaustive_span) =
                                find_attr!(cx.tcx.hir_attrs(item.hir_id()), AttributeKind::NonExhaustive(span) => *span)
                            {
                                diag.span_note(non_exhaustive_span, "the struct is already non-exhaustive");
                            } else {
                                let indent = snippet_indent(cx, item.span).unwrap_or_default();
                                diag.span_suggestion_verbose(
                                    item.span.shrink_to_lo(),
                                    "use the `#[non_exhaustive]` attribute instead",
                                    format!("#[non_exhaustive]\n{indent}"),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            diag.span_help(field.span, "remove this field");
                        },
                    );
                }
            },
            _ => {},
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Path(QPath::Resolved(None, p)) = &e.kind
            && let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), ctor_id) = p.res
            && let Some(local_ctor) = ctor_id.as_local()
        {
            let variant_id = cx.tcx.local_parent(local_ctor);
            self.constructed_enum_variants.insert(variant_id);
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for &(enum_id, _, enum_span, variant_span) in self
            .potential_enums
            .iter()
            .filter(|(_, variant_id, _, _)| !self.constructed_enum_variants.contains(variant_id))
        {
            let hir_id = cx.tcx.local_def_id_to_hir_id(enum_id);
            span_lint_hir_and_then(
                cx,
                MANUAL_NON_EXHAUSTIVE,
                hir_id,
                enum_span,
                "this seems like a manual implementation of the non-exhaustive pattern",
                |diag| {
                    let indent = snippet_indent(cx, enum_span).unwrap_or_default();
                    diag.span_suggestion_verbose(
                        enum_span.shrink_to_lo(),
                        "use the `#[non_exhaustive]` attribute instead",
                        format!("#[non_exhaustive]\n{indent}"),
                        Applicability::MaybeIncorrect,
                    );
                    diag.span_help(variant_span, "remove this variant");
                },
            );
        }
    }
}
