use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::is_doc_hidden;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{self, VisibilityKind};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{self as hir, Expr, ExprKind, QPath};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of the non-exhaustive pattern.
    ///
    /// ### Why is this bad?
    /// Using the #[non_exhaustive] attribute expresses better the intent
    /// and allows possible optimizations when applied to enums.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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

#[expect(clippy::module_name_repetitions)]
pub struct ManualNonExhaustiveStruct {
    msrv: Msrv,
}

impl ManualNonExhaustiveStruct {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualNonExhaustiveStruct => [MANUAL_NON_EXHAUSTIVE]);

#[expect(clippy::module_name_repetitions)]
pub struct ManualNonExhaustiveEnum {
    msrv: Msrv,
    constructed_enum_variants: FxHashSet<(DefId, DefId)>,
    potential_enums: Vec<(LocalDefId, LocalDefId, Span, Span)>,
}

impl ManualNonExhaustiveEnum {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self {
            msrv,
            constructed_enum_variants: FxHashSet::default(),
            potential_enums: Vec::new(),
        }
    }
}

impl_lint_pass!(ManualNonExhaustiveEnum => [MANUAL_NON_EXHAUSTIVE]);

impl EarlyLintPass for ManualNonExhaustiveStruct {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if !self.msrv.meets(msrvs::NON_EXHAUSTIVE) {
            return;
        }

        if let ast::ItemKind::Struct(variant_data, _) = &item.kind {
            let (fields, delimiter) = match variant_data {
                ast::VariantData::Struct(fields, _) => (&**fields, '{'),
                ast::VariantData::Tuple(fields, _) => (&**fields, '('),
                ast::VariantData::Unit(_) => return,
            };
            if fields.len() <= 1 {
                return;
            }
            let mut iter = fields.iter().filter_map(|f| match f.vis.kind {
                VisibilityKind::Public => None,
                VisibilityKind::Inherited => Some(Ok(f)),
                VisibilityKind::Restricted { .. } => Some(Err(())),
            });
            if let Some(Ok(field)) = iter.next()
                && iter.next().is_none()
                && field.ty.kind.is_unit()
                && field.ident.map_or(true, |name| name.as_str().starts_with('_'))
            {
                span_lint_and_then(
                    cx,
                    MANUAL_NON_EXHAUSTIVE,
                    item.span,
                    "this seems like a manual implementation of the non-exhaustive pattern",
                    |diag| {
                        if !item.attrs.iter().any(|attr| attr.has_name(sym::non_exhaustive))
                            && let header_span = cx.sess().source_map().span_until_char(item.span, delimiter)
                            && let Some(snippet) = snippet_opt(cx, header_span)
                        {
                            diag.span_suggestion(
                                header_span,
                                "add the attribute",
                                format!("#[non_exhaustive] {snippet}"),
                                Applicability::Unspecified,
                            );
                        }
                        diag.span_help(field.span, "remove this field");
                    }
                );
            }
        }
    }

    extract_msrv_attr!(EarlyContext);
}

impl<'tcx> LateLintPass<'tcx> for ManualNonExhaustiveEnum {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if !self.msrv.meets(msrvs::NON_EXHAUSTIVE) {
            return;
        }

        if let hir::ItemKind::Enum(def, _) = &item.kind
            && def.variants.len() > 1
        {
            let mut iter = def.variants.iter().filter_map(|v| {
                (matches!(v.data, hir::VariantData::Unit(_, _))
                    && v.ident.as_str().starts_with('_')
                    && is_doc_hidden(cx.tcx.hir().attrs(v.hir_id)))
                .then_some((v.def_id, v.span))
            });
            if let Some((id, span)) = iter.next()
                && iter.next().is_none()
            {
                self.potential_enums.push((item.owner_id.def_id, id, item.span, span));
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Path(QPath::Resolved(None, p)) = &e.kind
            && let [.., name] = p.segments
            && let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = p.res
            && name.ident.as_str().starts_with('_')
        {
            let variant_id = cx.tcx.parent(id);
            let enum_id = cx.tcx.parent(variant_id);

            self.constructed_enum_variants.insert((enum_id, variant_id));
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for &(enum_id, _, enum_span, variant_span) in
            self.potential_enums.iter().filter(|&&(enum_id, variant_id, _, _)| {
                !self
                    .constructed_enum_variants
                    .contains(&(enum_id.to_def_id(), variant_id.to_def_id()))
            })
        {
            let hir_id = cx.tcx.hir().local_def_id_to_hir_id(enum_id);
            span_lint_hir_and_then(
                cx,
                MANUAL_NON_EXHAUSTIVE,
                hir_id,
                enum_span,
                "this seems like a manual implementation of the non-exhaustive pattern",
                |diag| {
                    if !cx.tcx.adt_def(enum_id).is_variant_list_non_exhaustive()
                        && let header_span = cx.sess().source_map().span_until_char(enum_span, '{')
                        && let Some(snippet) = snippet_opt(cx, header_span)
                    {
                            diag.span_suggestion(
                                header_span,
                                "add the attribute",
                                format!("#[non_exhaustive] {snippet}"),
                                Applicability::Unspecified,
                            );
                    }
                    diag.span_help(variant_span, "remove this variant");
                },
            );
        }
    }

    extract_msrv_attr!(LateContext);
}
