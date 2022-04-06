use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::{is_lint_allowed, meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_ast::ast::{self, FieldDef, VisibilityKind};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{self as hir, Expr, ExprKind, QPath};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_middle::ty::DefIdTree;
use rustc_semver::RustcVersion;
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

#[allow(clippy::module_name_repetitions)]
pub struct ManualNonExhaustiveStruct {
    msrv: Option<RustcVersion>,
}

impl ManualNonExhaustiveStruct {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualNonExhaustiveStruct => [MANUAL_NON_EXHAUSTIVE]);

#[allow(clippy::module_name_repetitions)]
pub struct ManualNonExhaustiveEnum {
    msrv: Option<RustcVersion>,
    constructed_enum_variants: FxHashSet<(DefId, DefId)>,
    potential_enums: Vec<(LocalDefId, LocalDefId, Span, Span)>,
}

impl ManualNonExhaustiveEnum {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
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
        if !meets_msrv(self.msrv.as_ref(), &msrvs::NON_EXHAUSTIVE) {
            return;
        }

        if let ast::ItemKind::Struct(variant_data, _) = &item.kind {
            if let ast::VariantData::Unit(..) = variant_data {
                return;
            }

            check_manual_non_exhaustive_struct(cx, item, variant_data);
        }
    }

    extract_msrv_attr!(EarlyContext);
}

fn check_manual_non_exhaustive_struct(cx: &EarlyContext<'_>, item: &ast::Item, data: &ast::VariantData) {
    fn is_private(field: &FieldDef) -> bool {
        matches!(field.vis.kind, VisibilityKind::Inherited)
    }

    fn is_non_exhaustive_marker(field: &FieldDef) -> bool {
        is_private(field) && field.ty.kind.is_unit() && field.ident.map_or(true, |n| n.as_str().starts_with('_'))
    }

    fn find_header_span(cx: &EarlyContext<'_>, item: &ast::Item, data: &ast::VariantData) -> Span {
        let delimiter = match data {
            ast::VariantData::Struct(..) => '{',
            ast::VariantData::Tuple(..) => '(',
            ast::VariantData::Unit(_) => unreachable!("`VariantData::Unit` is already handled above"),
        };

        cx.sess().source_map().span_until_char(item.span, delimiter)
    }

    let fields = data.fields();
    let private_fields = fields.iter().filter(|f| is_private(f)).count();
    let public_fields = fields.iter().filter(|f| f.vis.kind.is_pub()).count();

    if_chain! {
        if private_fields == 1 && public_fields >= 1 && public_fields == fields.len() - 1;
        if let Some(marker) = fields.iter().find(|f| is_non_exhaustive_marker(f));
        then {
            span_lint_and_then(
                cx,
                MANUAL_NON_EXHAUSTIVE,
                item.span,
                "this seems like a manual implementation of the non-exhaustive pattern",
                |diag| {
                    if_chain! {
                        if !item.attrs.iter().any(|attr| attr.has_name(sym::non_exhaustive));
                        let header_span = find_header_span(cx, item, data);
                        if let Some(snippet) = snippet_opt(cx, header_span);
                        then {
                            diag.span_suggestion(
                                header_span,
                                "add the attribute",
                                format!("#[non_exhaustive] {}", snippet),
                                Applicability::Unspecified,
                            );
                        }
                    }
                    diag.span_help(marker.span, "remove this field");
                });
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualNonExhaustiveEnum {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &msrvs::NON_EXHAUSTIVE) {
            return;
        }

        if let hir::ItemKind::Enum(def, _) = &item.kind
            && def.variants.len() > 1
        {
            let mut iter = def.variants.iter().filter_map(|v| {
                let id = cx.tcx.hir().local_def_id(v.id);
                (matches!(v.data, hir::VariantData::Unit(_))
                    && v.ident.as_str().starts_with('_')
                    && is_doc_hidden(cx.tcx.get_attrs(id.to_def_id())))
                .then(|| (id, v.span))
            });
            if let Some((id, span)) = iter.next()
                && iter.next().is_none()
            {
                self.potential_enums.push((item.def_id, id, item.span, span));
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Path(QPath::Resolved(None, p)) = &e.kind
            && let [.., name] = p.segments
            && let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = p.res
            && name.ident.as_str().starts_with('_')
            && let Some(variant_id) = cx.tcx.parent(id)
            && let Some(enum_id) = cx.tcx.parent(variant_id)
        {
            self.constructed_enum_variants.insert((enum_id, variant_id));
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for &(enum_id, _, enum_span, variant_span) in
            self.potential_enums.iter().filter(|&&(enum_id, variant_id, _, _)| {
                !self
                    .constructed_enum_variants
                    .contains(&(enum_id.to_def_id(), variant_id.to_def_id()))
                    && !is_lint_allowed(cx, MANUAL_NON_EXHAUSTIVE, cx.tcx.hir().local_def_id_to_hir_id(enum_id))
            })
        {
            span_lint_and_then(
                cx,
                MANUAL_NON_EXHAUSTIVE,
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
                                format!("#[non_exhaustive] {}", snippet),
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
