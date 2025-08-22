use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::indent_of;
use clippy_utils::{is_default_equivalent, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{
    self as hir, Body, Expr, ExprKind, GenericArg, Impl, ImplItemKind, Item, ItemKind, Node, PathSegment, QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, PointerCoercion};
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, Ty, TypeckResults, VariantDef};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects manual `std::default::Default` implementations that are identical to a derived implementation.
    ///
    /// ### Why is this bad?
    /// It is less concise.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo {
    ///     bar: bool
    /// }
    ///
    /// impl Default for Foo {
    ///     fn default() -> Self {
    ///         Self {
    ///             bar: false
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// #[derive(Default)]
    /// struct Foo {
    ///     bar: bool
    /// }
    /// ```
    ///
    /// ### Known problems
    /// Derive macros [sometimes use incorrect bounds](https://github.com/rust-lang/rust/issues/26925)
    /// in generic types and the user defined `impl` may be more generalized or
    /// specialized than what derive will produce. This lint can't detect the manual `impl`
    /// has exactly equal bounds, and therefore this lint is disabled for types with
    /// generic parameters.
    #[clippy::version = "1.57.0"]
    pub DERIVABLE_IMPLS,
    complexity,
    "manual implementation of the `Default` trait which is equal to a derive"
}

pub struct DerivableImpls {
    msrv: Msrv,
}

impl DerivableImpls {
    pub fn new(conf: &'static Conf) -> Self {
        DerivableImpls { msrv: conf.msrv }
    }
}

impl_lint_pass!(DerivableImpls => [DERIVABLE_IMPLS]);

fn is_path_self(e: &Expr<'_>) -> bool {
    if let ExprKind::Path(QPath::Resolved(_, p)) = e.kind {
        matches!(p.res, Res::SelfCtor(..) | Res::Def(DefKind::Ctor(..), _))
    } else {
        false
    }
}

fn contains_trait_object(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Ref(_, ty, _) => contains_trait_object(*ty),
        ty::Adt(def, args) => def.is_box() && args[0].as_type().is_some_and(contains_trait_object),
        ty::Dynamic(..) => true,
        _ => false,
    }
}

fn determine_derive_macro(cx: &LateContext<'_>, is_const: bool) -> Option<&'static str> {
    (!is_const)
        .then_some("derive")
        .or_else(|| cx.tcx.features().enabled(sym::derive_const).then_some("derive_const"))
}

#[expect(clippy::too_many_arguments)]
fn check_struct<'tcx>(
    cx: &LateContext<'tcx>,
    item: &'tcx Item<'_>,
    self_ty: &hir::Ty<'_>,
    func_expr: &Expr<'_>,
    adt_def: AdtDef<'_>,
    ty_args: GenericArgsRef<'_>,
    typeck_results: &'tcx TypeckResults<'tcx>,
    is_const: bool,
) {
    if let TyKind::Path(QPath::Resolved(_, p)) = self_ty.kind
        && let Some(PathSegment { args, .. }) = p.segments.last()
    {
        let args = args.map(|a| a.args).unwrap_or(&[]);

        // ty_args contains the generic parameters of the type declaration, while args contains the
        // arguments used at instantiation time. If both len are not equal, it means that some
        // parameters were not provided (which means that the default values were used); in this
        // case we will not risk suggesting too broad a rewrite. We won't either if any argument
        // is a type or a const.
        if ty_args.len() != args.len() || args.iter().any(|arg| !matches!(arg, GenericArg::Lifetime(_))) {
            return;
        }
    }

    // the default() call might unsize coerce to a trait object (e.g. Box<T> to Box<dyn Trait>),
    // which would not be the same if derived (see #10158).
    // this closure checks both if the expr is equivalent to a `default()` call and does not
    // have such coercions.
    let is_default_without_adjusts = |expr| {
        is_default_equivalent(cx, expr)
            && typeck_results.expr_adjustments(expr).iter().all(|adj| {
                !matches!(adj.kind, Adjust::Pointer(PointerCoercion::Unsize)
                    if contains_trait_object(adj.target))
            })
    };

    let should_emit = match peel_blocks(func_expr).kind {
        ExprKind::Tup(fields) => fields.iter().all(is_default_without_adjusts),
        ExprKind::Call(callee, args) if is_path_self(callee) => args.iter().all(is_default_without_adjusts),
        ExprKind::Struct(_, fields, _) => fields.iter().all(|ef| is_default_without_adjusts(ef.expr)),
        _ => return,
    };

    if should_emit && let Some(derive_snippet) = determine_derive_macro(cx, is_const) {
        let struct_span = cx.tcx.def_span(adt_def.did());
        let indent_enum = indent_of(cx, struct_span).unwrap_or(0);
        let suggestions = vec![
            (item.span, String::new()), // Remove the manual implementation
            (
                struct_span.shrink_to_lo(),
                format!("#[{derive_snippet}(Default)]\n{}", " ".repeat(indent_enum)),
            ), // Add the derive attribute
        ];

        span_lint_and_then(cx, DERIVABLE_IMPLS, item.span, "this `impl` can be derived", |diag| {
            diag.multipart_suggestion(
                "replace the manual implementation with a derive attribute",
                suggestions,
                Applicability::MachineApplicable,
            );
        });
    }
}

fn extract_enum_variant<'tcx>(
    cx: &LateContext<'tcx>,
    func_expr: &'tcx Expr<'tcx>,
    adt_def: AdtDef<'tcx>,
) -> Option<&'tcx VariantDef> {
    match &peel_blocks(func_expr).kind {
        ExprKind::Path(QPath::Resolved(None, p))
            if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = p.res
                && let variant_id = cx.tcx.parent(id)
                && let Some(variant_def) = adt_def.variants().iter().find(|v| v.def_id == variant_id) =>
        {
            Some(variant_def)
        },
        ExprKind::Path(QPath::TypeRelative(ty, segment))
            if let TyKind::Path(QPath::Resolved(None, p)) = &ty.kind
                && let Res::SelfTyAlias {
                    is_trait_impl: true, ..
                } = p.res
                && let variant_ident = segment.ident
                && let Some(variant_def) = adt_def.variants().iter().find(|v| v.ident(cx.tcx) == variant_ident) =>
        {
            Some(variant_def)
        },
        _ => None,
    }
}

fn check_enum<'tcx>(
    cx: &LateContext<'tcx>,
    item: &'tcx Item<'tcx>,
    func_expr: &'tcx Expr<'tcx>,
    adt_def: AdtDef<'tcx>,
    is_const: bool,
) {
    if let Some(variant_def) = extract_enum_variant(cx, func_expr, adt_def)
        && variant_def.fields.is_empty()
        && !variant_def.is_field_list_non_exhaustive()
    {
        let enum_span = cx.tcx.def_span(adt_def.did());
        let indent_enum = indent_of(cx, enum_span).unwrap_or(0);
        let variant_span = cx.tcx.def_span(variant_def.def_id);
        let indent_variant = indent_of(cx, variant_span).unwrap_or(0);

        let Some(derive_snippet) = determine_derive_macro(cx, is_const) else {
            return;
        };

        let suggestions = vec![
            (item.span, String::new()), // Remove the manual implementation
            (
                enum_span.shrink_to_lo(),
                format!("#[{derive_snippet}(Default)]\n{}", " ".repeat(indent_enum)),
            ), // Add the derive attribute
            (
                variant_span.shrink_to_lo(),
                format!("#[default]\n{}", " ".repeat(indent_variant)),
            ), // Mark the default variant
        ];

        span_lint_and_then(cx, DERIVABLE_IMPLS, item.span, "this `impl` can be derived", |diag| {
            diag.multipart_suggestion(
                "replace the manual implementation with a derive attribute and mark the default variant",
                suggestions,
                Applicability::MachineApplicable,
            );
        });
    }
}

impl<'tcx> LateLintPass<'tcx> for DerivableImpls {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(of_trait),
            items: [child],
            self_ty,
            ..
        }) = item.kind
            && !cx.tcx.is_automatically_derived(item.owner_id.to_def_id())
            && !item.span.from_expansion()
            && let Some(def_id) = of_trait.trait_ref.trait_def_id()
            && cx.tcx.is_diagnostic_item(sym::Default, def_id)
            && let impl_item_hir = child.hir_id()
            && let Node::ImplItem(impl_item) = cx.tcx.hir_node(impl_item_hir)
            && let ImplItemKind::Fn(_, b) = &impl_item.kind
            && let Body { value: func_expr, .. } = cx.tcx.hir_body(*b)
            && let &ty::Adt(adt_def, args) = cx.tcx.type_of(item.owner_id).instantiate_identity().kind()
            && let attrs = cx.tcx.hir_attrs(item.hir_id())
            && !attrs.iter().any(|attr| attr.doc_str().is_some())
            && cx.tcx.hir_attrs(impl_item_hir).is_empty()
        {
            let is_const = of_trait.constness == hir::Constness::Const;
            if adt_def.is_struct() {
                check_struct(
                    cx,
                    item,
                    self_ty,
                    func_expr,
                    adt_def,
                    args,
                    cx.tcx.typeck_body(*b),
                    is_const,
                );
            } else if adt_def.is_enum() && self.msrv.meets(cx, msrvs::DEFAULT_ENUM_ATTRIBUTE) {
                check_enum(cx, item, func_expr, adt_def, is_const);
            }
        }
    }
}
