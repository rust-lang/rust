use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::indent_of;
use clippy_utils::{is_default_equivalent, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir,
    def::{CtorKind, CtorOf, DefKind, Res},
    Body, Expr, ExprKind, GenericArg, Impl, ImplItemKind, Item, ItemKind, Node, PathSegment, QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, PointerCast};
use rustc_middle::ty::{self, Adt, AdtDef, SubstsRef, Ty, TypeckResults};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects manual `std::default::Default` implementations that are identical to a derived implementation.
    ///
    /// ### Why is this bad?
    /// It is less concise.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        DerivableImpls { msrv }
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
        ty::Adt(def, substs) => def.is_box() && substs[0].as_type().map_or(false, contains_trait_object),
        ty::Dynamic(..) => true,
        _ => false,
    }
}

fn check_struct<'tcx>(
    cx: &LateContext<'tcx>,
    item: &'tcx Item<'_>,
    self_ty: &hir::Ty<'_>,
    func_expr: &Expr<'_>,
    adt_def: AdtDef<'_>,
    substs: SubstsRef<'_>,
    typeck_results: &'tcx TypeckResults<'tcx>,
) {
    if let TyKind::Path(QPath::Resolved(_, p)) = self_ty.kind {
        if let Some(PathSegment { args, .. }) = p.segments.last() {
            let args = args.map(|a| a.args).unwrap_or(&[]);

            // substs contains the generic parameters of the type declaration, while args contains the arguments
            // used at instantiation time. If both len are not equal, it means that some parameters were not
            // provided (which means that the default values were used); in this case we will not risk
            // suggesting too broad a rewrite. We won't either if any argument is a type or a const.
            if substs.len() != args.len() || args.iter().any(|arg| !matches!(arg, GenericArg::Lifetime(_))) {
                return;
            }
        }
    }

    // the default() call might unsize coerce to a trait object (e.g. Box<T> to Box<dyn Trait>),
    // which would not be the same if derived (see #10158).
    // this closure checks both if the expr is equivalent to a `default()` call and does not
    // have such coercions.
    let is_default_without_adjusts = |expr| {
        is_default_equivalent(cx, expr)
            && typeck_results.expr_adjustments(expr).iter().all(|adj| {
                !matches!(adj.kind, Adjust::Pointer(PointerCast::Unsize)
                    if contains_trait_object(adj.target))
            })
    };

    let should_emit = match peel_blocks(func_expr).kind {
        ExprKind::Tup(fields) => fields.iter().all(is_default_without_adjusts),
        ExprKind::Call(callee, args) if is_path_self(callee) => args.iter().all(is_default_without_adjusts),
        ExprKind::Struct(_, fields, _) => fields.iter().all(|ef| is_default_without_adjusts(ef.expr)),
        _ => false,
    };

    if should_emit {
        let struct_span = cx.tcx.def_span(adt_def.did());
        span_lint_and_then(cx, DERIVABLE_IMPLS, item.span, "this `impl` can be derived", |diag| {
            diag.span_suggestion_hidden(
                item.span,
                "remove the manual implementation...",
                String::new(),
                Applicability::MachineApplicable,
            );
            diag.span_suggestion(
                struct_span.shrink_to_lo(),
                "...and instead derive it",
                "#[derive(Default)]\n".to_string(),
                Applicability::MachineApplicable,
            );
        });
    }
}

fn check_enum<'tcx>(cx: &LateContext<'tcx>, item: &'tcx Item<'_>, func_expr: &Expr<'_>, adt_def: AdtDef<'_>) {
    if_chain! {
        if let ExprKind::Path(QPath::Resolved(None, p)) = &peel_blocks(func_expr).kind;
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = p.res;
        if let variant_id = cx.tcx.parent(id);
        if let Some(variant_def) = adt_def.variants().iter().find(|v| v.def_id == variant_id);
        if variant_def.fields.is_empty();
        if !variant_def.is_field_list_non_exhaustive();

        then {
            let enum_span = cx.tcx.def_span(adt_def.did());
            let indent_enum = indent_of(cx, enum_span).unwrap_or(0);
            let variant_span = cx.tcx.def_span(variant_def.def_id);
            let indent_variant = indent_of(cx, variant_span).unwrap_or(0);
            span_lint_and_then(
                cx,
                DERIVABLE_IMPLS,
                item.span,
                "this `impl` can be derived",
                |diag| {
                    diag.span_suggestion_hidden(
                        item.span,
                        "remove the manual implementation...",
                        String::new(),
                        Applicability::MachineApplicable
                    );
                    diag.span_suggestion(
                        enum_span.shrink_to_lo(),
                        "...and instead derive it...",
                        format!(
                            "#[derive(Default)]\n{indent}",
                            indent = " ".repeat(indent_enum),
                        ),
                        Applicability::MachineApplicable
                    );
                    diag.span_suggestion(
                        variant_span.shrink_to_lo(),
                        "...and mark the default variant",
                        format!(
                            "#[default]\n{indent}",
                            indent = " ".repeat(indent_variant),
                        ),
                        Applicability::MachineApplicable
                    );
                }
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for DerivableImpls {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if_chain! {
            if let ItemKind::Impl(Impl {
                of_trait: Some(ref trait_ref),
                items: [child],
                self_ty,
                ..
            }) = item.kind;
            if !cx.tcx.has_attr(item.owner_id, sym::automatically_derived);
            if !item.span.from_expansion();
            if let Some(def_id) = trait_ref.trait_def_id();
            if cx.tcx.is_diagnostic_item(sym::Default, def_id);
            if let impl_item_hir = child.id.hir_id();
            if let Some(Node::ImplItem(impl_item)) = cx.tcx.hir().find(impl_item_hir);
            if let ImplItemKind::Fn(_, b) = &impl_item.kind;
            if let Body { value: func_expr, .. } = cx.tcx.hir().body(*b);
            if let &Adt(adt_def, substs) = cx.tcx.type_of(item.owner_id).subst_identity().kind();
            if let attrs = cx.tcx.hir().attrs(item.hir_id());
            if !attrs.iter().any(|attr| attr.doc_str().is_some());
            if let child_attrs = cx.tcx.hir().attrs(impl_item_hir);
            if !child_attrs.iter().any(|attr| attr.doc_str().is_some());

            then {
                if adt_def.is_struct() {
                    check_struct(cx, item, self_ty, func_expr, adt_def, substs, cx.tcx.typeck_body(*b));
                } else if adt_def.is_enum() && self.msrv.meets(msrvs::DEFAULT_ENUM_ATTRIBUTE) {
                    check_enum(cx, item, func_expr, adt_def);
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
