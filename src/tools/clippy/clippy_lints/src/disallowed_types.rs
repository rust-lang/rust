use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::paths::PathNS;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{AmbigArg, Item, ItemKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured types in clippy.toml.
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// types are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some types are undesirable in certain contexts.
    ///
    /// ### Example:
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-types = [
    ///     # Can use a string as the path of the disallowed type.
    ///     "std::collections::BTreeMap",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::net::TcpListener" },
    ///     # When using an inline table, can add a `reason` for why the type
    ///     # is disallowed.
    ///     { path = "std::net::Ipv4Addr", reason = "no IPv4 allowed" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex" },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// use std::collections::BTreeMap;
    /// // or its use
    /// let x = std::collections::BTreeMap::new();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // A similar type that is allowed by the config
    /// use std::collections::HashMap;
    /// ```
    #[clippy::version = "1.55.0"]
    pub DISALLOWED_TYPES,
    style,
    "use of disallowed types"
}

pub struct DisallowedTypes {
    def_ids: DefIdMap<(&'static str, &'static DisallowedPath)>,
    prim_tys: FxHashMap<PrimTy, (&'static str, &'static DisallowedPath)>,
}

impl DisallowedTypes {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let (def_ids, prim_tys) = create_disallowed_map(
            tcx,
            &conf.disallowed_types,
            PathNS::Type,
            def_kind_predicate,
            "type",
            true,
        );
        Self { def_ids, prim_tys }
    }

    fn check_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        let (path, disallowed_path) = match res {
            Res::Def(_, did) if let Some(&x) = self.def_ids.get(did) => x,
            Res::PrimTy(prim) if let Some(&x) = self.prim_tys.get(prim) => x,
            _ => return,
        };
        span_lint_and_then(
            cx,
            DISALLOWED_TYPES,
            span,
            format!("use of a disallowed type `{path}`"),
            disallowed_path.diag_amendment(span),
        );
    }
}

pub fn def_kind_predicate(def_kind: DefKind) -> bool {
    matches!(
        def_kind,
        DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Trait
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::AssocTy
    )
}

impl_lint_pass!(DisallowedTypes => [DISALLOWED_TYPES]);

impl<'tcx> LateLintPass<'tcx> for DisallowedTypes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(path, UseKind::Single(_)) = &item.kind
            && let Some(res) = path.res.type_ns
        {
            self.check_res_emit(cx, &res, item.span);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx, AmbigArg>) {
        if let TyKind::Path(path) = &ty.kind {
            self.check_res_emit(cx, &cx.qpath_res(path, ty.hir_id), ty.span);
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        self.check_res_emit(cx, &poly.trait_ref.path.res, poly.trait_ref.path.span);
    }
}
