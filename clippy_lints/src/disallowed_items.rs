use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::paths::PathNS;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{AmbigArg, Expr, ExprKind, Item, ItemKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind, UseTree};
use rustc_lint::{LateContext, LateLintPass, impl_lint_pass};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured methods and functions in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// methods are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some methods are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = [
    ///     # Can use a string as the path of the disallowed method.
    ///     "std::boxed::Box::new",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::time::Instant::now" },
    ///     # When using an inline table, can add a `reason` for why the method
    ///     # is disallowed.
    ///     { path = "std::vec::Vec::leak", reason = "no leaking memory" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex::new", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex::new" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::fs::InvalidPath", reason = "use alternative instead", allow-invalid = true },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// let xs = vec![1, 2, 3, 4];
    /// xs.leak(); // Vec::leak is disallowed in the config.
    /// // The diagnostic contains the message "no leaking memory".
    ///
    /// let _now = Instant::now(); // Instant::now is disallowed in the config.
    ///
    /// let _box = Box::new(3); // Box::new is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let mut xs = Vec::new(); // Vec::new is _not_ disallowed in the config.
    /// xs.push(123); // Vec::push is _not_ disallowed in the config.
    /// ```
    #[clippy::version = "1.49.0"]
    pub DISALLOWED_METHODS,
    style,
    "use of a disallowed method call"
}

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
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::invalid::Type", reason = "use alternative instead", allow-invalid = true }
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

impl_lint_pass!(DisallowedItems => [DISALLOWED_METHODS, DISALLOWED_TYPES]);

pub struct DisallowedItems {
    methods: DefIdMap<(&'static str, &'static DisallowedPath)>,
    tys: DefIdMap<(&'static str, &'static DisallowedPath)>,
    prim_tys: FxHashMap<PrimTy, (&'static str, &'static DisallowedPath)>,
}

impl DisallowedItems {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let (methods, _) = create_disallowed_map(
            tcx,
            &conf.disallowed_methods,
            PathNS::Value,
            |def_kind| {
                matches!(
                    def_kind,
                    DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::AssocFn
                )
            },
            "function",
            false,
        );
        let (tys, prim_tys) = create_disallowed_map(
            tcx,
            &conf.disallowed_types,
            PathNS::Type,
            |def_kind| {
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
            },
            "type",
            true,
        );
        Self { methods, tys, prim_tys }
    }

    fn check_ty_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        let (path, disallowed_path) = match res {
            Res::Def(_, did) if let Some(&x) = self.tys.get(did) => x,
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

impl<'tcx> LateLintPass<'tcx> for DisallowedItems {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if self.methods.is_empty() {
            return;
        }
        if expr.span.desugaring_kind().is_some() {
            return;
        }
        let (id, span) = match &expr.kind {
            ExprKind::Path(path) if let Res::Def(_, id) = cx.qpath_res(path, expr.hir_id) => (id, expr.span),
            ExprKind::MethodCall(name, ..) if let Some(id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) => {
                (id, name.ident.span)
            },
            _ => return,
        };
        if let Some(&(path, disallowed_path)) = self.methods.get(&id) {
            span_lint_and_then(
                cx,
                DISALLOWED_METHODS,
                span,
                format!("use of a disallowed method `{path}`"),
                disallowed_path.diag_amendment(span),
            );
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(UseTree {
            prefix,
            kind: UseKind::Single(_),
        }) = &item.kind
            && let Some(res) = prefix.res.type_ns
        {
            self.check_ty_res_emit(cx, &res, item.span);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx, AmbigArg>) {
        if let TyKind::Path(path) = &ty.kind {
            self.check_ty_res_emit(cx, &cx.qpath_res(path, ty.hir_id), ty.span);
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        self.check_ty_res_emit(cx, &poly.trait_ref.path.res, poly.trait_ref.path.span);
    }
}
