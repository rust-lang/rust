use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Item, ItemKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind};
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
    def_ids: DefIdMap<(&'static str, Option<&'static str>)>,
    prim_tys: FxHashMap<PrimTy, (&'static str, Option<&'static str>)>,
}

impl DisallowedTypes {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let mut def_ids = DefIdMap::default();
        let mut prim_tys = FxHashMap::default();
        for x in &conf.disallowed_types {
            let path: Vec<_> = x.path().split("::").collect::<Vec<_>>();
            let reason = x.reason();
            for res in clippy_utils::def_path_res(tcx, &path) {
                match res {
                    Res::Def(_, id) => {
                        def_ids.insert(id, (x.path(), reason));
                    },
                    Res::PrimTy(ty) => {
                        prim_tys.insert(ty, (x.path(), reason));
                    },
                    _ => {},
                }
            }
        }
        Self { def_ids, prim_tys }
    }

    fn check_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        let (path, reason) = match res {
            Res::Def(_, did) if let Some(&x) = self.def_ids.get(did) => x,
            Res::PrimTy(prim) if let Some(&x) = self.prim_tys.get(prim) => x,
            _ => return,
        };
        span_lint_and_then(
            cx,
            DISALLOWED_TYPES,
            span,
            format!("use of a disallowed type `{path}`"),
            |diag| {
                if let Some(reason) = reason {
                    diag.note(reason);
                }
            },
        );
    }
}

impl_lint_pass!(DisallowedTypes => [DISALLOWED_TYPES]);

impl<'tcx> LateLintPass<'tcx> for DisallowedTypes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(path, UseKind::Single) = &item.kind {
            for res in &path.res {
                self.check_res_emit(cx, res, item.span);
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx>) {
        if let TyKind::Path(path) = &ty.kind {
            self.check_res_emit(cx, &cx.qpath_res(path, ty.hir_id), ty.span);
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        self.check_res_emit(cx, &poly.trait_ref.path.res, poly.trait_ref.path.span);
    }
}
