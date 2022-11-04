use clippy_utils::diagnostics::span_lint_and_then;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefId;
use rustc_hir::{Item, ItemKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

use crate::utils::conf;

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
#[derive(Clone, Debug)]
pub struct DisallowedTypes {
    conf_disallowed: Vec<conf::DisallowedPath>,
    def_ids: FxHashMap<DefId, usize>,
    prim_tys: FxHashMap<PrimTy, usize>,
}

impl DisallowedTypes {
    pub fn new(conf_disallowed: Vec<conf::DisallowedPath>) -> Self {
        Self {
            conf_disallowed,
            def_ids: FxHashMap::default(),
            prim_tys: FxHashMap::default(),
        }
    }

    fn check_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        match res {
            Res::Def(_, did) => {
                if let Some(&index) = self.def_ids.get(did) {
                    emit(cx, &cx.tcx.def_path_str(*did), span, &self.conf_disallowed[index]);
                }
            },
            Res::PrimTy(prim) => {
                if let Some(&index) = self.prim_tys.get(prim) {
                    emit(cx, prim.name_str(), span, &self.conf_disallowed[index]);
                }
            },
            _ => {},
        }
    }
}

impl_lint_pass!(DisallowedTypes => [DISALLOWED_TYPES]);

impl<'tcx> LateLintPass<'tcx> for DisallowedTypes {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for (index, conf) in self.conf_disallowed.iter().enumerate() {
            let segs: Vec<_> = conf.path().split("::").collect();

            for res in clippy_utils::def_path_res(cx, &segs) {
                match res {
                    Res::Def(_, id) => {
                        self.def_ids.insert(id, index);
                    },
                    Res::PrimTy(ty) => {
                        self.prim_tys.insert(ty, index);
                    },
                    _ => {},
                }
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(path, UseKind::Single) = &item.kind {
            self.check_res_emit(cx, &path.res, item.span);
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

fn emit(cx: &LateContext<'_>, name: &str, span: Span, conf: &conf::DisallowedPath) {
    span_lint_and_then(
        cx,
        DISALLOWED_TYPES,
        span,
        &format!("`{name}` is not allowed according to config"),
        |diag| {
            if let Some(reason) = conf.reason() {
                diag.note(reason);
            }
        },
    );
}
