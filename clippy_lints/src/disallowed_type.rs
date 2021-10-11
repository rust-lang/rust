use clippy_utils::diagnostics::span_lint_and_then;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{
    def::Res, def_id::DefId, Item, ItemKind, PolyTraitRef, PrimTy, TraitBoundModifier, Ty, TyKind, UseKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

use crate::utils::conf;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured types in clippy.toml.
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
    pub DISALLOWED_TYPE,
    nursery,
    "use of a disallowed type"
}
#[derive(Clone, Debug)]
pub struct DisallowedType {
    conf_disallowed: Vec<conf::DisallowedType>,
    def_ids: FxHashMap<DefId, Option<String>>,
    prim_tys: FxHashMap<PrimTy, Option<String>>,
}

impl DisallowedType {
    pub fn new(conf_disallowed: Vec<conf::DisallowedType>) -> Self {
        Self {
            conf_disallowed,
            def_ids: FxHashMap::default(),
            prim_tys: FxHashMap::default(),
        }
    }

    fn check_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        match res {
            Res::Def(_, did) => {
                if let Some(reason) = self.def_ids.get(did) {
                    emit(cx, &cx.tcx.def_path_str(*did), span, reason.as_deref());
                }
            },
            Res::PrimTy(prim) => {
                if let Some(reason) = self.prim_tys.get(prim) {
                    emit(cx, prim.name_str(), span, reason.as_deref());
                }
            },
            _ => {},
        }
    }
}

impl_lint_pass!(DisallowedType => [DISALLOWED_TYPE]);

impl<'tcx> LateLintPass<'tcx> for DisallowedType {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for conf in &self.conf_disallowed {
            let (path, reason) = match conf {
                conf::DisallowedType::Simple(path) => (path, None),
                conf::DisallowedType::WithReason { path, reason } => (
                    path,
                    reason.as_ref().map(|reason| format!("{} (from clippy.toml)", reason)),
                ),
            };
            let segs: Vec<_> = path.split("::").collect();
            match clippy_utils::path_to_res(cx, &segs) {
                Res::Def(_, id) => {
                    self.def_ids.insert(id, reason);
                },
                Res::PrimTy(ty) => {
                    self.prim_tys.insert(ty, reason);
                },
                _ => {},
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

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>, _: TraitBoundModifier) {
        self.check_res_emit(cx, &poly.trait_ref.path.res, poly.trait_ref.path.span);
    }
}

fn emit(cx: &LateContext<'_>, name: &str, span: Span, reason: Option<&str>) {
    span_lint_and_then(
        cx,
        DISALLOWED_TYPE,
        span,
        &format!("`{}` is not allowed according to config", name),
        |diag| {
            if let Some(reason) = reason {
                diag.note(reason);
            }
        },
    );
}
