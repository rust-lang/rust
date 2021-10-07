use clippy_utils::diagnostics::span_lint;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{
    def::Res, def_id::DefId, Item, ItemKind, PolyTraitRef, PrimTy, TraitBoundModifier, Ty, TyKind, UseKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};

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
    /// disallowed-types = ["std::collections::BTreeMap"]
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
    disallowed: FxHashSet<Vec<Symbol>>,
    def_ids: FxHashSet<DefId>,
    prim_tys: FxHashSet<PrimTy>,
}

impl DisallowedType {
    pub fn new(disallowed: &FxHashSet<String>) -> Self {
        Self {
            disallowed: disallowed
                .iter()
                .map(|s| s.split("::").map(Symbol::intern).collect::<Vec<_>>())
                .collect(),
            def_ids: FxHashSet::default(),
            prim_tys: FxHashSet::default(),
        }
    }

    fn check_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        match res {
            Res::Def(_, did) => {
                if self.def_ids.contains(did) {
                    emit(cx, &cx.tcx.def_path_str(*did), span);
                }
            },
            Res::PrimTy(prim) => {
                if self.prim_tys.contains(prim) {
                    emit(cx, prim.name_str(), span);
                }
            },
            _ => {},
        }
    }
}

impl_lint_pass!(DisallowedType => [DISALLOWED_TYPE]);

impl<'tcx> LateLintPass<'tcx> for DisallowedType {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        for path in &self.disallowed {
            let segs = path.iter().map(ToString::to_string).collect::<Vec<_>>();
            match clippy_utils::path_to_res(cx, &segs.iter().map(String::as_str).collect::<Vec<_>>()) {
                Res::Def(_, id) => {
                    self.def_ids.insert(id);
                },
                Res::PrimTy(ty) => {
                    self.prim_tys.insert(ty);
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

fn emit(cx: &LateContext<'_>, name: &str, span: Span) {
    span_lint(
        cx,
        DISALLOWED_TYPE,
        span,
        &format!("`{}` is not allowed according to config", name),
    );
}
