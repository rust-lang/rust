use clippy_utils::diagnostics::span_lint;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{
    def::Res, def_id::DefId, Crate, Item, ItemKind, PolyTraitRef, TraitBoundModifier, Ty, TyKind, UseKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// **What it does:** Denies the configured types in clippy.toml.
    ///
    /// **Why is this bad?** Some types are undesirable in certain contexts.
    ///
    /// **Known problems:** None.
    ///
    /// N.B. There is no way to ban primitive types.
    ///
    /// **Example:**
    ///
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = ["std::collections::BTreeMap"]
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
    def_ids: FxHashSet<(DefId, Vec<Symbol>)>,
}

impl DisallowedType {
    pub fn new(disallowed: &FxHashSet<String>) -> Self {
        Self {
            disallowed: disallowed
                .iter()
                .map(|s| s.split("::").map(|seg| Symbol::intern(seg)).collect::<Vec<_>>())
                .collect(),
            def_ids: FxHashSet::default(),
        }
    }
}

impl_lint_pass!(DisallowedType => [DISALLOWED_TYPE]);

impl<'tcx> LateLintPass<'tcx> for DisallowedType {
    fn check_crate(&mut self, cx: &LateContext<'_>, _: &Crate<'_>) {
        for path in &self.disallowed {
            let segs = path.iter().map(ToString::to_string).collect::<Vec<_>>();
            if let Res::Def(_, id) = clippy_utils::path_to_res(cx, &segs.iter().map(String::as_str).collect::<Vec<_>>())
            {
                self.def_ids.insert((id, path.clone()));
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if_chain! {
            if let ItemKind::Use(path, UseKind::Single) = &item.kind;
            if let Res::Def(_, did) = path.res;
            if let Some((_, name)) = self.def_ids.iter().find(|(id, _)| *id == did);
            then {
                emit(cx, name, item.span,);
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx>) {
        if_chain! {
            if let TyKind::Path(path) = &ty.kind;
            if let Some(did) = cx.qpath_res(path, ty.hir_id).opt_def_id();
            if let Some((_, name)) = self.def_ids.iter().find(|(id, _)| *id == did);
            then {
                emit(cx, name, path.span());
            }
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>, _: TraitBoundModifier) {
        if_chain! {
            if let Res::Def(_, did) = poly.trait_ref.path.res;
            if let Some((_, name)) = self.def_ids.iter().find(|(id, _)| *id == did);
            then {
                emit(cx, name, poly.trait_ref.path.span);
            }
        }
    }

    // TODO: if non primitive const generics are a thing
    // fn check_generic_arg(&mut self, cx: &LateContext<'tcx>, arg: &'tcx GenericArg<'tcx>) {
    //     match arg {
    //         GenericArg::Const(c) => {},
    //     }
    // }
    // fn check_generic_param(&mut self, cx: &LateContext<'tcx>, param: &'tcx GenericParam<'tcx>) {
    //     match param.kind {
    //         GenericParamKind::Const { .. } => {},
    //     }
    // }
}

fn emit(cx: &LateContext<'_>, name: &[Symbol], span: Span) {
    let name = name.iter().map(|s| s.to_ident_string()).collect::<Vec<_>>().join("::");
    span_lint(
        cx,
        DISALLOWED_TYPE,
        span,
        &format!("`{}` is not allowed according to config", name),
    );
}
