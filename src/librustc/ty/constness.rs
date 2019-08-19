use crate::ty::query::Providers;
use crate::hir::def_id::DefId;
use crate::hir;
use crate::ty::TyCtxt;
use syntax_pos::symbol::{sym, Symbol};
use crate::hir::map::blocks::FnLikeNode;
use syntax::attr;

impl<'tcx> TyCtxt<'tcx> {
    /// Whether the `def_id` counts as const fn in your current crate, considering all active
    /// feature gates
    pub fn is_const_fn(self, def_id: DefId) -> bool {
        self.is_const_fn_raw(def_id) && match self.is_unstable_const_fn(def_id) {
            Some(feature_name) => {
                // has a `rustc_const_unstable` attribute, check whether the user enabled the
                // corresponding feature gate, const_constructor is not a lib feature, so has
                // to be checked separately.
                self.features()
                    .declared_lib_features
                    .iter()
                    .any(|&(sym, _)| sym == feature_name)
                    || (feature_name == sym::const_constructor
                        && self.features().const_constructor)
            },
            // functions without const stability are either stable user written
            // const fn or the user is using feature gates and we thus don't
            // care what they do
            None => true,
        }
    }

    /// Whether the `def_id` is an unstable const fn and what feature gate is necessary to enable it
    pub fn is_unstable_const_fn(self, def_id: DefId) -> Option<Symbol> {
        if self.is_constructor(def_id) {
            Some(sym::const_constructor)
        } else if self.is_const_fn_raw(def_id) {
            self.lookup_stability(def_id)?.const_stability
        } else {
            None
        }
    }

    /// Returns `true` if this function must conform to `min_const_fn`
    pub fn is_min_const_fn(self, def_id: DefId) -> bool {
        // Bail out if the signature doesn't contain `const`
        if !self.is_const_fn_raw(def_id) {
            return false;
        }

        if self.features().staged_api {
            // in order for a libstd function to be considered min_const_fn
            // it needs to be stable and have no `rustc_const_unstable` attribute
            match self.lookup_stability(def_id) {
                // stable functions with unstable const fn aren't `min_const_fn`
                Some(&attr::Stability { const_stability: Some(_), .. }) => false,
                // unstable functions don't need to conform
                Some(&attr::Stability { ref level, .. }) if level.is_unstable() => false,
                // everything else needs to conform, because it would be callable from
                // other `min_const_fn` functions
                _ => true,
            }
        } else {
            // users enabling the `const_fn` feature gate can do what they want
            !self.features().const_fn
        }
    }
}


pub fn provide(providers: &mut Providers<'_>) {
    /// only checks whether the function has a `const` modifier
    fn is_const_fn_raw(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        let hir_id = tcx.hir().as_local_hir_id(def_id)
                              .expect("Non-local call to local provider is_const_fn");

        let node = tcx.hir().get(hir_id);
        if let Some(fn_like) = FnLikeNode::from_node(node) {
            fn_like.constness() == hir::Constness::Const
        } else if let hir::Node::Ctor(_) = node {
            true
        } else {
            false
        }
    }

    fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        tcx.is_const_fn(def_id) && match tcx.lookup_stability(def_id) {
            Some(stab) => {
                if cfg!(debug_assertions) && stab.promotable {
                    let sig = tcx.fn_sig(def_id);
                    assert_eq!(
                        sig.unsafety(),
                        hir::Unsafety::Normal,
                        "don't mark const unsafe fns as promotable",
                        // https://github.com/rust-lang/rust/pull/53851#issuecomment-418760682
                    );
                }
                stab.promotable
            },
            None => false,
        }
    }

    fn const_fn_is_allowed_fn_ptr(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        tcx.is_const_fn(def_id) &&
            tcx.lookup_stability(def_id)
                .map(|stab| stab.allow_const_fn_ptr).unwrap_or(false)
    }

    *providers = Providers {
        is_const_fn_raw,
        is_promotable_const_fn,
        const_fn_is_allowed_fn_ptr,
        ..*providers
    };
}
