use crate::ty::query::Providers;
use crate::hir::def_id::DefId;
use crate::hir;
use crate::ty::TyCtxt;
use syntax_pos::symbol::{sym, Symbol};
use rustc_target::spec::abi::Abi;
use crate::hir::map::blocks::FnLikeNode;
use syntax::attr;

impl<'tcx> TyCtxt<'tcx> {
    /// Whether the `def_id` counts as const fn in your current crate, considering all active
    /// feature gates
    pub fn is_const_fn(self, def_id: DefId) -> bool {
        self.is_const_fn_raw(def_id) && match self.is_unstable_const_fn(def_id) {
            Some(feature_name) => {
                // has a `rustc_const_unstable` attribute, check whether the user enabled the
                // corresponding feature gate.
                self.features()
                    .declared_lib_features
                    .iter()
                    .any(|&(sym, _)| sym == feature_name)
            },
            // functions without const stability are either stable user written
            // const fn or the user is using feature gates and we thus don't
            // care what they do
            None => true,
        }
    }

    /// Whether the `def_id` is an unstable const fn and what feature gate is necessary to enable it
    pub fn is_unstable_const_fn(self, def_id: DefId) -> Option<Symbol> {
        if self.is_const_fn_raw(def_id) {
            let const_stab = self.lookup_const_stability(def_id)?;
            if const_stab.level.is_unstable() {
                Some(const_stab.feature)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Returns `true` if the `def_id` refers to an intrisic which we've whitelisted
    /// for being called from stable `const fn`s (`min_const_fn`).
    ///
    /// Adding more intrinsics requires sign-off from @rust-lang/lang.
    ///
    /// This list differs from the list in `is_const_intrinsic` in the sense that any item on this
    /// list must be on the `is_const_intrinsic` list, too, because if an intrinsic is callable from
    /// stable, it must be callable at all.
    fn is_intrinsic_min_const_fn(self, def_id: DefId) -> bool {
        match self.item_name(def_id) {
            | sym::size_of
            | sym::min_align_of
            | sym::needs_drop
            // Arithmetic:
            | sym::add_with_overflow // ~> .overflowing_add
            | sym::sub_with_overflow // ~> .overflowing_sub
            | sym::mul_with_overflow // ~> .overflowing_mul
            | sym::wrapping_add // ~> .wrapping_add
            | sym::wrapping_sub // ~> .wrapping_sub
            | sym::wrapping_mul // ~> .wrapping_mul
            | sym::saturating_add // ~> .saturating_add
            | sym::saturating_sub // ~> .saturating_sub
            | sym::unchecked_shl // ~> .wrapping_shl
            | sym::unchecked_shr // ~> .wrapping_shr
            | sym::rotate_left // ~> .rotate_left
            | sym::rotate_right // ~> .rotate_right
            | sym::ctpop // ~> .count_ones
            | sym::ctlz // ~> .leading_zeros
            | sym::cttz // ~> .trailing_zeros
            | sym::bswap // ~> .swap_bytes
            | sym::bitreverse // ~> .reverse_bits
            => true,
            _ => false,
        }
    }

    /// Returns `true` if this function must conform to `min_const_fn`
    pub fn is_min_const_fn(self, def_id: DefId) -> bool {
        // Bail out if the signature doesn't contain `const`
        if !self.is_const_fn_raw(def_id) {
            return false;
        }
        if let Abi::RustIntrinsic = self.fn_sig(def_id).abi() {
            return self.is_intrinsic_min_const_fn(def_id);
        }

        if self.features().staged_api {
            // In order for a libstd function to be considered min_const_fn
            // it needs to be stable and have no `rustc_const_unstable` attribute.
            match self.lookup_const_stability(def_id) {
                // `rustc_const_unstable` functions don't need to conform.
                Some(&attr::ConstStability { ref level, .. }) if level.is_unstable() => false,
                None => if let Some(stab) = self.lookup_stability(def_id) {
                    if stab.level.is_stable() {
                        self.sess.span_err(
                            self.def_span(def_id),
                            "stable const functions must have either `rustc_const_stable` or \
                            `rustc_const_unstable` attribute",
                        );
                        // While we errored above, because we don't know if we need to conform, we
                        // err on the "safe" side and require min_const_fn.
                        true
                    } else {
                        // Unstable functions need not conform to min_const_fn.
                        false
                    }
                } else {
                    // Internal functions are forced to conform to min_const_fn.
                    // Annotate the internal function with a const stability attribute if
                    // you need to use unstable features.
                    // Note: this is an arbitrary choice that does not affect stability or const
                    // safety or anything, it just changes whether we need to annotate some
                    // internal functions with `rustc_const_stable` or with `rustc_const_unstable`
                    true
                },
                // Everything else needs to conform, because it would be callable from
                // other `min_const_fn` functions.
                _ => true,
            }
        } else {
            // users enabling the `const_fn` feature gate can do what they want
            !self.features().const_fn
        }
    }
}


pub fn provide(providers: &mut Providers<'_>) {
    /// Const evaluability whitelist is here to check evaluability at the
    /// top level beforehand.
    fn is_const_intrinsic(tcx: TyCtxt<'_>, def_id: DefId) -> Option<bool> {
        match tcx.fn_sig(def_id).abi() {
            Abi::RustIntrinsic |
            Abi::PlatformIntrinsic => {
                // FIXME: deduplicate these two lists as much as possible
                match tcx.item_name(def_id) {
                    // Keep this list in the same order as the match patterns in
                    // `librustc_mir/interpret/intrinsics.rs`

                    // This whitelist is a list of intrinsics that have a miri-engine implementation
                    // and can thus be called when enabling enough feature gates. The similar
                    // whitelist in `is_intrinsic_min_const_fn` (in this file), exists for allowing
                    // the intrinsics to be called by stable const fns.
                    | sym::caller_location

                    | sym::min_align_of
                    | sym::pref_align_of
                    | sym::needs_drop
                    | sym::size_of
                    | sym::type_id
                    | sym::type_name

                    | sym::ctpop
                    | sym::cttz
                    | sym::cttz_nonzero
                    | sym::ctlz
                    | sym::ctlz_nonzero
                    | sym::bswap
                    | sym::bitreverse

                    | sym::wrapping_add
                    | sym::wrapping_sub
                    | sym::wrapping_mul
                    | sym::add_with_overflow
                    | sym::sub_with_overflow
                    | sym::mul_with_overflow

                    | sym::saturating_add
                    | sym::saturating_sub

                    | sym::unchecked_shl
                    | sym::unchecked_shr

                    | sym::rotate_left
                    | sym::rotate_right

                    | sym::ptr_offset_from

                    | sym::transmute

                    | sym::simd_insert

                    | sym::simd_extract

                    => Some(true),

                    _ => Some(false)
                }
            }
            _ => None
        }
    }

    /// Checks whether the function has a `const` modifier or, in case it is an intrinsic, whether
    /// said intrinsic is on the whitelist for being const callable.
    fn is_const_fn_raw(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        let hir_id = tcx.hir().as_local_hir_id(def_id)
                              .expect("Non-local call to local provider is_const_fn");

        let node = tcx.hir().get(hir_id);

        if let Some(whitelisted) = is_const_intrinsic(tcx, def_id) {
            whitelisted
        } else if let Some(fn_like) = FnLikeNode::from_node(node) {
            fn_like.constness() == hir::Constness::Const
        } else if let hir::Node::Ctor(_) = node {
            true
        } else {
            false
        }
    }

    fn is_promotable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        tcx.is_const_fn(def_id) && match tcx.lookup_const_stability(def_id) {
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
            tcx.lookup_const_stability(def_id)
                .map(|stab| stab.allow_const_fn_ptr).unwrap_or(false)
    }

    *providers = Providers {
        is_const_fn_raw,
        is_promotable_const_fn,
        const_fn_is_allowed_fn_ptr,
        ..*providers
    };
}
