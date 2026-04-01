use rustc_macros::{Decodable, Encodable, HashStable};

use crate::ty::{Ty, TyCtxt, TypingEnv};

/// Summarizes how a parameter (a return place or an argument) is used inside a MIR body.
#[derive(Clone, Copy, PartialEq, Debug, Decodable, Encodable, HashStable)]
pub struct UsageSummary(u8);

bitflags::bitflags! {
    impl UsageSummary: u8 {
        /// This parameter is dropped when it `needs_drop`.
        const DROP = 1 << 0;
        /// There is a shared borrow to this parameter.
        /// It allows for mutation unless parameter is `Freeze`.
        const SHARED_BORROW = 1 << 1;
        /// This parameter is mutated (excluding through a drop or a shared borrow).
        const MUTATE = 1 << 2;
        /// This parameter is captured (excluding through a drop).
        const CAPTURE = 1 << 3;
    }
}

/// Parameter attributes that can only be determined by examining the body of a function instead
/// of just its signature.
///
/// These can be useful for optimization purposes when a function is directly called. We compute
/// them and store them into the crate metadata so that downstream crates can make use of them.
///
/// Right now, we have `readonly` and `captures(none)`, but `no_alias` might be useful in the
/// future.
#[derive(Clone, Copy, PartialEq, Debug, Decodable, Encodable, HashStable)]
pub struct DeducedParamAttrs {
    pub usage: UsageSummary,
}

impl DeducedParamAttrs {
    /// Returns true if no attributes have been deduced.
    #[inline]
    pub fn is_default(self) -> bool {
        self.usage.contains(UsageSummary::MUTATE | UsageSummary::CAPTURE)
    }

    /// For parameters passed indirectly, returns true if pointer is never written through.
    pub fn read_only<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        // Only if all checks pass is this truly read-only.
        if self.usage.contains(UsageSummary::MUTATE) {
            return false;
        }
        if self.usage.contains(UsageSummary::DROP) && ty.needs_drop(tcx, typing_env) {
            return false;
        }
        if self.usage.contains(UsageSummary::SHARED_BORROW) && !ty.is_freeze(tcx, typing_env) {
            return false;
        }
        true
    }

    /// For parameters passed indirectly, returns true if pointer is not captured, i.e., its
    /// address is not captured, and pointer is used neither for reads nor writes after function
    /// returns.
    pub fn captures_none<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        if self.usage.contains(UsageSummary::CAPTURE) {
            return false;
        }
        if self.usage.contains(UsageSummary::DROP) && ty.needs_drop(tcx, typing_env) {
            return false;
        }
        true
    }
}
