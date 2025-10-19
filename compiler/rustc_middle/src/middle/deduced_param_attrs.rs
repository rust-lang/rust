use rustc_macros::{Decodable, Encodable, HashStable};

use crate::ty::{Ty, TyCtxt, TypingEnv};

/// Flags that dictate how a parameter is mutated. If the flags are empty, the param is
/// read-only. If non-empty, it is read-only if *all* flags' conditions are met.
#[derive(Clone, Copy, PartialEq, Debug, Decodable, Encodable, HashStable)]
pub struct DeducedReadOnlyParam(u8);

bitflags::bitflags! {
    impl DeducedReadOnlyParam: u8 {
        /// This parameter is dropped. It is read-only if `!needs_drop`.
        const IF_NO_DROP = 1 << 0;
        /// This parameter is borrowed. It is read-only if `Freeze`.
        const IF_FREEZE   = 1 << 1;
        /// This parameter is mutated. It is never read-only.
        const MUTATED     = 1 << 2;
    }
}

/// Parameter attributes that can only be determined by examining the body of a function instead
/// of just its signature.
///
/// These can be useful for optimization purposes when a function is directly called. We compute
/// them and store them into the crate metadata so that downstream crates can make use of them.
///
/// Right now, we only have `read_only`, but `no_capture` and `no_alias` might be useful in the
/// future.
#[derive(Clone, Copy, PartialEq, Debug, Decodable, Encodable, HashStable)]
pub struct DeducedParamAttrs {
    /// The parameter is marked immutable in the function.
    pub read_only: DeducedReadOnlyParam,
}

// By default, consider the parameters to be mutated.
impl Default for DeducedParamAttrs {
    #[inline]
    fn default() -> DeducedParamAttrs {
        DeducedParamAttrs { read_only: DeducedReadOnlyParam::MUTATED }
    }
}

impl DeducedParamAttrs {
    #[inline]
    pub fn is_default(self) -> bool {
        self.read_only.contains(DeducedReadOnlyParam::MUTATED)
    }

    pub fn read_only<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let read_only = self.read_only;
        // We have to check *all* set bits; only if all checks pass is this truly read-only.
        if read_only.contains(DeducedReadOnlyParam::MUTATED) {
            return false;
        }
        if read_only.contains(DeducedReadOnlyParam::IF_NO_DROP) && ty.needs_drop(tcx, typing_env) {
            return false;
        }
        if read_only.contains(DeducedReadOnlyParam::IF_FREEZE) && !ty.is_freeze(tcx, typing_env) {
            return false;
        }
        true
    }
}
