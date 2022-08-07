use std::{cmp, ops};

use rustc_hir::HirId;
use rustc_span::Span;

/// Tracks whether executing a node may exit normally (versus
/// return/break/panic, which "diverge", leaving dead code in their
/// wake). Tracked semi-automatically (through type variables marked
/// as diverging), with some manual adjustments for control-flow
/// primitives (approximating a CFG).
#[derive(Copy, Clone, Debug)]
pub(crate) enum Diverges {
    /// Potentially unknown, some cases converge,
    /// others require a CFG to determine them.
    Maybe,

    /// This expression is uninhabited, we want to
    /// emit a diagnostic but not pollute type checking.
    UninhabitedExpr(HirId, Span),

    /// Same as `UninhabitedExpr` but with reachability
    /// warning already emitted.
    Warned,

    /// Definitely known to diverge and therefore
    /// not reach the next sibling or its parent.
    Always(DivergeReason, Span),

    /// Same as `Always` but with a reachability
    /// warning already emitted.
    WarnedAlways,
}

// Convenience impls for combining `Diverges`.

impl ops::BitAnd for Diverges {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        cmp::min_by_key(self, other, Self::ordinal)
    }
}

impl ops::BitOr for Diverges {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        // argument order is to prefer `self` if ordinal is equal
        cmp::max_by_key(other, self, Self::ordinal)
    }
}

impl ops::BitAndAssign for Diverges {
    fn bitand_assign(&mut self, other: Self) {
        *self = *self & other;
    }
}

impl ops::BitOrAssign for Diverges {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other;
    }
}

impl Diverges {
    pub(super) fn is_always(self) -> bool {
        match self {
            Self::Maybe | Diverges::UninhabitedExpr(..) | Diverges::Warned => false,
            Self::Always(..) | Self::WarnedAlways => true,
        }
    }

    fn ordinal(&self) -> u8 {
        match self {
            Self::Maybe => 0,
            Self::UninhabitedExpr(..) => 1,
            Self::Warned => 2,
            Self::Always { .. } => 3,
            Self::WarnedAlways => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DivergeReason {
    AllArmsDiverge,
    NeverPattern,
    UninhabitedExpr(HirId),
    Other,
}
