use std::ops;

use rustc_span::Span;

/// Tracks whether executing a node may exit normally (versus
/// return/break/panic, which "diverge", leaving dead code in their
/// wake). Tracked semi-automatically (through type variables marked
/// as diverging), with some manual adjustments for control-flow
/// primitives (approximating a CFG).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Diverges {
    /// Potentially unknown, some cases converge,
    /// others require a CFG to determine them.
    Maybe,

    /// Definitely known to diverge and therefore
    /// not reach the next sibling or its parent.
    Always {
        /// The `Span` points to the expression
        /// that caused us to diverge
        /// (e.g. `return`, `break`, etc).
        span: Span,
        /// In some cases (e.g. a `match` expression
        /// where all arms diverge), we may be
        /// able to provide a more informative
        /// message to the user.
        /// If this is `None`, a default message
        /// will be generated, which is suitable
        /// for most cases.
        custom_note: Option<&'static str>,
    },

    /// Same as `Always` but with a reachability
    /// warning already emitted.
    WarnedAlways,
}

// Convenience impls for combining `Diverges`.

impl ops::BitAnd for Diverges {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        match (self, other) {
            (x @ Self::Maybe, _) => x,
            (_, x @ Self::Maybe) => x,
            // Using `span`/`custom_note` from `self` seems to give better error messages.
            (x @ Self::Always { .. }, Self::Always { .. }) => x,
            (Self::WarnedAlways, x) => x,
            (x, Self::WarnedAlways) => x,
        }
    }
}

impl ops::BitOr for Diverges {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        match (self, other) {
            (x @ Self::WarnedAlways, _) => x,
            (_, x @ Self::WarnedAlways) => x,
            // Using `span`/`custom_note` from `self` seems to give better error messages.
            (x @ Self::Always { .. }, Self::Always { .. }) => x,
            (Self::Maybe, x) => x,
            (x, Self::Maybe) => x,
        }
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
    /// Creates a `Diverges::Always` with the provided `span` and the default note message.
    pub(super) fn always(span: Span) -> Diverges {
        Diverges::Always { span, custom_note: None }
    }

    pub(super) fn is_always(self) -> bool {
        matches!(self, Diverges::WarnedAlways | Diverges::Always { .. })
    }
}
