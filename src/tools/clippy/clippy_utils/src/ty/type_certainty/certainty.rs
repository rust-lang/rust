use rustc_hir::def_id::DefId;
use std::fmt::Debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Certainty {
    /// Determining the type requires contextual information.
    Uncertain,

    /// The type can be determined purely from subexpressions. If the argument is `Some(..)`, the
    /// specific `DefId` is known. Such arguments are needed to handle path segments whose `res` is
    /// `Res::Err`.
    Certain(Option<DefId>),

    /// The heuristic believes that more than one `DefId` applies to a type---this is a bug.
    Contradiction,
}

pub trait Meet {
    fn meet(self, other: Self) -> Self;
}

pub trait TryJoin: Sized {
    fn try_join(self, other: Self) -> Option<Self>;
}

impl Meet for Option<DefId> {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (None, _) | (_, None) => None,
            (Some(lhs), Some(rhs)) => (lhs == rhs).then_some(lhs),
        }
    }
}

impl TryJoin for Option<DefId> {
    fn try_join(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Some(lhs), Some(rhs)) => (lhs == rhs).then_some(Some(lhs)),
            (Some(def_id), _) | (_, Some(def_id)) => Some(Some(def_id)),
            (None, None) => Some(None),
        }
    }
}

impl Meet for Certainty {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Certainty::Uncertain, _) | (_, Certainty::Uncertain) => Certainty::Uncertain,
            (Certainty::Certain(lhs), Certainty::Certain(rhs)) => Certainty::Certain(lhs.meet(rhs)),
            (Certainty::Certain(inner), _) | (_, Certainty::Certain(inner)) => Certainty::Certain(inner),
            (Certainty::Contradiction, Certainty::Contradiction) => Certainty::Contradiction,
        }
    }
}

impl Certainty {
    /// Join two `Certainty`s preserving their `DefId`s (if any). Generally speaking, this method
    /// should be used only when `self` and `other` refer directly to types. Otherwise,
    /// `join_clearing_def_ids` should be used.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Certainty::Contradiction, _) | (_, Certainty::Contradiction) => Certainty::Contradiction,

            (Certainty::Certain(lhs), Certainty::Certain(rhs)) => {
                if let Some(inner) = lhs.try_join(rhs) {
                    Certainty::Certain(inner)
                } else {
                    debug_assert!(false, "Contradiction with {lhs:?} and {rhs:?}");
                    Certainty::Contradiction
                }
            },

            (Certainty::Certain(inner), _) | (_, Certainty::Certain(inner)) => Certainty::Certain(inner),

            (Certainty::Uncertain, Certainty::Uncertain) => Certainty::Uncertain,
        }
    }

    /// Join two `Certainty`s after clearing their `DefId`s. This method should be used when `self`
    /// or `other` do not necessarily refer to types, e.g., when they are aggregations of other
    /// `Certainty`s.
    pub fn join_clearing_def_ids(self, other: Self) -> Self {
        self.clear_def_id().join(other.clear_def_id())
    }

    pub fn clear_def_id(self) -> Certainty {
        if matches!(self, Certainty::Certain(_)) {
            Certainty::Certain(None)
        } else {
            self
        }
    }

    pub fn with_def_id(self, def_id: DefId) -> Certainty {
        if matches!(self, Certainty::Certain(_)) {
            Certainty::Certain(Some(def_id))
        } else {
            self
        }
    }

    pub fn to_def_id(self) -> Option<DefId> {
        match self {
            Certainty::Certain(inner) => inner,
            _ => None,
        }
    }

    pub fn is_certain(self) -> bool {
        matches!(self, Self::Certain(_))
    }
}

/// Think: `iter.all(/* is certain */)`
pub fn meet(iter: impl Iterator<Item = Certainty>) -> Certainty {
    iter.fold(Certainty::Certain(None), Certainty::meet)
}

/// Think: `iter.any(/* is certain */)`
pub fn join(iter: impl Iterator<Item = Certainty>) -> Certainty {
    iter.fold(Certainty::Uncertain, Certainty::join)
}
