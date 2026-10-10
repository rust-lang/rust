/// The result of a transmutability query under the supplied `Assume` options.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Answer<R, T> {
    /// The analysis requires no further conditions.
    Yes,
    /// The analysis could not establish transmutability.
    No(Reason<T>),
    /// Transmutability depends on conditions that the trait solver must discharge.
    If(Condition<R, T>),
}

/// A condition which must hold for safe transmutation to be possible.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Condition<R, T> {
    /// `Src` is transmutable into `Dst`, if `src` is transmutable into `dst`.
    Transmutable { src: T, dst: T },

    /// The region `long` must outlive `short`.
    Outlives { long: R, short: R },

    /// The type `ty` must satisfy `Freeze`.
    Immutable { ty: T },

    /// `Src` is transmutable into `Dst`, if all of the enclosed requirements are met.
    IfAll(Vec<Condition<R, T>>),

    /// `Src` is transmutable into `Dst` if any of the enclosed requirements are met.
    IfAny(Vec<Condition<R, T>>),
}

/// Answers "why wasn't the source type transmutable into the destination type?"
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub enum Reason<T> {
    /// The layout of the source type is not yet supported.
    SrcIsNotYetSupported,
    /// The layout of the destination type is not yet supported.
    DstIsNotYetSupported,
    /// The layout of the destination type is bit-incompatible with the source type.
    DstIsBitIncompatible,
    /// The destination type is uninhabited.
    DstUninhabited,
    /// The destination type may carry safety invariants.
    DstMayHaveSafetyInvariants,
    /// `Dst` is larger than `Src`, and the excess bytes were not exclusively uninitialized.
    DstIsTooBig,
    /// The destination referent is larger than the source referent.
    DstRefIsTooBig {
        /// The referent of the source type.
        src: T,
        /// The size of the source type's referent.
        src_size: usize,
        /// The too-large referent of the destination type.
        dst: T,
        /// The size of the destination type's referent.
        dst_size: usize,
    },
    /// The destination referent requires stricter alignment than the source referent.
    DstHasStricterAlignment { src_min_align: usize, dst_min_align: usize },
    /// Can't go from shared pointer to unique pointer
    DstIsMoreUnique,
    /// Encountered a type error
    TypeError,
    /// The layout of src is unknown
    SrcLayoutUnknown,
    /// The layout of dst is unknown
    DstLayoutUnknown,
    /// The size of src is overflow
    SrcSizeOverflow,
    /// The size of dst is overflow
    DstSizeOverflow,
}

impl<R, T> Answer<R, T> {
    /// Requires both answers, combining their conditions into a conjunction.
    pub fn and(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // Prefer a specific reason over generic bit incompatibility;
            // otherwise, retain the left-hand reason.
            (Answer::No(Reason::DstIsBitIncompatible), Answer::No(reason))
            | (Answer::No(reason), Answer::No(_))
            // If either is an error, return it
            | (Answer::No(reason), _) | (_, Answer::No(reason)) => Answer::No(reason),
            // If only one side has a condition, pass it along
            (Answer::Yes, other) | (other, Answer::Yes) => other,
            // If both sides have IfAll conditions, merge them
            (Answer::If(Condition::IfAll(mut lhs)), Answer::If(Condition::IfAll(ref mut rhs))) => {
                lhs.append(rhs);
                Answer::If(Condition::IfAll(lhs))
            }
            // If only one side is an IfAll, add the other Condition to it
            (Answer::If(cond), Answer::If(Condition::IfAll(mut conds)))
            | (Answer::If(Condition::IfAll(mut conds)), Answer::If(cond)) => {
                conds.push(cond);
                Answer::If(Condition::IfAll(conds))
            }
            // Otherwise, both lhs and rhs conditions can be combined in a parent IfAll
            (Answer::If(lhs), Answer::If(rhs)) => Answer::If(Condition::IfAll(vec![lhs, rhs])),
        }
    }

    /// Combines alternative answers and collects their conditions in an `IfAny`.
    ///
    /// Currently, combining `Yes` with `If` retains the condition. This differs
    /// from Boolean disjunction, where unconditional success would suffice.
    pub fn or(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // Prefer a specific reason over generic bit incompatibility;
            // otherwise, retain the left-hand reason.
            (Answer::No(Reason::DstIsBitIncompatible), Answer::No(reason))
            | (Answer::No(reason), Answer::No(_)) => Answer::No(reason),
            // Otherwise, errors can be ignored for the rest of the pattern matching
            (Answer::No(_), other) | (other, Answer::No(_)) => other.or(Answer::Yes),
            // If only one side has a condition, pass it along
            (Answer::Yes, other) | (other, Answer::Yes) => other,
            // If both sides have IfAny conditions, merge them
            (Answer::If(Condition::IfAny(mut lhs)), Answer::If(Condition::IfAny(ref mut rhs))) => {
                lhs.append(rhs);
                Answer::If(Condition::IfAny(lhs))
            }
            // If only one side is an IfAny, add the other Condition to it
            (Answer::If(cond), Answer::If(Condition::IfAny(mut conds)))
            | (Answer::If(Condition::IfAny(mut conds)), Answer::If(cond)) => {
                conds.push(cond);
                Answer::If(Condition::IfAny(conds))
            }
            // Otherwise, both lhs and rhs conditions can be combined in a parent IfAny
            (Answer::If(lhs), Answer::If(rhs)) => Answer::If(Condition::IfAny(vec![lhs, rhs])),
        }
    }
}
