use std::cmp::{Ordering, PartialOrd};
use std::fmt;

use crate::borrow_tracker::tree_borrows::diagnostics::TransitionError;
use crate::borrow_tracker::tree_borrows::tree::AccessRelatedness;
use crate::borrow_tracker::AccessKind;

/// The activation states of a pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PermissionPriv {
    /// represents: a local reference that has not yet been written to;
    /// allows: child reads, foreign reads, foreign writes if type is freeze;
    /// affected by: child writes (becomes Active),
    /// rejects: foreign writes (Disabled, except if type is not freeze).
    ///
    /// special case: behaves differently when protected, which is where `conflicted`
    /// is relevant
    /// - `conflicted` is set on foreign reads,
    /// - `conflicted` must not be set on child writes (there is UB otherwise).
    /// This is so that the behavior of `Reserved` adheres to the rules of `noalias`:
    /// - foreign-read then child-write is UB due to `conflicted`,
    /// - child-write then foreign-read is UB since child-write will activate and then
    ///   foreign-read disables a protected `Active`, which is UB.
    Reserved { ty_is_freeze: bool, conflicted: bool },
    /// represents: a unique pointer;
    /// allows: child reads, child writes;
    /// rejects: foreign reads (Frozen), foreign writes (Disabled).
    Active,
    /// represents: a shared pointer;
    /// allows: all read accesses;
    /// rejects child writes (UB), foreign writes (Disabled).
    Frozen,
    /// represents: a dead pointer;
    /// allows: all foreign accesses;
    /// rejects: all child accesses (UB).
    Disabled,
}
use PermissionPriv::*;

impl PartialOrd for PermissionPriv {
    /// PermissionPriv is ordered by the reflexive transitive closure of
    /// `Reserved(conflicted=false) < Reserved(conflicted=true) < Active < Frozen < Disabled`.
    /// `Reserved` that have incompatible `ty_is_freeze` are incomparable to each other.
    /// This ordering matches the reachability by transitions, as asserted by the exhaustive test
    /// `permissionpriv_partialord_is_reachability`.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Ordering::*;
        Some(match (self, other) {
            (a, b) if a == b => Equal,
            (Disabled, _) => Greater,
            (_, Disabled) => Less,
            (Frozen, _) => Greater,
            (_, Frozen) => Less,
            (Active, _) => Greater,
            (_, Active) => Less,
            (
                Reserved { ty_is_freeze: f1, conflicted: c1 },
                Reserved { ty_is_freeze: f2, conflicted: c2 },
            ) => {
                // No transition ever changes `ty_is_freeze`.
                if f1 != f2 {
                    return None;
                }
                // `bool` is ordered such that `false <= true`, so this works as intended.
                c1.cmp(c2)
            }
        })
    }
}

impl PermissionPriv {
    /// Check if `self` can be the initial state of a pointer.
    fn is_initial(&self) -> bool {
        matches!(self, Reserved { conflicted: false, .. } | Frozen)
    }
}

/// This module controls how each permission individually reacts to an access.
/// Although these functions take `protected` as an argument, this is NOT because
/// we check protector violations here, but because some permissions behave differently
/// when protected.
mod transition {
    use super::*;
    /// A child node was read-accessed: UB on Disabled, noop on the rest.
    fn child_read(state: PermissionPriv, _protected: bool) -> Option<PermissionPriv> {
        Some(match state {
            Disabled => return None,
            // The inner data `ty_is_freeze` of `Reserved` is always irrelevant for Read
            // accesses, since the data is not being mutated. Hence the `{ .. }`.
            readable @ (Reserved { .. } | Active | Frozen) => readable,
        })
    }

    /// A non-child node was read-accessed: keep `Reserved` but mark it as `conflicted` if it
    /// is protected; invalidate `Active`.
    fn foreign_read(state: PermissionPriv, protected: bool) -> Option<PermissionPriv> {
        Some(match state {
            // Non-writeable states just ignore foreign reads.
            non_writeable @ (Frozen | Disabled) => non_writeable,
            // Writeable states are more tricky, and depend on whether things are protected.
            // The inner data `ty_is_freeze` of `Reserved` is always irrelevant for Read
            // accesses, since the data is not being mutated. Hence the `{ .. }`

            // Someone else read. To make sure we won't write before function exit,
            // we set the "conflicted" flag, which will disallow writes while we are protected.
            Reserved { ty_is_freeze, .. } if protected =>
                Reserved { ty_is_freeze, conflicted: true },
            // Before activation and without protectors, foreign reads are fine.
            // That's the entire point of 2-phase borrows.
            res @ Reserved { .. } => res,
            Active =>
                if protected {
                    // We wrote, someone else reads -- that's bad.
                    // (If this is initialized, this move-to-protected will mean insta-UB.)
                    Disabled
                } else {
                    // We don't want to disable here to allow read-read reordering: it is crucial
                    // that the foreign read does not invalidate future reads through this tag.
                    Frozen
                },
        })
    }

    /// A child node was write-accessed: `Reserved` must become `Active` to obtain
    /// write permissions, `Frozen` and `Disabled` cannot obtain such permissions and produce UB.
    fn child_write(state: PermissionPriv, protected: bool) -> Option<PermissionPriv> {
        Some(match state {
            // If the `conflicted` flag is set, then there was a foreign read during
            // the function call that is still ongoing (still `protected`),
            // this is UB (`noalias` violation).
            Reserved { conflicted: true, .. } if protected => return None,
            // A write always activates the 2-phase borrow, even with interior
            // mutability
            Reserved { .. } | Active => Active,
            Frozen | Disabled => return None,
        })
    }

    /// A non-child node was write-accessed: this makes everything `Disabled` except for
    /// non-protected interior mutable `Reserved` which stay the same.
    fn foreign_write(state: PermissionPriv, protected: bool) -> Option<PermissionPriv> {
        Some(match state {
            Reserved { .. } if protected => Disabled,
            res @ Reserved { ty_is_freeze: false, .. } => res,
            _ => Disabled,
        })
    }

    /// Dispatch handler depending on the kind of access and its position.
    pub(super) fn perform_access(
        kind: AccessKind,
        rel_pos: AccessRelatedness,
        child: PermissionPriv,
        protected: bool,
    ) -> Option<PermissionPriv> {
        match (kind, rel_pos.is_foreign()) {
            (AccessKind::Write, true) => foreign_write(child, protected),
            (AccessKind::Read, true) => foreign_read(child, protected),
            (AccessKind::Write, false) => child_write(child, protected),
            (AccessKind::Read, false) => child_read(child, protected),
        }
    }
}

/// Public interface to the state machine that controls read-write permissions.
/// This is the "private `enum`" pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub struct Permission {
    inner: PermissionPriv,
}

/// Transition from one permission to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PermTransition {
    from: PermissionPriv,
    to: PermissionPriv,
}

impl Permission {
    /// Check if `self` can be the initial state of a pointer.
    pub fn is_initial(&self) -> bool {
        self.inner.is_initial()
    }

    /// Default initial permission of the root of a new tree.
    /// Must *only* be used for the root, this is not in general an "initial" permission!
    pub fn new_active() -> Self {
        Self { inner: Active }
    }

    /// Default initial permission of a reborrowed mutable reference.
    pub fn new_reserved(ty_is_freeze: bool) -> Self {
        Self { inner: Reserved { ty_is_freeze, conflicted: false } }
    }

    /// Default initial permission of a reborrowed shared reference
    pub fn new_frozen() -> Self {
        Self { inner: Frozen }
    }

    /// Apply the transition to the inner PermissionPriv.
    pub fn perform_access(
        kind: AccessKind,
        rel_pos: AccessRelatedness,
        old_perm: Self,
        protected: bool,
    ) -> Option<PermTransition> {
        let old_state = old_perm.inner;
        transition::perform_access(kind, rel_pos, old_state, protected)
            .map(|new_state| PermTransition { from: old_state, to: new_state })
    }
}

impl PermTransition {
    /// All transitions created through normal means (using `perform_access`)
    /// should be possible, but the same is not guaranteed by construction of
    /// transitions inferred by diagnostics. This checks that a transition
    /// reconstructed by diagnostics is indeed one that could happen.
    fn is_possible(self) -> bool {
        self.from <= self.to
    }

    pub fn from(from: Permission, to: Permission) -> Option<Self> {
        let t = Self { from: from.inner, to: to.inner };
        t.is_possible().then_some(t)
    }

    pub fn is_noop(self) -> bool {
        self.from == self.to
    }

    /// Extract result of a transition (checks that the starting point matches).
    pub fn applied(self, starting_point: Permission) -> Option<Permission> {
        (starting_point.inner == self.from).then_some(Permission { inner: self.to })
    }

    /// Extract starting point of a transition
    pub fn started(self) -> Permission {
        Permission { inner: self.from }
    }

    /// Determines if this transition would disable the permission.
    pub fn produces_disabled(self) -> bool {
        self.to == Disabled
    }
}

pub mod diagnostics {
    use super::*;
    impl fmt::Display for PermissionPriv {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "{}",
                match self {
                    Reserved { ty_is_freeze: true, conflicted: false } => "Reserved",
                    Reserved { ty_is_freeze: true, conflicted: true } => "Reserved (conflicted)",
                    Reserved { ty_is_freeze: false, conflicted: false } =>
                        "Reserved (interior mutable)",
                    Reserved { ty_is_freeze: false, conflicted: true } =>
                        "Reserved (interior mutable, conflicted)",
                    Active => "Active",
                    Frozen => "Frozen",
                    Disabled => "Disabled",
                }
            )
        }
    }

    impl fmt::Display for PermTransition {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "from {} to {}", self.from, self.to)
        }
    }

    impl fmt::Display for Permission {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.inner)
        }
    }

    impl Permission {
        /// Abbreviated name of the permission (uniformly 3 letters for nice alignment).
        pub fn short_name(self) -> &'static str {
            // Make sure there are all of the same length as each other
            // and also as `diagnostics::DisplayFmtPermission.uninit` otherwise
            // alignment will be incorrect.
            match self.inner {
                Reserved { ty_is_freeze: true, conflicted: false } => "Rs  ",
                Reserved { ty_is_freeze: true, conflicted: true } => "RsC ",
                Reserved { ty_is_freeze: false, conflicted: false } => "RsM ",
                Reserved { ty_is_freeze: false, conflicted: true } => "RsCM",
                Active => "Act ",
                Frozen => "Frz ",
                Disabled => "Dis ",
            }
        }
    }

    impl PermTransition {
        /// Readable explanation of the consequences of an event.
        /// Fits in the sentence "This accessed caused {trans.summary()}".
        pub fn summary(&self) -> &'static str {
            assert!(self.is_possible());
            match (self.from, self.to) {
                (_, Active) => "the first write to a 2-phase borrowed mutable reference",
                (_, Frozen) => "a loss of write permissions",
                (Reserved { conflicted: false, .. }, Reserved { conflicted: true, .. }) =>
                    "a temporary loss of write permissions until function exit",
                (Frozen, Disabled) => "a loss of read permissions",
                (_, Disabled) => "a loss of read and write permissions",
                (old, new) =>
                    unreachable!("Transition from {old:?} to {new:?} should never be possible"),
            }
        }

        /// Determines whether `self` is a relevant transition for the error `err`.
        /// `self` will be a transition that happened to a tag some time before
        /// that tag caused the error.
        ///
        /// Irrelevant events:
        /// - modifications of write permissions when the error is related to read permissions
        ///   (on failed reads and protected `Frozen -> Disabled`, ignore `Reserved -> Active`,
        ///   `Reserved(conflicted=false) -> Reserved(conflicted=true)`, and `Active -> Frozen`)
        /// - all transitions for attempts to deallocate strongly protected tags
        ///
        /// # Panics
        ///
        /// This function assumes that its arguments apply to the same location
        /// and that they were obtained during a normal execution. It will panic otherwise.
        /// - all transitions involved in `self` and `err` should be increasing
        /// (Reserved < Active < Frozen < Disabled);
        /// - between `self` and `err` the permission should also be increasing,
        /// so all permissions inside `err` should be greater than `self.1`;
        /// - `Active` and `Reserved(conflicted=false)` cannot cause an error
        /// due to insufficient permissions, so `err` cannot be a `ChildAccessForbidden(_)`
        /// of either of them;
        /// - `err` should not be `ProtectedDisabled(Disabled)`, because the protected
        /// tag should not have been `Disabled` in the first place (if this occurs it means
        /// we have unprotected tags that become protected)
        pub(in super::super) fn is_relevant(&self, err: TransitionError) -> bool {
            // NOTE: `super::super` is the visibility of `TransitionError`
            assert!(self.is_possible());
            if self.is_noop() {
                return false;
            }
            match err {
                TransitionError::ChildAccessForbidden(insufficient) => {
                    // Show where the permission was gained then lost,
                    // but ignore unrelated permissions.
                    // This eliminates transitions like `Active -> Frozen`
                    // when the error is a failed `Read`.
                    match (self.to, insufficient.inner) {
                        (Frozen, Frozen) => true,
                        (Active, Frozen) => true,
                        (Disabled, Disabled) => true,
                        (Reserved { conflicted: true, .. }, Reserved { conflicted: true, .. }) =>
                            true,
                        // A pointer being `Disabled` is a strictly stronger source of
                        // errors than it being `Frozen`. If we try to access a `Disabled`,
                        // then where it became `Frozen` (or `Active` or `Reserved`) is the least
                        // of our concerns for now.
                        (Reserved { conflicted: true, .. } | Active | Frozen, Disabled) => false,
                        (Reserved { conflicted: true, .. }, Frozen) => false,

                        // `Active` and `Reserved` have all permissions, so a
                        // `ChildAccessForbidden(Reserved | Active)` can never exist.
                        (_, Active) | (_, Reserved { conflicted: false, .. }) =>
                            unreachable!("this permission cannot cause an error"),
                        // No transition has `Reserved(conflicted=false)` as its `.to` unless it's a noop.
                        (Reserved { conflicted: false, .. }, _) =>
                            unreachable!("self is a noop transition"),
                        // All transitions produced in normal executions (using `apply_access`)
                        // change permissions in the order `Reserved -> Active -> Frozen -> Disabled`.
                        // We assume that the error was triggered on the same location that
                        // the transition `self` applies to, so permissions found must be increasing
                        // in the order `self.from < self.to <= insufficient.inner`
                        (Active | Frozen | Disabled, Reserved { .. }) | (Disabled, Frozen) =>
                            unreachable!("permissions between self and err must be increasing"),
                    }
                }
                TransitionError::ProtectedDisabled(before_disabled) => {
                    // Show how we got to the starting point of the forbidden transition,
                    // but ignore what came before.
                    // This eliminates transitions like `Reserved -> Active`
                    // when the error is a `Frozen -> Disabled`.
                    match (self.to, before_disabled.inner) {
                        // We absolutely want to know where it was activated/frozen/marked
                        // conflicted.
                        (Active, Active) => true,
                        (Frozen, Frozen) => true,
                        (Reserved { conflicted: true, .. }, Reserved { conflicted: true, .. }) =>
                            true,
                        // If the error is a transition `Frozen -> Disabled`, then we don't really
                        // care whether before that was `Reserved -> Active -> Frozen` or
                        // `Frozen` directly.
                        // The error will only show either
                        // - created as Reserved { conflicted: false },
                        //   then Reserved { .. } -> Disabled is forbidden
                        // - created as Reserved { conflicted: false },
                        //   then Active -> Disabled is forbidden
                        // A potential `Reserved { conflicted: false }
                        //   -> Reserved { conflicted: true }` is inexistant or irrelevant,
                        // and so is the `Reserved { conflicted: false } -> Active`
                        (Active, Frozen) => false,
                        (Reserved { conflicted: true, .. }, _) => false,

                        (_, Disabled) =>
                            unreachable!(
                                "permission that results in Disabled should not itself be Disabled in the first place"
                            ),
                        // No transition has `Reserved { conflicted: false }` as its `.to`
                        // unless it's a noop.
                        (Reserved { conflicted: false, .. }, _) =>
                            unreachable!("self is a noop transition"),

                        // Permissions only evolve in the order `Reserved -> Active -> Frozen -> Disabled`,
                        // so permissions found must be increasing in the order
                        // `self.from < self.to <= forbidden.from < forbidden.to`.
                        (Disabled, Reserved { .. } | Active | Frozen)
                        | (Frozen, Reserved { .. } | Active)
                        | (Active, Reserved { .. }) =>
                            unreachable!("permissions between self and err must be increasing"),
                    }
                }
                // We don't care because protectors evolve independently from
                // permissions.
                TransitionError::ProtectedDealloc => false,
            }
        }

        /// Endpoint of a transition.
        /// Meant only for diagnostics, use `applied` in non-diagnostics
        /// code, which also checks that the starting point matches the current state.
        pub fn endpoint(&self) -> Permission {
            Permission { inner: self.to }
        }
    }
}

#[cfg(test)]
impl Permission {
    pub fn is_reserved_with_conflicted(&self, expected_conflicted: bool) -> bool {
        match self.inner {
            Reserved { conflicted, .. } => conflicted == expected_conflicted,
            _ => false,
        }
    }
}

#[cfg(test)]
mod propagation_optimization_checks {
    pub use super::*;
    use crate::borrow_tracker::tree_borrows::exhaustive::{precondition, Exhaustive};

    impl Exhaustive for PermissionPriv {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(
                vec![Active, Frozen, Disabled].into_iter().chain(
                    <[bool; 2]>::exhaustive()
                        .map(|[ty_is_freeze, conflicted]| Reserved { ty_is_freeze, conflicted }),
                ),
            )
        }
    }

    impl Exhaustive for Permission {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(PermissionPriv::exhaustive().map(|inner| Self { inner }))
        }
    }

    impl Exhaustive for AccessKind {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            use AccessKind::*;
            Box::new(vec![Read, Write].into_iter())
        }
    }

    impl Exhaustive for AccessRelatedness {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            use AccessRelatedness::*;
            Box::new(vec![This, StrictChildAccess, AncestorAccess, DistantAccess].into_iter())
        }
    }

    #[test]
    // For any kind of access, if we do it twice the second should be a no-op.
    // Even if the protector has disappeared.
    fn all_transitions_idempotent() {
        use transition::*;
        for old in PermissionPriv::exhaustive() {
            for (old_protected, new_protected) in <(bool, bool)>::exhaustive() {
                // Protector can't appear out of nowhere: either the permission was
                // created with a protector (`old_protected = true`) and it then may
                // or may not lose it (`new_protected = false`, resp. `new_protected = true`),
                // or it didn't have one upon creation and never will
                // (`old_protected = new_protected = false`).
                // We thus eliminate from this test and all other tests
                // the case where the tag is initially unprotected and later becomes protected.
                precondition!(old_protected || !new_protected);
                for (access, rel_pos) in <(AccessKind, AccessRelatedness)>::exhaustive() {
                    if let Some(new) = perform_access(access, rel_pos, old, old_protected) {
                        assert_eq!(
                            new,
                            perform_access(access, rel_pos, new, new_protected).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[rustfmt::skip]
    fn foreign_read_is_noop_after_foreign_write() {
        use transition::*;
        let old_access = AccessKind::Write;
        let new_access = AccessKind::Read;
        for old in PermissionPriv::exhaustive() {
            for [old_protected, new_protected] in <[bool; 2]>::exhaustive() {
                precondition!(old_protected || !new_protected);
                for rel_pos in AccessRelatedness::exhaustive() {
                    precondition!(rel_pos.is_foreign());
                    if let Some(new) = perform_access(old_access, rel_pos, old, old_protected) {
                        assert_eq!(
                            new,
                            perform_access(new_access, rel_pos, new, new_protected).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    // Check that all transitions are consistent with the order on PermissionPriv,
    // i.e. Reserved -> Active -> Frozen -> Disabled
    fn permissionpriv_partialord_is_reachability() {
        let reach = {
            let mut reach = rustc_data_structures::fx::FxHashSet::default();
            // One-step transitions + reflexivity
            for start in PermissionPriv::exhaustive() {
                reach.insert((start, start));
                for (access, rel) in <(AccessKind, AccessRelatedness)>::exhaustive() {
                    for prot in bool::exhaustive() {
                        if let Some(end) = transition::perform_access(access, rel, start, prot) {
                            reach.insert((start, end));
                        }
                    }
                }
            }
            // Transitive closure
            let mut finished = false;
            while !finished {
                finished = true;
                for [start, mid, end] in <[PermissionPriv; 3]>::exhaustive() {
                    if reach.contains(&(start, mid))
                        && reach.contains(&(mid, end))
                        && !reach.contains(&(start, end))
                    {
                        finished = false;
                        reach.insert((start, end));
                    }
                }
            }
            reach
        };
        // Check that it matches `<`
        for [p1, p2] in <[PermissionPriv; 2]>::exhaustive() {
            let le12 = p1 <= p2;
            let reach12 = reach.contains(&(p1, p2));
            assert!(
                le12 == reach12,
                "`{p1} reach {p2}` ({reach12}) does not match `{p1} <= {p2}` ({le12})"
            );
        }
    }
}
