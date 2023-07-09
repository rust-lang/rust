use std::cmp::{Ordering, PartialOrd};
use std::fmt;

use crate::borrow_tracker::tree_borrows::diagnostics::TransitionError;
use crate::borrow_tracker::tree_borrows::tree::AccessRelatedness;
use crate::borrow_tracker::AccessKind;

/// The activation states of a pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionPriv {
    /// represents: a local reference that has not yet been written to;
    /// allows: child reads, foreign reads, foreign writes if type is freeze;
    /// rejects: child writes (Active), foreign writes (Disabled, except if type is not freeze).
    /// special case: behaves differently when protected to adhere more closely to noalias
    Reserved { ty_is_freeze: bool },
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
    /// PermissionPriv is ordered as follows:
    /// - Reserved(_) < Active < Frozen < Disabled;
    /// - different kinds of `Reserved` (with or without interior mutability)
    /// are incomparable to each other.
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
            (Reserved { .. }, Reserved { .. }) => return None,
        })
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
            // accesses, since the data is not being mutated. Hence the `{ .. }`
            readable @ (Reserved { .. } | Active | Frozen) => readable,
        })
    }

    /// A non-child node was read-accessed: noop on non-protected Reserved, advance to Frozen otherwise.
    fn foreign_read(state: PermissionPriv, protected: bool) -> Option<PermissionPriv> {
        use Option::*;
        Some(match state {
            // The inner data `ty_is_freeze` of `Reserved` is always irrelevant for Read
            // accesses, since the data is not being mutated. Hence the `{ .. }`
            res @ Reserved { .. } if !protected => res,
            Reserved { .. } => Frozen, // protected reserved
            Active => Frozen,
            non_writeable @ (Frozen | Disabled) => non_writeable,
        })
    }

    /// A child node was write-accessed: `Reserved` must become `Active` to obtain
    /// write permissions, `Frozen` and `Disabled` cannot obtain such permissions and produce UB.
    fn child_write(state: PermissionPriv, _protected: bool) -> Option<PermissionPriv> {
        Some(match state {
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
            cell @ Reserved { ty_is_freeze: false } if !protected => cell,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Default initial permission of the root of a new tree.
    pub fn new_root() -> Self {
        Self { inner: Active }
    }

    /// Default initial permission of a reborrowed mutable reference.
    pub fn new_unique_2phase(ty_is_freeze: bool) -> Self {
        Self { inner: Reserved { ty_is_freeze } }
    }

    /// Default initial permission for return place.
    pub fn new_active() -> Self {
        Self { inner: Active }
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

    /// Determines whether a transition that occured is compatible with the presence
    /// of a Protector. This is not included in the `transition` functions because
    /// it would distract from the few places where the transition is modified
    /// because of a protector, but not forbidden.
    ///
    /// Note: this is not in charge of checking that there *is* a protector,
    /// it should be used as
    /// ```
    /// let no_protector_error = if is_protected(tag) {
    ///     transition.is_allowed_by_protector()
    /// };
    /// ```
    pub fn is_allowed_by_protector(&self) -> bool {
        assert!(self.is_possible());
        match (self.from, self.to) {
            _ if self.from == self.to => true,
            // It is always a protector violation to not be readable anymore
            (_, Disabled) => false,
            // In the case of a `Reserved` under a protector, both transitions
            // `Reserved => Active` and `Reserved => Frozen` can legitimately occur.
            // The first is standard (Child Write), the second is for Foreign Writes
            // on protected Reserved where we must ensure that the pointer is not
            // written to in the future.
            (Reserved { .. }, Active) | (Reserved { .. }, Frozen) => true,
            // This pointer should have stayed writeable for the whole function
            (Active, Frozen) => false,
            _ => unreachable!("Transition {} should never be possible", self),
        }
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
                    Reserved { .. } => "Reserved",
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
                Reserved { ty_is_freeze: true } => "Res",
                Reserved { ty_is_freeze: false } => "Re*",
                Active => "Act",
                Frozen => "Frz",
                Disabled => "Dis",
            }
        }
    }

    impl PermTransition {
        /// Readable explanation of the consequences of an event.
        /// Fits in the sentence "This accessed caused {trans.summary()}".
        ///
        /// Important: for the purposes of this explanation, `Reserved` is considered
        /// to have write permissions, because that's what the diagnostics care about
        /// (otherwise `Reserved -> Frozen` would be considered a noop).
        pub fn summary(&self) -> &'static str {
            assert!(self.is_possible());
            match (self.from, self.to) {
                (_, Active) => "the first write to a 2-phase borrowed mutable reference",
                (_, Frozen) => "a loss of write permissions",
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
        ///   `Reserved -> Frozen`, and `Active -> Frozen`)
        /// - all transitions for attempts to deallocate strongly protected tags
        ///
        /// # Panics
        ///
        /// This function assumes that its arguments apply to the same location
        /// and that they were obtained during a normal execution. It will panic otherwise.
        /// - `err` cannot be a `ProtectedTransition(_)` of a noop transition, as those
        /// never trigger protectors;
        /// - all transitions involved in `self` and `err` should be increasing
        /// (Reserved < Active < Frozen < Disabled);
        /// - between `self` and `err` the permission should also be increasing,
        /// so all permissions inside `err` should be greater than `self.1`;
        /// - `Active` and `Reserved` cannot cause an error due to insufficient permissions,
        /// so `err` cannot be a `ChildAccessForbidden(_)` of either of them;
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
                        // A pointer being `Disabled` is a strictly stronger source of
                        // errors than it being `Frozen`. If we try to access a `Disabled`,
                        // then where it became `Frozen` (or `Active`) is the least of our concerns for now.
                        (Active | Frozen, Disabled) => false,

                        // `Active` and `Reserved` have all permissions, so a
                        // `ChildAccessForbidden(Reserved | Active)` can never exist.
                        (_, Active) | (_, Reserved { .. }) =>
                            unreachable!("this permission cannot cause an error"),
                        // No transition has `Reserved` as its `.to` unless it's a noop.
                        (Reserved { .. }, _) => unreachable!("self is a noop transition"),
                        // All transitions produced in normal executions (using `apply_access`)
                        // change permissions in the order `Reserved -> Active -> Frozen -> Disabled`.
                        // We assume that the error was triggered on the same location that
                        // the transition `self` applies to, so permissions found must be increasing
                        // in the order `self.from < self.to <= insufficient.inner`
                        (Disabled, Frozen) =>
                            unreachable!("permissions between self and err must be increasing"),
                    }
                }
                TransitionError::ProtectedTransition(forbidden) => {
                    assert!(!forbidden.is_noop());
                    // Show how we got to the starting point of the forbidden transition,
                    // but ignore what came before.
                    // This eliminates transitions like `Reserved -> Active`
                    // when the error is a `Frozen -> Disabled`.
                    match (self.to, forbidden.from, forbidden.to) {
                        // We absolutely want to know where it was activated.
                        (Active, Active, Frozen | Disabled) => true,
                        // And knowing where it became Frozen is also important.
                        (Frozen, Frozen, Disabled) => true,
                        // If the error is a transition `Frozen -> Disabled`, then we don't really
                        // care whether before that was `Reserved -> Active -> Frozen` or
                        // `Reserved -> Frozen` or even `Frozen` directly.
                        // The error will only show either
                        // - created as Frozen, then Frozen -> Disabled is forbidden
                        // - created as Reserved, later became Frozen, then Frozen -> Disabled is forbidden
                        // In both cases the `Reserved -> Active` part is inexistant or irrelevant.
                        (Active, Frozen, Disabled) => false,

                        // `Reserved -> Frozen` does not trigger protectors.
                        (_, Reserved { .. }, Frozen) =>
                            unreachable!("this transition cannot cause an error"),
                        // No transition has `Reserved` as its `.to` unless it's a noop.
                        (Reserved { .. }, _, _) => unreachable!("self is a noop transition"),
                        (_, Disabled, Disabled) | (_, Frozen, Frozen) | (_, Active, Active) =>
                            unreachable!("err contains a noop transition"),

                        // Permissions only evolve in the order `Reserved -> Active -> Frozen -> Disabled`,
                        // so permissions found must be increasing in the order
                        // `self.from < self.to <= forbidden.from < forbidden.to`.
                        (Disabled, Reserved { .. } | Active | Frozen, _)
                        | (Frozen, Reserved { .. } | Active, _)
                        | (Active, Reserved { .. }, _) =>
                            unreachable!("permissions between self and err must be increasing"),
                        (_, Disabled, Reserved { .. } | Active | Frozen)
                        | (_, Frozen, Reserved { .. } | Active)
                        | (_, Active, Reserved { .. }) =>
                            unreachable!("permissions within err must be increasing"),
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
mod propagation_optimization_checks {
    pub use super::*;

    mod util {
        pub use super::*;
        impl PermissionPriv {
            /// Enumerate all states
            pub fn all() -> impl Iterator<Item = PermissionPriv> {
                vec![
                    Active,
                    Reserved { ty_is_freeze: true },
                    Reserved { ty_is_freeze: false },
                    Frozen,
                    Disabled,
                ]
                .into_iter()
            }
        }

        impl AccessKind {
            /// Enumerate all AccessKind.
            pub fn all() -> impl Iterator<Item = AccessKind> {
                use AccessKind::*;
                [Read, Write].into_iter()
            }
        }

        impl AccessRelatedness {
            /// Enumerate all relative positions
            pub fn all() -> impl Iterator<Item = AccessRelatedness> {
                use AccessRelatedness::*;
                [This, StrictChildAccess, AncestorAccess, DistantAccess].into_iter()
            }
        }
    }

    #[test]
    // For any kind of access, if we do it twice the second should be a no-op.
    // Even if the protector has disappeared.
    fn all_transitions_idempotent() {
        use transition::*;
        for old in PermissionPriv::all() {
            for (old_protected, new_protected) in [(true, true), (true, false), (false, false)] {
                for access in AccessKind::all() {
                    for rel_pos in AccessRelatedness::all() {
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
    }

    #[test]
    fn foreign_read_is_noop_after_write() {
        use transition::*;
        let old_access = AccessKind::Write;
        let new_access = AccessKind::Read;
        for old in PermissionPriv::all() {
            for (old_protected, new_protected) in [(true, true), (true, false), (false, false)] {
                for rel_pos in AccessRelatedness::all().filter(|rel| rel.is_foreign()) {
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
    fn access_transitions_progress_increasing() {
        use transition::*;
        for old in PermissionPriv::all() {
            for protected in [true, false] {
                for access in AccessKind::all() {
                    for rel_pos in AccessRelatedness::all() {
                        if let Some(new) = perform_access(access, rel_pos, old, protected) {
                            assert!(old <= new);
                        }
                    }
                }
            }
        }
    }
}
