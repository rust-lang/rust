use std::cmp::{Ordering, PartialOrd};
use std::fmt;

use crate::AccessKind;
use crate::borrow_tracker::tree_borrows::diagnostics::TransitionError;
use crate::borrow_tracker::tree_borrows::tree::AccessRelatedness;

/// The activation states of a pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PermissionPriv {
    /// represents: a local mutable reference that has not yet been written to;
    /// allows: child reads, foreign reads;
    /// affected by: child writes (becomes Active),
    /// rejects: foreign writes (Disabled).
    ///
    /// `ReservedFrz` is mostly for types that are `Freeze` (no interior mutability).
    /// If the type has interior mutability, see `ReservedIM` instead.
    /// (Note: since the discovery of `tests/fail/tree_borrows/reservedim_spurious_write.rs`,
    /// we also use `ReservedFreeze` for mutable references that were retagged with a protector
    /// independently of interior mutability)
    ///
    /// special case: behaves differently when protected, which is where `conflicted`
    /// is relevant
    /// - `conflicted` is set on foreign reads,
    /// - `conflicted` must not be set on child writes (there is UB otherwise).
    ///
    /// This is so that the behavior of `Reserved` adheres to the rules of `noalias`:
    /// - foreign-read then child-write is UB due to `conflicted`,
    /// - child-write then foreign-read is UB since child-write will activate and then
    ///   foreign-read disables a protected `Active`, which is UB.
    ReservedFrz { conflicted: bool },
    /// Alternative version of `ReservedFrz` made for types with interior mutability.
    /// allows: child reads, foreign reads, foreign writes (extra);
    /// affected by: child writes (becomes Active);
    /// rejects: nothing.
    ReservedIM,
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
use self::PermissionPriv::*;

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
            (ReservedIM, ReservedIM) => Equal,
            (ReservedFrz { conflicted: c1 }, ReservedFrz { conflicted: c2 }) => {
                // `bool` is ordered such that `false <= true`, so this works as intended.
                c1.cmp(c2)
            }
            // Versions of `Reserved` with different interior mutability are incomparable with each
            // other.
            (ReservedIM, ReservedFrz { .. }) | (ReservedFrz { .. }, ReservedIM) => return None,
        })
    }
}

impl PermissionPriv {
    /// Check if `self` can be the initial state of a pointer.
    fn is_initial(&self) -> bool {
        matches!(self, ReservedFrz { conflicted: false } | Frozen | ReservedIM)
    }

    /// Reject `ReservedIM` that cannot exist in the presence of a protector.
    fn compatible_with_protector(&self) -> bool {
        !matches!(self, ReservedIM)
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
            readable @ (ReservedFrz { .. } | ReservedIM | Active | Frozen) => readable,
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
            ReservedFrz { .. } if protected => ReservedFrz { conflicted: true },
            // Before activation and without protectors, foreign reads are fine.
            // That's the entire point of 2-phase borrows.
            res @ (ReservedFrz { .. } | ReservedIM) => {
                // Even though we haven't checked `ReservedIM if protected` separately,
                // it is a state that cannot occur because under a protector we only
                // create `ReservedFrz` never `ReservedIM`.
                assert!(!protected);
                res
            }
            Active =>
                if protected {
                    // We wrote, someone else reads -- that's bad.
                    // (Since Active is always initialized, this move-to-protected will mean insta-UB.)
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
            ReservedFrz { conflicted: true } if protected => return None,
            // A write always activates the 2-phase borrow, even with interior
            // mutability
            ReservedFrz { .. } | ReservedIM | Active => Active,
            Frozen | Disabled => return None,
        })
    }

    /// A non-child node was write-accessed: this makes everything `Disabled` except for
    /// non-protected interior mutable `Reserved` which stay the same.
    fn foreign_write(state: PermissionPriv, protected: bool) -> Option<PermissionPriv> {
        // There is no explicit dependency on `protected`, but recall that interior mutable
        // types receive a `ReservedFrz` instead of `ReservedIM` when retagged under a protector,
        // so the result of this function does indirectly depend on (past) protector status.
        Some(match state {
            res @ ReservedIM => {
                // We can never create a `ReservedIM` under a protector, only `ReservedFrz`.
                assert!(!protected);
                res
            }
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
    /// Check if `self` is the terminal state of a pointer (is `Disabled`).
    pub fn is_disabled(&self) -> bool {
        self.inner == Disabled
    }
    /// Check if `self` is the post-child-write state of a pointer (is `Active`).
    pub fn is_active(&self) -> bool {
        self.inner == Active
    }

    /// Default initial permission of the root of a new tree at inbounds positions.
    /// Must *only* be used for the root, this is not in general an "initial" permission!
    pub fn new_active() -> Self {
        Self { inner: Active }
    }

    /// Default initial permission of a reborrowed mutable reference that is either
    /// protected or not interior mutable.
    fn new_reserved_frz() -> Self {
        Self { inner: ReservedFrz { conflicted: false } }
    }

    /// Default initial permission of an unprotected interior mutable reference.
    fn new_reserved_im() -> Self {
        Self { inner: ReservedIM }
    }

    /// Wrapper around `new_reserved_frz` and `new_reserved_im` that decides
    /// which to call based on the interior mutability and the retag kind (whether there
    /// is a protector is relevant because being protected takes priority over being
    /// interior mutable)
    pub fn new_reserved(ty_is_freeze: bool, protected: bool) -> Self {
        if ty_is_freeze || protected { Self::new_reserved_frz() } else { Self::new_reserved_im() }
    }

    /// Default initial permission of a reborrowed shared reference.
    pub fn new_frozen() -> Self {
        Self { inner: Frozen }
    }

    /// Default initial permission of  the root of a new tree at out-of-bounds positions.
    /// Must *only* be used for the root, this is not in general an "initial" permission!
    pub fn new_disabled() -> Self {
        Self { inner: Disabled }
    }

    /// Reject `ReservedIM` that cannot exist in the presence of a protector.
    pub fn compatible_with_protector(&self) -> bool {
        self.inner.compatible_with_protector()
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

    /// During a provenance GC, we want to compact the tree.
    /// For this, we want to merge nodes upwards if they have a singleton parent.
    /// But we need to be careful: If the parent is Frozen, and the child is Reserved,
    /// we can not do such a merge. In general, such a merge is possible if the parent
    /// allows similar accesses, and in particular if the parent never causes UB on its
    /// own. This is enforced by a test, namely `tree_compacting_is_sound`. See that
    /// test for more information.
    /// This method is only sound if the parent is not protected. We never attempt to
    /// remove protected parents.
    pub fn can_be_replaced_by_child(self, child: Self) -> bool {
        match (self.inner, child.inner) {
            // ReservedIM can be replaced by anything, as it allows all
            // transitions.
            (ReservedIM, _) => true,
            // Reserved (as parent, where conflictedness does not matter)
            // can be replaced by all but ReservedIM,
            // since ReservedIM alone would survive foreign writes
            (ReservedFrz { .. }, ReservedIM) => false,
            (ReservedFrz { .. }, _) => true,
            // Active can not be replaced by something surviving
            // foreign reads and then remaining writable.
            (Active, ReservedIM) => false,
            (Active, ReservedFrz { .. }) => false,
            // Replacing a state by itself is always okay, even if the child state is protected.
            (Active, Active) => true,
            // Active can be replaced by Frozen, since it is not protected.
            (Active, Frozen) => true,
            (Active, Disabled) => true,
            // Frozen can only be replaced by Disabled (and itself).
            (Frozen, Frozen) => true,
            (Frozen, Disabled) => true,
            (Frozen, _) => false,
            // Disabled can not be replaced by anything else.
            (Disabled, Disabled) => true,
            (Disabled, _) => false,
        }
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
            write!(f, "{}", match self {
                ReservedFrz { conflicted: false } => "Reserved",
                ReservedFrz { conflicted: true } => "Reserved (conflicted)",
                ReservedIM => "Reserved (interior mutable)",
                Active => "Active",
                Frozen => "Frozen",
                Disabled => "Disabled",
            })
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
                ReservedFrz { conflicted: false } => "Res ",
                ReservedFrz { conflicted: true } => "ResC",
                ReservedIM => "ReIM",
                Active => "Act ",
                Frozen => "Frz ",
                Disabled => "Dis ",
            }
        }
    }

    impl PermTransition {
        /// Readable explanation of the consequences of an event.
        /// Fits in the sentence "This transition corresponds to {trans.summary()}".
        pub fn summary(&self) -> &'static str {
            assert!(self.is_possible());
            assert!(!self.is_noop());
            match (self.from, self.to) {
                (_, Active) => "the first write to a 2-phase borrowed mutable reference",
                (_, Frozen) => "a loss of write permissions",
                (ReservedFrz { conflicted: false }, ReservedFrz { conflicted: true }) =>
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
        ///   (Reserved < Active < Frozen < Disabled);
        /// - between `self` and `err` the permission should also be increasing,
        ///   so all permissions inside `err` should be greater than `self.1`;
        /// - `Active` and `Reserved(conflicted=false)` cannot cause an error
        ///   due to insufficient permissions, so `err` cannot be a `ChildAccessForbidden(_)`
        ///   of either of them;
        /// - `err` should not be `ProtectedDisabled(Disabled)`, because the protected
        ///   tag should not have been `Disabled` in the first place (if this occurs it means
        ///   we have unprotected tags that become protected)
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
                        (
                            ReservedFrz { conflicted: true, .. },
                            ReservedFrz { conflicted: true, .. },
                        ) => true,
                        // A pointer being `Disabled` is a strictly stronger source of
                        // errors than it being `Frozen`. If we try to access a `Disabled`,
                        // then where it became `Frozen` (or `Active` or `Reserved`) is the least
                        // of our concerns for now.
                        (ReservedFrz { conflicted: true } | Active | Frozen, Disabled) => false,
                        (ReservedFrz { conflicted: true }, Frozen) => false,

                        // `Active` and `Reserved` have all permissions, so a
                        // `ChildAccessForbidden(Reserved | Active)` can never exist.
                        (_, Active) | (_, ReservedFrz { conflicted: false }) =>
                            unreachable!("this permission cannot cause an error"),
                        // No transition has `Reserved { conflicted: false }` or `ReservedIM`
                        // as its `.to` unless it's a noop.
                        (ReservedFrz { conflicted: false } | ReservedIM, _) =>
                            unreachable!("self is a noop transition"),
                        // All transitions produced in normal executions (using `apply_access`)
                        // change permissions in the order `Reserved -> Active -> Frozen -> Disabled`.
                        // We assume that the error was triggered on the same location that
                        // the transition `self` applies to, so permissions found must be increasing
                        // in the order `self.from < self.to <= insufficient.inner`
                        (Active | Frozen | Disabled, ReservedFrz { .. } | ReservedIM)
                        | (Disabled, Frozen)
                        | (ReservedFrz { .. }, ReservedIM) =>
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
                        (
                            ReservedFrz { conflicted: true, .. },
                            ReservedFrz { conflicted: true, .. },
                        ) => true,
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
                        (ReservedFrz { conflicted: true }, _) => false,

                        (_, Disabled) =>
                            unreachable!(
                                "permission that results in Disabled should not itself be Disabled in the first place"
                            ),
                        // No transition has `Reserved { conflicted: false }` or `ReservedIM` as its `.to`
                        // unless it's a noop.
                        (ReservedFrz { conflicted: false } | ReservedIM, _) =>
                            unreachable!("self is a noop transition"),

                        // Permissions only evolve in the order `Reserved -> Active -> Frozen -> Disabled`,
                        // so permissions found must be increasing in the order
                        // `self.from < self.to <= forbidden.from < forbidden.to`.
                        (Disabled, ReservedFrz { .. } | ReservedIM | Active | Frozen)
                        | (Frozen, ReservedFrz { .. } | ReservedIM | Active)
                        | (Active, ReservedFrz { .. } | ReservedIM) =>
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
    pub fn is_reserved_frz_with_conflicted(&self, expected_conflicted: bool) -> bool {
        match self.inner {
            ReservedFrz { conflicted } => conflicted == expected_conflicted,
            _ => false,
        }
    }
}

#[cfg(test)]
mod propagation_optimization_checks {
    pub use super::*;
    use crate::borrow_tracker::tree_borrows::exhaustive::{Exhaustive, precondition};

    impl Exhaustive for PermissionPriv {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(
                vec![Active, Frozen, Disabled, ReservedIM]
                    .into_iter()
                    .chain(<bool>::exhaustive().map(|conflicted| ReservedFrz { conflicted })),
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
                if old_protected {
                    precondition!(old.compatible_with_protector());
                }
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
                if old_protected {
                    precondition!(old.compatible_with_protector());
                }
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
                        if prot {
                            precondition!(start.compatible_with_protector());
                        }
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
