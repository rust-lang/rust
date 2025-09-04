//! Tests for the tree
#![cfg(test)]

use std::fmt;

use super::*;
use crate::borrow_tracker::tree_borrows::exhaustive::{Exhaustive, precondition};

impl Exhaustive for LocationState {
    fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
        // We keep `latest_foreign_access` at `None` as that's just a cache.
        Box::new(<(Permission, bool)>::exhaustive().map(|(permission, accessed)| {
            Self {
                permission,
                accessed,
                idempotent_foreign_access: IdempotentForeignAccess::default(),
            }
        }))
    }
}

impl LocationState {
    /// Ensure that the current internal state can exist at the same time as a protector.
    /// In practice this only eliminates `ReservedIM` that is never used in the presence
    /// of a protector (we instead emit `ReservedFrz` on retag).
    pub fn compatible_with_protector(&self) -> bool {
        self.permission.compatible_with_protector()
    }
}

#[test]
#[rustfmt::skip]
// Exhaustive check that for any starting configuration loc,
// for any two read accesses r1 and r2, if `loc + r1 + r2` is not UB
// and results in `loc'`, then `loc + r2 + r1` is also not UB and results
// in the same final state `loc'`.
// This lets us justify arbitrary read-read reorderings.
fn all_read_accesses_commute() {
    let kind = AccessKind::Read;
    // Two of the four combinations of `AccessRelatedness` are trivial,
    // but we might as well check them all.
    for [rel1, rel2] in <[AccessRelatedness; 2]>::exhaustive() {
        // Any protector state works, but we can't move reads across function boundaries
        // so the two read accesses occur under the same protector.
        for protected in bool::exhaustive() {
            for loc in LocationState::exhaustive() {
                if protected {
                    precondition!(loc.compatible_with_protector());
                }
                // Apply 1 then 2. Failure here means that there is UB in the source
                // and we skip the check in the target.
                let mut loc12 = loc;
                precondition!(loc12.perform_access(kind, rel1, protected).is_ok());
                precondition!(loc12.perform_access(kind, rel2, protected).is_ok());

                // If 1 followed by 2 succeeded, then 2 followed by 1 must also succeed...
                let mut loc21 = loc;
                loc21.perform_access(kind, rel2, protected).unwrap();
                loc21.perform_access(kind, rel1, protected).unwrap();

                // ... and produce the same final result.
                assert_eq!(
                    loc12, loc21,
                    "Read accesses {rel1:?} followed by {rel2:?} do not commute !"
                );
            }
        }
    }
}

fn as_foreign_or_child(related: AccessRelatedness) -> &'static str {
    if related.is_foreign() { "foreign" } else { "child" }
}

fn as_protected(b: bool) -> &'static str {
    if b { " (protected)" } else { "" }
}

fn as_lazy_or_accessed(b: bool) -> &'static str {
    if b { "accessed" } else { "lazy" }
}

/// Test that tree compacting (as performed by the GC) is sound.
/// Specifically, the GC will replace a parent by a child if the parent is not
/// protected, and if `can_be_replaced_by_child(parent, child)` is true.
/// To check that this is sound, the function must be a simulation, i.e.
/// if both are accessed, the results must still be in simulation, and also
/// if an access is UB, it must also be UB if done only at the child.
#[test]
fn tree_compacting_is_sound() {
    // The parent is unprotected
    let parent_protected = false;
    for ([parent, child], child_protected) in <([LocationState; 2], bool)>::exhaustive() {
        if child_protected {
            precondition!(child.compatible_with_protector())
        }
        precondition!(parent.permission().can_be_replaced_by_child(child.permission()));
        for (kind, rel) in <(AccessKind, AccessRelatedness)>::exhaustive() {
            let new_parent = parent.perform_access_no_fluff(kind, rel, parent_protected);
            let new_child = child.perform_access_no_fluff(kind, rel, child_protected);
            match (new_parent, new_child) {
                (Some(np), Some(nc)) => {
                    assert!(
                        np.permission().can_be_replaced_by_child(nc.permission()),
                        "`can_be_replaced_by_child` is not a simulation: on a {} {} to a {} parent and a {} {}{} child, the parent becomes {}, the child becomes {}, and these are not in simulation!",
                        as_foreign_or_child(rel),
                        kind,
                        parent.permission(),
                        as_lazy_or_accessed(child.is_accessed()),
                        child.permission(),
                        as_protected(child_protected),
                        np.permission(),
                        nc.permission()
                    )
                }
                (_, None) => {
                    // the child produced UB, this is fine no matter what the parent does
                }
                (None, Some(nc)) => {
                    panic!(
                        "`can_be_replaced_by_child` does not have the UB property: on a {} {} to a(n) {} parent and a(n) {} {}{} child, only the parent causes UB, while the child becomes {}, and it is not allowed for only the parent to cause UB!",
                        as_foreign_or_child(rel),
                        kind,
                        parent.permission(),
                        as_lazy_or_accessed(child.is_accessed()),
                        child.permission(),
                        as_protected(child_protected),
                        nc.permission()
                    )
                }
            }
        }
    }
}

#[test]
#[rustfmt::skip]
// Ensure that of 2 accesses happen, one foreign and one a child, and we are protected, that we
// get UB unless they are both reads.
fn protected_enforces_noalias() {
    for [rel1, rel2] in <[AccessRelatedness; 2]>::exhaustive() {
        // We want to check pairs of accesses where one is foreign and one is not.
        precondition!(rel1.is_foreign() != rel2.is_foreign());
        for [kind1, kind2] in <[AccessKind; 2]>::exhaustive() {
            let protected = true;
            for mut state in LocationState::exhaustive().filter(|s| s.compatible_with_protector()) {
                precondition!(state.perform_access(kind1, rel1, protected).is_ok());
                precondition!(state.perform_access(kind2, rel2, protected).is_ok());
                // If these were both allowed, it must have been two reads.
                assert!(
                    kind1 == AccessKind::Read && kind2 == AccessKind::Read,
                    "failed to enforce noalias between two accesses that are not both reads"
                );
            }
        }
    }
}

/// We are going to exhaustively test the possibility of inserting
/// a spurious read in some code.
///
/// We choose some pointer `x` through which we want a spurious read to be inserted.
/// `x` must thus be reborrowed, not have any children, and initially start protected.
///
/// To check if inserting a spurious read is possible, we observe the behavior
/// of some pointer `y` different from `x` (possibly from a different thread, thus
/// the protectors on `x` and `y` are not necessarily well-nested).
/// It must be the case that no matter the context, the insertion of a spurious read
/// through `x` does not introduce UB in code that did not already have UB.
///
/// Testing this will need some setup to simulate the evolution of the permissions
/// of `x` and `y` under arbitrary code. This arbitrary code of course includes
/// read and write accesses through `x` and `y`, but it must also consider
/// the less obvious:
/// - accesses through pointers that are *neither* `x` nor `y`,
/// - retags of `y` that change its relative position to `x`.
///
///
/// The general code pattern thus looks like
///     [thread 1]             || [thread 2]
///                            || y exists
///     retag x (protect)      ||
///                      arbitrary code
///                           read/write x/y/other
///                        or retag y
///                        or unprotect y
///     <spurious read x>      ||
///                      arbitrary code
///                           read/write x/y/other
///                        or retag y
///                        or unprotect y
///                        or unprotect x
///
/// `x` must still be protected at the moment the spurious read is inserted
/// because spurious reads are impossible in general on unprotected tags.
mod spurious_read {
    use super::*;

    /// Accessed pointer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PtrSelector {
        X,
        Y,
        Other,
    }

    /// Relative position of `x` and `y`.
    /// `y` cannot be a child of `x` because `x` gets retagged as the first
    /// thing in the pattern, so it cannot have children.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum RelPosXY {
        MutuallyForeign,
        /// X is a child of Y.
        XChildY,
    }

    impl Exhaustive for PtrSelector {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            use PtrSelector::*;
            Box::new(vec![X, Y, Other].into_iter())
        }
    }

    impl fmt::Display for PtrSelector {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                PtrSelector::X => write!(f, "x"),
                PtrSelector::Y => write!(f, "y"),
                PtrSelector::Other => write!(f, "z"),
            }
        }
    }

    impl Exhaustive for RelPosXY {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            use RelPosXY::*;
            Box::new(vec![MutuallyForeign, XChildY].into_iter())
        }
    }

    impl fmt::Display for RelPosXY {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                RelPosXY::MutuallyForeign => write!(f, "x and y are mutually foreign"),
                RelPosXY::XChildY => write!(f, "x is a child of y"),
            }
        }
    }

    impl PtrSelector {
        /// Knowing the relative position of `x` to `y`, determine the relative
        /// position of the accessed pointer defined by `self` relative to each `x`
        /// and `y`.
        ///
        /// The output is not necessarily well-defined in general, but it
        /// is unique when considered up to equivalence by `AccessRelatedness::is_foreign`
        /// (e.g. having `RelPosXY::XChildY` and `PtrSelector::Other`, strictly
        /// speaking it is impossible to determine if `Other` is a `DistantAccess`
        /// or an `AncestorAccess` relative to `y`, but it doesn't really matter
        /// because `DistantAccess.is_foreign() == AncestorAccess.is_foreign()`).
        fn rel_pair(self, xy_rel: RelPosXY) -> (AccessRelatedness, AccessRelatedness) {
            use AccessRelatedness::*;
            match xy_rel {
                RelPosXY::MutuallyForeign =>
                    match self {
                        PtrSelector::X => (This, CousinAccess),
                        PtrSelector::Y => (CousinAccess, This),
                        PtrSelector::Other => (CousinAccess, CousinAccess),
                    },
                RelPosXY::XChildY =>
                    match self {
                        PtrSelector::X => (This, StrictChildAccess),
                        PtrSelector::Y => (AncestorAccess, This),
                        PtrSelector::Other => (CousinAccess, CousinAccess),
                    },
            }
        }
    }

    /// Arbitrary access parametrized by the relative position of `x` and `y`
    /// to each other.
    #[derive(Debug, Clone, Copy)]
    struct TestAccess {
        ptr: PtrSelector,
        kind: AccessKind,
    }

    impl Exhaustive for TestAccess {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(
                <(PtrSelector, AccessKind)>::exhaustive()
                    .map(|(ptr, kind)| TestAccess { ptr, kind }),
            )
        }
    }

    impl fmt::Display for TestAccess {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let kind_text = match self.kind {
                AccessKind::Read => "read",
                AccessKind::Write => "write",
            };
            write!(f, "{kind_text} {}", self.ptr)
        }
    }

    type AllowRet = ();
    type NoRet = !;
    #[derive(Clone)]
    /// Events relevant to the evolution of 2 pointers are
    /// - any access to the same location
    /// - end of one of them being protected
    /// - a retag that would change their relative position
    ///
    /// The type `TestEvent` models these kinds of events.
    ///
    /// In order to prevent `x` or `y` from losing their protector,
    /// choose a type `RetX` or `RetY` that is not inhabited.
    /// e.g.
    /// - `TestEvent<AllowRet, AllowRet>` is any event including end of protector on either `x` or `y`
    /// - `TestEvent<NoRet, NoRet>` is any access
    /// - `TestEvent<NoRet, AllowRet>` allows for `y` to lose its protector but not `x`
    enum TestEvent<RetX, RetY> {
        Access(TestAccess),
        RetX(RetX),
        RetY(RetY),
        /// The inner `LocStateProt` must be an initial state (as per the `is_initial` function)
        RetagY(LocStateProt),
    }

    impl<RetX, RetY> Exhaustive for TestEvent<RetX, RetY>
    where
        RetX: Exhaustive + 'static,
        RetY: Exhaustive + 'static,
    {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(
                <TestAccess>::exhaustive()
                    .map(|acc| Self::Access(acc))
                    .chain(RetX::exhaustive().map(|retx| Self::RetX(retx)))
                    .chain(RetY::exhaustive().map(|rety| Self::RetY(rety)))
                    .chain(
                        LocStateProt::exhaustive()
                            .filter_map(|s| s.is_initial().then_some(Self::RetagY(s))),
                    ),
            )
        }
    }

    impl<RetX, RetY> fmt::Display for TestEvent<RetX, RetY> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TestEvent::Access(acc) => write!(f, "{acc}"),
                // The fields of the `Ret` variants just serve to make them
                // impossible to instantiate via the `RetX = NoRet` type; we can
                // always ignore their value.
                TestEvent::RetX(_) => write!(f, "ret x"),
                TestEvent::RetY(_) => write!(f, "ret y"),
                TestEvent::RetagY(newp) => write!(f, "retag y ({newp})"),
            }
        }
    }

    #[derive(Clone, PartialEq, Eq, Hash)]
    /// The state of a pointer on a location, including the protector.
    /// It is represented here with the protector bound to the `LocationState` rather
    /// than the `Map<Location, LocationState>` as is normally the case,
    /// but since all our exhaustive tests look at a  single location
    /// there's no risk of `prot` for different locations of the same tag getting
    /// out of sync.
    struct LocStateProt {
        state: LocationState,
        prot: bool,
    }

    impl LocStateProt {
        fn is_initial(&self) -> bool {
            self.state.is_initial()
        }

        fn perform_access(&self, kind: AccessKind, rel: AccessRelatedness) -> Result<Self, ()> {
            let mut state = self.state;
            state.perform_access(kind, rel, self.prot).map_err(|_| ())?;
            Ok(Self { state, prot: self.prot })
        }

        /// Remove the protector.
        /// This does not perform the implicit read on function exit because
        /// `LocStateProt` does not have enough context to apply its effect.
        fn end_protector(&self) -> Self {
            Self { prot: false, state: self.state }
        }
    }

    impl Exhaustive for LocStateProt {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(
                <(LocationState, bool)>::exhaustive().map(|(state, prot)| Self { state, prot }),
            )
        }
    }

    impl fmt::Display for LocStateProt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.state)?;
            if self.prot {
                write!(f, ", protect")?;
            }
            Ok(())
        }
    }

    #[derive(Clone, PartialEq, Eq, Hash)]
    /// The states of two pointers to the same location,
    /// and their relationship to each other in the tree.
    ///
    /// Note that the two states interact: using one pointer can have
    /// an impact on the other.
    /// This makes `LocStateProtPair` more meaningful than a simple
    /// `(LocStateProt, LocStateProt)` where the two states are not guaranteed
    /// to be updated at the same time.
    /// Some `LocStateProtPair` may be unreachable through normal means
    /// such as `x: Active, y: Active` in the case of mutually foreign pointers.
    struct LocStateProtPair {
        xy_rel: RelPosXY,
        x: LocStateProt,
        y: LocStateProt,
    }

    impl LocStateProtPair {
        fn perform_test_access(self, acc: &TestAccess) -> Result<Self, ()> {
            let (xrel, yrel) = acc.ptr.rel_pair(self.xy_rel);
            let x = self.x.perform_access(acc.kind, xrel)?;
            let y = self.y.perform_access(acc.kind, yrel)?;
            Ok(Self { x, y, ..self })
        }

        /// Perform a read on the given pointer if its state is `accessed`.
        /// Must be called just after reborrowing a pointer, and just after
        /// removing a protector.
        fn read_if_accessed(self, ptr: PtrSelector) -> Result<Self, ()> {
            let accessed = match ptr {
                PtrSelector::X => self.x.state.accessed,
                PtrSelector::Y => self.y.state.accessed,
                PtrSelector::Other =>
                    panic!(
                        "the `accessed` status of `PtrSelector::Other` is unknown, do not pass it to `read_if_accessed`"
                    ),
            };
            if accessed {
                self.perform_test_access(&TestAccess { ptr, kind: AccessKind::Read })
            } else {
                Ok(self)
            }
        }

        /// Remove the protector of `x`, including the implicit read on function exit.
        fn end_protector_x(self) -> Result<Self, ()> {
            let x = self.x.end_protector();
            Self { x, ..self }.read_if_accessed(PtrSelector::X)
        }

        /// Remove the protector of `y`, including the implicit read on function exit.
        fn end_protector_y(self) -> Result<Self, ()> {
            let y = self.y.end_protector();
            Self { y, ..self }.read_if_accessed(PtrSelector::Y)
        }

        fn retag_y(self, new_y: LocStateProt) -> Result<Self, ()> {
            assert!(new_y.is_initial());
            if new_y.prot && !new_y.state.compatible_with_protector() {
                return Err(());
            }
            // `xy_rel` changes to "mutually foreign" now: `y` can no longer be a parent of `x`.
            Self { y: new_y, xy_rel: RelPosXY::MutuallyForeign, ..self }
                .read_if_accessed(PtrSelector::Y)
        }

        fn perform_test_event<RetX, RetY>(self, evt: &TestEvent<RetX, RetY>) -> Result<Self, ()> {
            match evt {
                TestEvent::Access(acc) => self.perform_test_access(acc),
                // The fields of the `Ret` variants just serve to make them
                // impossible to instantiate via the `RetX = NoRet` type; we can
                // always ignore their value.
                TestEvent::RetX(_) => self.end_protector_x(),
                TestEvent::RetY(_) => self.end_protector_y(),
                TestEvent::RetagY(newp) => self.retag_y(newp.clone()),
            }
        }
    }

    impl Exhaustive for LocStateProtPair {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            Box::new(<[LocStateProt; 2]>::exhaustive().flat_map(|[x, y]| {
                RelPosXY::exhaustive()
                    .map(move |xy_rel| Self { x: x.clone(), y: y.clone(), xy_rel })
            }))
        }
    }

    impl fmt::Display for LocStateProtPair {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "x:{}, y:{}", self.x, self.y)
        }
    }

    /// Arbitrary sequence of events, as experienced by two mutually foreign pointers
    /// to the same location.
    #[derive(Clone)]
    struct OpaqueCode<RetX, RetY> {
        events: Vec<TestEvent<RetX, RetY>>,
    }

    impl<RetX, RetY> fmt::Display for OpaqueCode<RetX, RetY> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for evt in &self.events {
                write!(f, "{evt}; ")?;
            }
            Ok(())
        }
    }

    impl LocStateProtPair {
        /// List all sequences of operations that start at `self` and do not cause UB
        /// There are no duplicates: all sequences returned lead to distinct final states
        /// (though the sequence is not guaranteed to be the shortest possible sequence of events).
        /// Yields the states it reaches, and the sequence of operations that got us there.
        fn all_states_reachable_via_opaque_code<RetX, RetY>(
            self,
        ) -> impl Iterator<Item = (Self, OpaqueCode<RetX, RetY>)>
        where
            RetX: Exhaustive + Clone + 'static,
            RetY: Exhaustive + Clone + 'static,
        {
            // We compute the reachable set of `Self` from `self` by non-UB `OpaqueCode`.
            // Configurations are `(reach: Self, code: OpaqueCode)` tuples
            // for which `code` applied to `self` returns `Ok(reach)`.

            // Stack of all configurations left to handle.
            let mut handle: Vec<(Self, OpaqueCode<_, _>)> =
                vec![(self, OpaqueCode { events: Vec::new() })];
            // Code that can be applied to `self`, and final state.
            let mut paths: Vec<(Self, OpaqueCode<_, _>)> = Default::default();
            // Already explored states reachable from `self`
            let mut seen: FxHashSet<Self> = Default::default();
            // This is essentially just computing the transitive closure by `perform_test_event`,
            // most of the work lies in remembering the path up to the current state.
            while let Some((state, path)) = handle.pop() {
                for evt in <TestEvent<RetX, RetY>>::exhaustive() {
                    if let Ok(next) = state.clone().perform_test_event(&evt) {
                        if seen.insert(next.clone()) {
                            let mut evts = path.clone();
                            evts.events.push(evt);
                            paths.push((next.clone(), evts.clone()));
                            handle.push((next, evts));
                        }
                    }
                }
            }
            paths.into_iter()
        }
    }

    impl LocStateProtPair {
        #[rustfmt::skip]
        /// Two states (by convention `self` is the source and `other` is the target)
        /// are "distinguishable" if there exists a sequence of instructions
        /// that causes UB in the target but not in the source.
        /// This implementation simply explores the reachable space
        /// by all sequences of `TestEvent`.
        /// This function can be instantiated with `RetX` and `RetY`
        /// among `NoRet` or `AllowRet` to resp. forbid/allow `x`/`y` to lose their
        /// protector.
        fn distinguishable<RetX, RetY>(&self, other: &Self) -> bool
        where
            RetX: Exhaustive + 'static,
            RetY: Exhaustive + 'static,
        {
            if self == other { return false; }
            let mut states = vec![(self.clone(), other.clone())];
            let mut seen = FxHashSet::default();
            while let Some(state) = states.pop() {
                if !seen.insert(state.clone()) { continue; };
                let (source, target) = state;
                for evt in <TestEvent<RetX, RetY>>::exhaustive() {
                    // Generate successor states through events (accesses and protector ends)
                    let Ok(new_source) = source.clone().perform_test_event(&evt) else { continue; };
                    let Ok(new_target) = target.clone().perform_test_event(&evt) else { return true; };
                    if new_source == new_target { continue; }
                    states.push((new_source, new_target));
                }
            }
            false
        }
    }

    #[test]
    // `Reserved { conflicted: false }` and `Reserved { conflicted: true }` are properly indistinguishable
    // under the conditions where we want to insert a spurious read.
    fn reserved_aliased_protected_indistinguishable() {
        let source = LocStateProtPair {
            xy_rel: RelPosXY::MutuallyForeign,
            x: LocStateProt {
                // For the tests, the strongest idempotent foreign access does not matter, so we use `Default::default`
                state: LocationState::new_accessed(
                    Permission::new_frozen(),
                    IdempotentForeignAccess::default(),
                ),
                prot: true,
            },
            y: LocStateProt {
                state: LocationState::new_non_accessed(
                    Permission::new_reserved_frz(),
                    IdempotentForeignAccess::default(),
                ),
                prot: true,
            },
        };
        let acc = TestAccess { ptr: PtrSelector::X, kind: AccessKind::Read };
        let target = source.clone().perform_test_access(&acc).unwrap();
        assert!(source.y.state.permission.is_reserved_frz_with_conflicted(false));
        assert!(target.y.state.permission.is_reserved_frz_with_conflicted(true));
        assert!(!source.distinguishable::<(), ()>(&target))
    }

    #[derive(Clone, Debug)]
    struct Pattern {
        /// The relative position of `x` and `y` at the beginning of the arbitrary
        /// code (i.e., just after `x` got created).
        /// Might change during the execution if said arbitrary code contains any `retag y`.
        xy_rel: RelPosXY,
        /// Permission that `x` will be created as
        /// (always protected until a possible `ret x` in the second phase).
        /// This one should be initial (as per `is_initial`).
        x_retag_perm: LocationState,
        /// Permission that `y` has at the beginning of the pattern.
        /// Can be any state, not necessarily initial
        /// (since `y` exists already before the pattern starts).
        /// This state might be reset during the execution if the opaque code
        /// contains any `retag y`, but only to an initial state this time.
        y_current_perm: LocationState,
        /// Whether `y` starts with a protector.
        /// Might change if the opaque code contains any `ret y`.
        y_protected: bool,
    }

    impl Exhaustive for Pattern {
        fn exhaustive() -> Box<dyn Iterator<Item = Self>> {
            let mut v = Vec::new();
            for xy_rel in RelPosXY::exhaustive() {
                for (x_retag_perm, y_current_perm) in <(LocationState, LocationState)>::exhaustive()
                {
                    // We can only do spurious reads for accessed locations anyway.
                    precondition!(x_retag_perm.accessed);
                    // And `x` just got retagged, so it must be initial.
                    precondition!(x_retag_perm.permission.is_initial());
                    // As stated earlier, `x` is always protected in the patterns we consider here.
                    precondition!(x_retag_perm.compatible_with_protector());
                    for y_protected in bool::exhaustive() {
                        // Finally `y` that is optionally protected must have a compatible permission.
                        if y_protected {
                            precondition!(y_current_perm.compatible_with_protector());
                        }
                        v.push(Pattern { xy_rel, x_retag_perm, y_current_perm, y_protected });
                    }
                }
            }
            Box::new(v.into_iter())
        }
    }

    impl fmt::Display for Pattern {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let (x, y) = self.retag_permissions();
            write!(f, "{}; ", self.xy_rel)?;
            write!(f, "y: ({y}); ")?;
            write!(f, "retag x ({x}); ")?;

            write!(f, "<arbitrary code>; <spurious read x>;")?;
            Ok(())
        }
    }

    impl Pattern {
        /// Return the permission that `y` starts as, and the permission that we
        /// will retag `x` with.
        /// This does not yet include a possible read-on-reborrow through `x`.
        fn retag_permissions(&self) -> (LocStateProt, LocStateProt) {
            let x = LocStateProt { state: self.x_retag_perm, prot: true };
            let y = LocStateProt { state: self.y_current_perm, prot: self.y_protected };
            (x, y)
        }

        /// State that the pattern deterministically produces immediately after
        /// the retag of `x`.
        fn initial_state(&self) -> Result<LocStateProtPair, ()> {
            let (x, y) = self.retag_permissions();
            let state = LocStateProtPair { xy_rel: self.xy_rel, x, y };
            state.read_if_accessed(PtrSelector::X)
        }
    }

    #[test]
    /// For each of the patterns described above, execute it once
    /// as-is, and once with a spurious read inserted. Report any UB
    /// in the target but not in the source.
    fn test_all_patterns() {
        let mut ok = 0;
        let mut err = 0;
        for pat in Pattern::exhaustive() {
            let Ok(initial_source) = pat.initial_state() else {
                // Failed to retag `x` in the source (e.g. `y` was protected Active)
                continue;
            };
            // `x` must stay protected, but the function protecting `y` might return here
            for (final_source, opaque) in
                initial_source.all_states_reachable_via_opaque_code::</*X*/ NoRet, /*Y*/ AllowRet>()
            {
                // Both executions are identical up to here.
                // Now we do nothing in the source and in the target we do a spurious read.
                // Then we check if the resulting states are distinguishable.
                let distinguishable = {
                    assert!(final_source.x.prot);
                    let spurious_read = TestAccess { ptr: PtrSelector::X, kind: AccessKind::Read };
                    if let Ok(final_target) =
                        final_source.clone().perform_test_access(&spurious_read)
                    {
                        // Only after the spurious read has been executed can `x` lose its
                        // protector.
                        final_source
                            .distinguishable::</*X*/ AllowRet, /*Y*/ AllowRet>(&final_target)
                            .then_some(format!("{final_target}"))
                    } else {
                        Some(format!("UB"))
                    }
                };
                if let Some(final_target) = distinguishable {
                    eprintln!(
                        "For pattern '{pat}', inserting a spurious read through x makes the final state '{final_target}' \
                        instead of '{final_source}' which is observable"
                    );
                    eprintln!("  (arbitrary code instanciated with '{opaque}')");
                    err += 1;
                    // We found an instanciation of the opaque code that makes this Pattern
                    // fail, we don't really need to check the rest.
                    break;
                }
                ok += 1;
            }
        }
        if err > 0 {
            panic!(
                "Test failed after {}/{} patterns had UB in the target but not the source",
                err,
                ok + err
            )
        }
    }
}
