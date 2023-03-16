use std::cmp::{Ordering, PartialOrd};
use std::fmt;

/// The activation states of a pointer
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
}

impl PermissionPriv {
    /// Determines whether a transition that occured is compatible with the presence
    /// of a Protector. This is not included in the `transition` functions because
    /// it would distract from the few places where the transition is modified
    /// because of a protector, but not forbidden.
    fn protector_allows_transition(self, new: Self) -> bool {
        match (self, new) {
            _ if self == new => true,
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
            _ => unreachable!("Transition from {self:?} to {new:?} should never be possible"),
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
