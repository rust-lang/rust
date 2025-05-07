use super::AccessKind;
use super::tree::AccessRelatedness;

/// To speed up tree traversals, we want to skip traversing subtrees when we know the traversal will have no effect.
/// This is often the case for foreign accesses, since usually foreign accesses happen several times in a row, but also
/// foreign accesses are idempotent. In particular, see tests `foreign_read_is_noop_after_foreign_write` and `all_transitions_idempotent`.
/// Thus, for each node we keep track of the "strongest idempotent foreign access" (SIFA), i.e. which foreign access can be skipped.
/// Note that for correctness, it is not required that this is the strongest access, just any access it is idempotent under. In particular, setting
/// it to `None` is always correct, but the point of this optimization is to have it be as strong as possible so that more accesses can be skipped.
/// This enum represents the kinds of values we store:
/// - `None` means that the node (and its subtrees) are not (guaranteed to be) idempotent under any foreign access.
/// - `Read` means that the node (and its subtrees) are idempotent under foreign reads, but not (yet / necessarily) under foreign writes.
/// - `Write` means that the node (and its subtrees) are idempotent under foreign writes. This also implies that it is idempotent under foreign
///   reads, since reads are stronger than writes (see test `foreign_read_is_noop_after_foreign_write`). In other words, this node can be skipped
///   for all foreign accesses.
///
/// Since a traversal does not just visit a node, but instead the entire subtree, the SIFA field for a given node indicates that the access to
/// *the entire subtree* rooted at that node can be skipped. In order for this to work, we maintain the global invariant that at
/// each location, the SIFA at each child must be stronger than that at the parent. For normal reads and writes, this is easily accomplished by
/// tracking each foreign access as it occurs, so that then the next access can be skipped. This also obviously maintains the invariant, because
/// if a node undergoes a foreign access, then all its children also see this as a foreign access. However, the invariant is broken during retags,
/// because retags act across the entire allocation, but only emit a read event across a specific range. This means that for all nodes outside that
/// range, the invariant is potentially broken, since a new child with a weaker SIFA is inserted. Thus, during retags, special care is taken to
/// "manually" reset the parent's SIFA to be at least as strong as the new child's. This is accomplished with the `ensure_no_stronger_than` method.
///
/// Note that we derive Ord and PartialOrd, so the order in which variants are listed below matters:
/// None < Read < Write. Do not change that order. See the `test_order` test.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum IdempotentForeignAccess {
    #[default]
    None,
    Read,
    Write,
}

impl IdempotentForeignAccess {
    /// Returns true if a node where the strongest idempotent foreign access is `self`
    /// can skip the access `happening_next`. Note that if this returns
    /// `true`, then the entire subtree will be skipped.
    pub fn can_skip_foreign_access(self, happening_next: IdempotentForeignAccess) -> bool {
        debug_assert!(happening_next.is_foreign());
        // This ordering is correct. Intuitively, if the last access here was
        // a foreign write, everything can be skipped, since after a foreign write,
        // all further foreign accesses are idempotent
        happening_next <= self
    }

    /// Updates `self` to account for a foreign access.
    pub fn record_new(&mut self, just_happened: IdempotentForeignAccess) {
        if just_happened.is_local() {
            // If the access is local, reset it.
            *self = IdempotentForeignAccess::None;
        } else {
            // Otherwise, keep it or stengthen it.
            *self = just_happened.max(*self);
        }
    }

    /// Returns true if this access is local.
    pub fn is_local(self) -> bool {
        matches!(self, IdempotentForeignAccess::None)
    }

    /// Returns true if this access is foreign, i.e. not local.
    pub fn is_foreign(self) -> bool {
        !self.is_local()
    }

    /// Constructs a foreign access from an `AccessKind`
    pub fn from_foreign(acc: AccessKind) -> IdempotentForeignAccess {
        match acc {
            AccessKind::Read => Self::Read,
            AccessKind::Write => Self::Write,
        }
    }

    /// Usually, tree traversals have an `AccessKind` and an `AccessRelatedness`.
    /// This methods converts these into the corresponding `IdempotentForeignAccess`, to be used
    /// to e.g. invoke `can_skip_foreign_access`.
    pub fn from_acc_and_rel(acc: AccessKind, rel: AccessRelatedness) -> IdempotentForeignAccess {
        if rel.is_foreign() { Self::from_foreign(acc) } else { Self::None }
    }

    /// During retags, the SIFA needs to be weakened to account for children with weaker SIFAs being inserted.
    /// Thus, this method is called from the bottom up on each parent, until it returns false, which means the
    /// "children have stronger SIFAs" invariant is restored.
    pub fn ensure_no_stronger_than(&mut self, strongest_allowed: IdempotentForeignAccess) -> bool {
        if *self > strongest_allowed {
            *self = strongest_allowed;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IdempotentForeignAccess;

    #[test]
    fn test_order() {
        // The internal logic relies on this order.
        // Do not change.
        assert!(IdempotentForeignAccess::None < IdempotentForeignAccess::Read);
        assert!(IdempotentForeignAccess::Read < IdempotentForeignAccess::Write);
    }
}
