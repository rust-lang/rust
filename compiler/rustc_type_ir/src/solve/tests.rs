// Check that deduplication preserves the leak-check visibility rules.
use super::VisibleForLeakCheck::{No, Unreachable, Yes};
use super::*;

#[test]
fn visibility_join_is_commutative_associative_and_idempotent() {
    let values = [Yes, No, Unreachable];
    for a in values {
        assert_eq!(a.or(a), a);
        for b in values {
            assert_eq!(a.or(b), b.or(a));
            for c in values {
                assert_eq!(a.or(b).or(c), a.or(b.or(c)));
            }
        }
    }
}

#[test]
fn visibility_join_respects_priority() {
    assert_eq!(Yes.or(No), Yes);
    assert_eq!(No.or(Unreachable), Unreachable);
    assert_eq!(Yes.or(Unreachable), Unreachable);
}

#[cfg(test)]
mod region_constraint_dedup_tests {
    use super::VisibleForLeakCheck;

    #[test]
    fn merging_region_constraint_visibility() {
        use VisibleForLeakCheck::{No, Unreachable, Yes};

        assert_eq!(No.or(Yes), Yes);
        assert_eq!(Yes.or(No), Yes);
        assert_eq!(No.or(No), No);
        assert_eq!(Yes.or(Yes), Yes);

        assert_eq!(Unreachable.or(Yes), Unreachable);
        assert_eq!(Yes.or(Unreachable), Unreachable);
        assert_eq!(Unreachable.or(No), Unreachable);
        assert_eq!(No.or(Unreachable), Unreachable);
    }
}
