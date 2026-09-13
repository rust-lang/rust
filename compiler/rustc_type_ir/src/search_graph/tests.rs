use super::{CycleKey, PathKind};

#[test]
fn closing_edge_participates_in_productivity() {
    let cycle =
        CycleKey::new(vec!["head", "tail"], vec![PathKind::Inductive, PathKind::Coinductive]);
    assert_eq!(cycle.path_kind(), PathKind::Coinductive);
    assert!(cycle.is_productive());
}

#[test]
fn cycle_identity_is_root_directed_and_keeps_every_edge() {
    let rooted_at_a =
        CycleKey::new(vec!['A', 'B'], vec![PathKind::Coinductive, PathKind::Inductive]);
    let rooted_at_b =
        CycleKey::new(vec!['B', 'A'], vec![PathKind::Inductive, PathKind::Coinductive]);
    assert_ne!(rooted_at_a, rooted_at_b);
    assert_eq!(rooted_at_a.head(), &'A');
    assert_eq!(rooted_at_a.edge_kinds().len(), rooted_at_a.participants().len());
}

#[test]
fn forced_ambiguity_is_never_a_productive_proof() {
    let cycle = CycleKey::new(vec![0, 1], vec![PathKind::Coinductive, PathKind::ForcedAmbiguity]);
    assert_eq!(cycle.path_kind(), PathKind::ForcedAmbiguity);
    assert!(!cycle.is_productive());
}

#[test]
#[should_panic(expected = "one outgoing edge per participant")]
fn cycle_requires_a_closing_edge() {
    let _ = CycleKey::new(vec![0, 1], vec![PathKind::Coinductive]);
}
