use super::DisjointSet;

#[test]
fn smoke() {
    let mut sets: DisjointSet<u32> = DisjointSet::with_cardinality(12);

    assert!(sets.is_disjoint(1, 4));
    assert!(sets.is_disjoint(5, 11));
    assert!(sets.is_joint(7, 7));

    // [1, 4, 5, 11]
    sets.union(1, 4);
    sets.union(5, 11);
    sets.union(4, 5);

    // [2, 3]
    sets.union(2, 3);

    assert!(sets.is_joint(1, 11));
    assert!(sets.is_joint(5, 11));
    assert!(sets.is_joint(2, 3));
    assert!(sets.is_disjoint(2, 4));
    assert!(sets.is_disjoint(1, 3));

    // [1, 2, 3, 4, 5, 11]
    sets.union(3, 11);

    assert!(sets.is_joint(5, 2));
}
