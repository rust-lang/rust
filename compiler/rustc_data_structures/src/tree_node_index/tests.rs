use crate::tree_node_index::TreeNodeIndex;

#[test]
fn up_to_16() {
    for n in 1..128 {
        for i in 0..n {
            TreeNodeIndex::root().branch(i, n).branch(n - i - 1, n);
        }
    }
}

#[test]
fn ceil_log2() {
    const EVALUATION_TABLE: [(u64, u32); 9] =
        [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (7, 3), (8, 3), (u64::MAX, 64)];
    for &(x, y) in &EVALUATION_TABLE {
        let r = super::ceil_ilog2(x);
        assert!(r == y, "ceil_ilog2({x}) == {r} != {y}");
    }
}

#[test]
fn some_cases() {
    let mut tni = TreeNodeIndex::root();
    tni = tni.branch(0xDEAD, 0xFADE);
    assert_eq!(tni.0, 0xDEAD8000_00000000);
    tni = tni.branch(0xBEEF, 0xCCCC);
    assert_eq!(tni.0, 0xDEADBEEF_80000000);
    tni = tni.branch(1, 2);
    assert_eq!(tni.0, 0xDEADBEEF_C0000000);
    tni = tni.branch(0, 2);
    assert_eq!(tni.0, 0xDEADBEEF_A0000000);
    tni = tni.branch(3, 4);
    assert_eq!(tni.0, 0xDEADBEEF_B8000000);
    tni = tni.branch(0xAAAAAA, 0xBBBBBB);
    assert_eq!(tni.0, 0xDEADBEEF_BAAAAAA8);
}

#[test]
fn edge_cases() {
    const ROOT: TreeNodeIndex = TreeNodeIndex::root();
    assert_eq!(ROOT.branch(0, 1), TreeNodeIndex::root());
    assert_eq!(ROOT.branch(u64::MAX >> 1, 1 << 63).0, u64::MAX);
    assert_eq!(ROOT.branch(0, 1 << 63).0, 1);
}
