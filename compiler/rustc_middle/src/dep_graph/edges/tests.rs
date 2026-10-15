use super::*;

#[test]
fn task_reads_are_deduplicated_in_order() {
    let pool = Lock::new(Vec::new());
    let mut reads = TaskReads::new();
    let index = DepNodeIndex::from_u32;

    assert!(reads.insert(index(0), &pool));
    assert!(!reads.insert(index(0), &pool));
    assert!(reads.insert(index(64), &pool));
    assert!(!reads.insert(index(0), &pool));
    assert!(!reads.insert(index(0), &pool));
    for i in 1..15 {
        assert!(reads.insert(index(i), &pool));
    }
    assert!(!reads.insert(index(64), &pool));
    assert!(matches!(reads, TaskReads::Small { .. }));
    assert!(reads.insert(index(128), &pool));
    assert!(matches!(reads, TaskReads::Recorded(_)));
    assert!(!reads.insert(index(0), &pool));
    assert!(!reads.insert(index(128), &pool));
    assert!(reads.insert(index(129), &pool));

    let expected: Vec<_> = [0, 64].into_iter().chain(1..15).chain([128, 129]).map(index).collect();
    assert_eq!(reads.edges(), expected);
}
