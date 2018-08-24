use super::SnapshotMap;

#[test]
fn basic() {
    let mut map = SnapshotMap::new();
    map.insert(22, "twenty-two");
    let snapshot = map.snapshot();
    map.insert(22, "thirty-three");
    assert_eq!(map[&22], "thirty-three");
    map.insert(44, "fourty-four");
    assert_eq!(map[&44], "fourty-four");
    assert_eq!(map.get(&33), None);
    map.rollback_to(&snapshot);
    assert_eq!(map[&22], "twenty-two");
    assert_eq!(map.get(&33), None);
    assert_eq!(map.get(&44), None);
}

#[test]
#[should_panic]
fn out_of_order() {
    let mut map = SnapshotMap::new();
    map.insert(22, "twenty-two");
    let snapshot1 = map.snapshot();
    let _snapshot2 = map.snapshot();
    map.rollback_to(&snapshot1);
}

#[test]
fn nested_commit_then_rollback() {
    let mut map = SnapshotMap::new();
    map.insert(22, "twenty-two");
    let snapshot1 = map.snapshot();
    let snapshot2 = map.snapshot();
    map.insert(22, "thirty-three");
    map.commit(&snapshot2);
    assert_eq!(map[&22], "thirty-three");
    map.rollback_to(&snapshot1);
    assert_eq!(map[&22], "twenty-two");
}
