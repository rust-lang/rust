use super::DropRange;

#[test]
fn drop_range_uses_last_event() {
    let mut range = DropRange::empty();
    range.drop(10);
    range.reinit(10);
    assert!(!range.is_dropped_at(10));

    let mut range = DropRange::empty();
    range.reinit(10);
    range.drop(10);
    assert!(range.is_dropped_at(10));
}
