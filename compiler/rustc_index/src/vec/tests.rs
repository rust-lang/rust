// Allows the macro invocation below to work
use crate as rustc_index;

crate::newtype_index! {
    #[orderable]
    #[max = 0xFFFF_FFFA]
    struct MyIdx {}
}

#[test]
fn index_size_is_optimized() {
    assert_eq!(size_of::<MyIdx>(), 4);
    // Uses 0xFFFF_FFFB
    assert_eq!(size_of::<Option<MyIdx>>(), 4);
    // Uses 0xFFFF_FFFC
    assert_eq!(size_of::<Option<Option<MyIdx>>>(), 4);
    // Uses 0xFFFF_FFFD
    assert_eq!(size_of::<Option<Option<Option<MyIdx>>>>(), 4);
    // Uses 0xFFFF_FFFE
    assert_eq!(size_of::<Option<Option<Option<Option<MyIdx>>>>>(), 4);
    // Uses 0xFFFF_FFFF
    assert_eq!(size_of::<Option<Option<Option<Option<Option<MyIdx>>>>>>(), 4);
    // Uses a tag
    assert_eq!(size_of::<Option<Option<Option<Option<Option<Option<MyIdx>>>>>>>(), 8);
}

#[test]
fn range_iterator_iterates_forwards() {
    let range = MyIdx::from_u32(1)..MyIdx::from_u32(4);
    assert_eq!(
        range.collect::<Vec<_>>(),
        [MyIdx::from_u32(1), MyIdx::from_u32(2), MyIdx::from_u32(3)]
    );
}

#[test]
fn range_iterator_iterates_backwards() {
    let range = MyIdx::from_u32(1)..MyIdx::from_u32(4);
    assert_eq!(
        range.rev().collect::<Vec<_>>(),
        [MyIdx::from_u32(3), MyIdx::from_u32(2), MyIdx::from_u32(1)]
    );
}

#[test]
fn range_count_is_correct() {
    let range = MyIdx::from_u32(1)..MyIdx::from_u32(4);
    assert_eq!(range.count(), 3);
}

#[test]
fn range_size_hint_is_correct() {
    let range = MyIdx::from_u32(1)..MyIdx::from_u32(4);
    assert_eq!(range.size_hint(), (3, Some(3)));
}
