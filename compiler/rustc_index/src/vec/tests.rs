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

#[test]
fn range_inclusive_iterator_iterates_backwards() {
    let range = MyIdx::ZERO..=MyIdx::from_u32(1);
    assert_eq!(range.rev().collect::<Vec<_>>(), [MyIdx::from_u32(1), MyIdx::ZERO]);
}

#[test]
fn range_inclusive_iterator_iterates_forwards() {
    let range = MyIdx::from_u32(MyIdx::MAX_AS_U32 - 1)..=MyIdx::MAX;
    assert_eq!(range.collect::<Vec<_>>(), [MyIdx::from_u32(MyIdx::MAX_AS_U32 - 1), MyIdx::MAX]);
}

#[test]
fn step_forward_checked_is_correct() {
    assert_eq!(std::iter::Step::forward_checked(MyIdx::MAX, 1), None);
}

#[test]
fn range_inclusive_nth_past_the_bounds_does_not_panic() {
    let max = MyIdx::MAX_AS_U32 as usize;
    assert_eq!((MyIdx::ZERO..=MyIdx::MAX).nth(max), Some(MyIdx::MAX));
    assert_eq!((MyIdx::ZERO..=MyIdx::MAX).nth(max + 1), None);
    assert_eq!((MyIdx::ZERO..=MyIdx::MAX).nth_back(max), Some(MyIdx::ZERO));
    assert_eq!((MyIdx::ZERO..=MyIdx::MAX).nth_back(max + 1), None);
}
