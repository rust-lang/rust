use alloc::boxed::ThinBox;
use core::mem::size_of;

#[test]
fn want_niche_optimization() {
    fn uses_niche<T: ?Sized>() -> bool {
        size_of::<*const ()>() == size_of::<Option<ThinBox<T>>>()
    }

    trait Tr {}
    assert!(uses_niche::<dyn Tr>());
    assert!(uses_niche::<[i32]>());
    assert!(uses_niche::<i32>());
}

#[test]
fn want_thin() {
    fn is_thin<T: ?Sized>() -> bool {
        size_of::<*const ()>() == size_of::<ThinBox<T>>()
    }

    trait Tr {}
    assert!(is_thin::<dyn Tr>());
    assert!(is_thin::<[i32]>());
    assert!(is_thin::<i32>());
}
