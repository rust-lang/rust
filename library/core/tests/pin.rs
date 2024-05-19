use core::pin::Pin;

#[test]
fn pin_const() {
    // test that the methods of `Pin` are usable in a const context

    const POINTER: &'static usize = &2;

    const PINNED: Pin<&'static usize> = Pin::new(POINTER);
    const PINNED_UNCHECKED: Pin<&'static usize> = unsafe { Pin::new_unchecked(POINTER) };
    assert_eq!(PINNED_UNCHECKED, PINNED);

    const INNER: &'static usize = Pin::into_inner(PINNED);
    assert_eq!(INNER, POINTER);

    const INNER_UNCHECKED: &'static usize = unsafe { Pin::into_inner_unchecked(PINNED) };
    assert_eq!(INNER_UNCHECKED, POINTER);

    const REF: &'static usize = PINNED.get_ref();
    assert_eq!(REF, POINTER);

    // Note: `pin_mut_const` tests that the methods of `Pin<&mut T>` are usable in a const context.
    // A const fn is used because `&mut` is not (yet) usable in constants.
    const fn pin_mut_const() {
        let _ = Pin::new(&mut 2).into_ref();
        let _ = Pin::new(&mut 2).get_mut();
        let _ = unsafe { Pin::new(&mut 2).get_unchecked_mut() };
    }

    pin_mut_const();
}
