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

    const INT: u8 = 42;
    const STATIC_REF: Pin<&'static u8> = Pin::static_ref(&INT);
    assert_eq!(*STATIC_REF, INT);

    // Note: `pin_mut_const` tests that the methods of `Pin<&mut T>` are usable in a const context.
    // A const fn is used because `&mut` is not (yet) usable in constants.
    const fn pin_mut_const() {
        let _ = Pin::new(&mut 2).into_ref();
        let _ = Pin::new(&mut 2).get_mut();
        unsafe {
            let _ = Pin::new(&mut 2).get_unchecked_mut();
        }
    }

    pin_mut_const();
}

#[allow(unused)]
mod pin_coerce_unsized {
    use core::cell::{Cell, RefCell, UnsafeCell};
    use core::pin::Pin;
    use core::ptr::NonNull;

    pub trait MyTrait {}
    impl MyTrait for String {}

    // These Pins should continue to compile.
    // Do note that these instances of Pin types cannot be used
    // meaningfully because all methods require a Deref/DerefMut
    // bounds on the pointer type and Cell, RefCell and UnsafeCell
    // do not implement Deref/DerefMut.

    pub fn cell(arg: Pin<Cell<Box<String>>>) -> Pin<Cell<Box<dyn MyTrait>>> {
        arg
    }
    pub fn ref_cell(arg: Pin<RefCell<Box<String>>>) -> Pin<RefCell<Box<dyn MyTrait>>> {
        arg
    }
    pub fn unsafe_cell(arg: Pin<UnsafeCell<Box<String>>>) -> Pin<UnsafeCell<Box<dyn MyTrait>>> {
        arg
    }

    // These sensible Pin coercions are possible.
    pub fn pin_mut_ref(arg: Pin<&mut String>) -> Pin<&mut dyn MyTrait> {
        arg
    }
    pub fn pin_ref(arg: Pin<&String>) -> Pin<&dyn MyTrait> {
        arg
    }
    pub fn pin_ptr(arg: Pin<*const String>) -> Pin<*const dyn MyTrait> {
        arg
    }
    pub fn pin_ptr_mut(arg: Pin<*mut String>) -> Pin<*mut dyn MyTrait> {
        arg
    }
    pub fn pin_non_null(arg: Pin<NonNull<String>>) -> Pin<NonNull<dyn MyTrait>> {
        arg
    }
    pub fn nesting_pins(arg: Pin<Pin<&String>>) -> Pin<Pin<&dyn MyTrait>> {
        arg
    }
}
