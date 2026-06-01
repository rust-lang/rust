#![allow(undropped_manually_drops)]

use core::mem::ManuallyDrop;
use core::pin::Pin;

#[test]
fn smoke() {
    #[derive(Clone)]
    struct TypeWithDrop;
    impl Drop for TypeWithDrop {
        fn drop(&mut self) {
            unreachable!("Should not get dropped");
        }
    }

    let x = ManuallyDrop::new(TypeWithDrop);
    drop(x);

    // also test unsizing
    let x: Box<ManuallyDrop<[TypeWithDrop]>> =
        Box::new(ManuallyDrop::new([TypeWithDrop, TypeWithDrop]));
    drop(x);

    // test clone and clone_from implementations
    let mut x = ManuallyDrop::new(TypeWithDrop);
    let y = x.clone();
    x.clone_from(&y);
    drop(x);
    drop(y);
}

#[test]
fn const_drop_in_place() {
    const COUNTER: usize = {
        use core::cell::Cell;

        let counter = Cell::new(0);

        // only exists to make `Drop` indirect impl
        #[allow(dead_code)]
        struct Test<'a>(Dropped<'a>);

        struct Dropped<'a>(&'a Cell<usize>);
        const impl Drop for Dropped<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let mut one = ManuallyDrop::new(Test(Dropped(&counter)));
        let mut two = ManuallyDrop::new(Test(Dropped(&counter)));
        let mut three = ManuallyDrop::new(Test(Dropped(&counter)));
        assert!(counter.get() == 0);
        unsafe {
            ManuallyDrop::drop(&mut one);
        }
        assert!(counter.get() == 1);
        unsafe {
            ManuallyDrop::drop(&mut two);
        }
        assert!(counter.get() == 2);
        unsafe {
            ManuallyDrop::drop(&mut three);
        }
        counter.get()
    };
    assert_eq!(COUNTER, 3);
}

#[test]
fn const_pinned_drop_in_place() {
    const COUNTER: usize = {
        use core::cell::Cell;

        let counter = Cell::new(0);

        // only exists to make `Drop` indirect impl
        #[allow(dead_code)]
        struct Test<'a>(Dropped<'a>);

        struct Dropped<'a>(&'a Cell<usize>);
        const impl Drop for Dropped<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let mut one = ManuallyDrop::new(Test(Dropped(&counter)));
        let mut two = ManuallyDrop::new(Test(Dropped(&counter)));
        let mut three = ManuallyDrop::new(Test(Dropped(&counter)));

        let mut pinned_one = Pin::new(&mut one);
        let mut pinned_two = Pin::new(&mut two);
        let mut pinned_three = Pin::new(&mut three);
        assert!(counter.get() == 0);
        unsafe {
            ManuallyDrop::pinned_drop(pinned_one.as_mut());
        }
        assert!(counter.get() == 1);
        unsafe {
            ManuallyDrop::pinned_drop(pinned_two.as_mut());
        }
        assert!(counter.get() == 2);
        unsafe {
            ManuallyDrop::pinned_drop(pinned_three.as_mut());
        }
        counter.get()
    };
    assert_eq!(COUNTER, 3);
}

#[test]
fn pinned_deref_in_place() {
    use core::cell::Cell;

    let counter = Cell::new(0);

    #[allow(dead_code)]
    // only exists to make `Drop` indirect impl
    struct Dropped<'a>(&'a Cell<usize>);
    impl Drop for Dropped<'_> {
        fn drop(&mut self) {
            unreachable!("Should not get dropped");
        }
    }

    let one = ManuallyDrop::new(Dropped(&counter));
    let two = ManuallyDrop::new(Dropped(&counter));
    let three = ManuallyDrop::new(Dropped(&counter));

    let pinned_one = Pin::new(&one);
    let pinned_two = Pin::new(&two);
    let pinned_three = Pin::new(&three);

    let manually_pinned_one = ManuallyDrop::pinned_deref(pinned_one.as_ref());
    assert_eq!(manually_pinned_one.as_ref().0.get(), 0);

    let manually_pinned_two = ManuallyDrop::pinned_deref(pinned_two.as_ref());
    assert_eq!(manually_pinned_two.as_ref().0.get(), 0);

    let manually_pinned_three = ManuallyDrop::pinned_deref(pinned_three.as_ref());
    assert_eq!(manually_pinned_three.as_ref().0.get(), 0);
}

#[test]
fn pinned_deref_mut_in_place() {
    use core::cell::Cell;

    let counter = Cell::new(0);

    #[allow(dead_code)]
    // only exists to make `Drop` indirect impl
    struct Dropped<'a>(&'a Cell<usize>);
    impl Drop for Dropped<'_> {
        fn drop(&mut self) {
            unreachable!("Should not get dropped");
        }
    }

    let mut one = ManuallyDrop::new(Dropped(&counter));
    let mut two = ManuallyDrop::new(Dropped(&counter));
    let mut three = ManuallyDrop::new(Dropped(&counter));

    let mut pinned_one = Pin::new(&mut one);
    let mut pinned_two = Pin::new(&mut two);
    let mut pinned_three = Pin::new(&mut three);

    let mut manually_pinned_one = ManuallyDrop::pinned_deref_mut(pinned_one.as_mut());
    assert_eq!(manually_pinned_one.as_mut().0.get(), 0);

    let mut manually_pinned_two = ManuallyDrop::pinned_deref_mut(pinned_two.as_mut());
    assert_eq!(manually_pinned_two.as_mut().0.get(), 0);

    let mut manually_pinned_three = ManuallyDrop::pinned_deref_mut(pinned_three.as_mut());
    assert_eq!(manually_pinned_three.as_mut().0.get(), 0);
}
