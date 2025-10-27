#![allow(undropped_manually_drops)]

use core::mem::ManuallyDrop;

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
        impl const Drop for Dropped<'_> {
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
