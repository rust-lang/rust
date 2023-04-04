//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(arbitrary_self_types, unsize, coerce_unsized, dispatch_from_dyn)]
#![feature(rustc_attrs)]

fn pin_box_dyn() {
    use std::pin::Pin;

    trait Foo {
        fn bar(self: Pin<&mut Self>) -> bool;
    }

    impl Foo for &'static str {
        fn bar(self: Pin<&mut Self>) -> bool {
            true
        }
    }

    let mut test: Pin<Box<dyn Foo>> = Box::pin("foo");
    test.as_mut().bar();
}

fn stdlib_pointers() {
    use std::{pin::Pin, rc::Rc, sync::Arc};

    trait Trait {
        fn by_rc(self: Rc<Self>) -> i64;
        fn by_arc(self: Arc<Self>) -> i64;
        fn by_pin_mut(self: Pin<&mut Self>) -> i64;
        fn by_pin_box(self: Pin<Box<Self>>) -> i64;
    }

    impl Trait for i64 {
        fn by_rc(self: Rc<Self>) -> i64 {
            *self
        }
        fn by_arc(self: Arc<Self>) -> i64 {
            *self
        }
        fn by_pin_mut(self: Pin<&mut Self>) -> i64 {
            *self
        }
        fn by_pin_box(self: Pin<Box<Self>>) -> i64 {
            *self
        }
    }

    let rc = Rc::new(1i64) as Rc<dyn Trait>;
    assert_eq!(1, rc.by_rc());

    let arc = Arc::new(2i64) as Arc<dyn Trait>;
    assert_eq!(2, arc.by_arc());

    let mut value = 3i64;
    let pin_mut = Pin::new(&mut value) as Pin<&mut dyn Trait>;
    assert_eq!(3, pin_mut.by_pin_mut());

    let pin_box = Into::<Pin<Box<i64>>>::into(Box::new(4i64)) as Pin<Box<dyn Trait>>;
    assert_eq!(4, pin_box.by_pin_box());
}

fn pointers_and_wrappers() {
    use std::{
        marker::Unsize,
        ops::{CoerceUnsized, Deref, DispatchFromDyn},
    };

    struct Ptr<T: ?Sized>(Box<T>);

    impl<T: ?Sized> Deref for Ptr<T> {
        type Target = T;

        fn deref(&self) -> &T {
            &*self.0
        }
    }

    impl<T: Unsize<U> + ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> {}
    impl<T: Unsize<U> + ?Sized, U: ?Sized> DispatchFromDyn<Ptr<U>> for Ptr<T> {}

    struct Wrapper<T: ?Sized>(T);

    impl<T: ?Sized> Deref for Wrapper<T> {
        type Target = T;

        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl<T: CoerceUnsized<U>, U> CoerceUnsized<Wrapper<U>> for Wrapper<T> {}
    impl<T: DispatchFromDyn<U>, U> DispatchFromDyn<Wrapper<U>> for Wrapper<T> {}

    trait Trait {
        // This method isn't object-safe yet. Unsized by-value `self` is object-safe (but not callable
        // without unsized_locals), but wrappers arond `Self` currently are not.
        // FIXME (mikeyhew) uncomment this when unsized rvalues object-safety is implemented
        // fn wrapper(self: Wrapper<Self>) -> i32;
        fn ptr_wrapper(self: Ptr<Wrapper<Self>>) -> i32;
        fn wrapper_ptr(self: Wrapper<Ptr<Self>>) -> i32;
        fn wrapper_ptr_wrapper(self: Wrapper<Ptr<Wrapper<Self>>>) -> i32;
    }

    impl Trait for i32 {
        fn ptr_wrapper(self: Ptr<Wrapper<Self>>) -> i32 {
            **self
        }
        fn wrapper_ptr(self: Wrapper<Ptr<Self>>) -> i32 {
            **self
        }
        fn wrapper_ptr_wrapper(self: Wrapper<Ptr<Wrapper<Self>>>) -> i32 {
            ***self
        }
    }

    let pw = Ptr(Box::new(Wrapper(5))) as Ptr<Wrapper<dyn Trait>>;
    assert_eq!(pw.ptr_wrapper(), 5);

    let wp = Wrapper(Ptr(Box::new(6))) as Wrapper<Ptr<dyn Trait>>;
    assert_eq!(wp.wrapper_ptr(), 6);

    let wpw = Wrapper(Ptr(Box::new(Wrapper(7)))) as Wrapper<Ptr<Wrapper<dyn Trait>>>;
    assert_eq!(wpw.wrapper_ptr_wrapper(), 7);
}

fn raw_ptr_receiver() {
    use std::ptr;

    trait Foo {
        fn foo(self: *const Self) -> &'static str;
    }

    impl Foo for i32 {
        fn foo(self: *const Self) -> &'static str {
            "I'm an i32!"
        }
    }

    impl Foo for u32 {
        fn foo(self: *const Self) -> &'static str {
            "I'm a u32!"
        }
    }

    let null_i32 = ptr::null::<i32>() as *const dyn Foo;
    let null_u32 = ptr::null::<u32>() as *const dyn Foo;

    assert_eq!("I'm an i32!", null_i32.foo());
    assert_eq!("I'm a u32!", null_u32.foo());
}

fn main() {
    pin_box_dyn();
    stdlib_pointers();
    pointers_and_wrappers();
    raw_ptr_receiver();
}
