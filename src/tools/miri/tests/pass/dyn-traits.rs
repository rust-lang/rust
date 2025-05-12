fn ref_box_dyn() {
    struct Struct(i32);

    trait Trait {
        fn method(&self);

        fn box_method(self: Box<Self>);
    }

    impl Trait for Struct {
        fn method(&self) {
            assert_eq!(self.0, 42);
        }

        fn box_method(self: Box<Self>) {
            assert_eq!(self.0, 7);
        }
    }

    struct Foo<T: ?Sized>(T);

    let y: &dyn Trait = &Struct(42);
    y.method();

    let x: Foo<Struct> = Foo(Struct(42));
    let y: &Foo<dyn Trait> = &x;
    y.0.method();

    let y: Box<dyn Trait> = Box::new(Struct(42));
    y.method();

    let y = &y;
    y.method();

    let y: Box<dyn Trait> = Box::new(Struct(7));
    y.box_method();
}

fn box_box_trait() {
    struct DroppableStruct;

    static mut DROPPED: bool = false;

    impl Drop for DroppableStruct {
        fn drop(&mut self) {
            unsafe {
                DROPPED = true;
            }
        }
    }

    trait MyTrait {
        fn dummy(&self) {}
    }
    impl MyTrait for Box<DroppableStruct> {}

    struct Whatever {
        w: Box<dyn MyTrait + 'static>,
    }

    impl Whatever {
        fn new(w: Box<dyn MyTrait + 'static>) -> Whatever {
            Whatever { w: w }
        }
    }

    {
        let f = Box::new(DroppableStruct);
        let a = Whatever::new(Box::new(f) as Box<dyn MyTrait>);
        a.w.dummy();
    }
    assert!(unsafe { DROPPED });
}

// Disabled for now: unsized locals are not supported,
// their current MIR encoding is just not great.
/*
fn unsized_dyn() {
    pub trait Foo {
        fn foo(self) -> String;
    }

    struct A;

    impl Foo for A {
        fn foo(self) -> String {
            format!("hello")
        }
    }

    let x = *(Box::new(A) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("hello"));

    // I'm not sure whether we want this to work
    let x = Box::new(A) as Box<dyn Foo>;
    assert_eq!(x.foo(), format!("hello"));
}
fn unsized_dyn_autoderef() {
    pub trait Foo {
        fn foo(self) -> String;
    }

    impl Foo for [char] {
        fn foo(self) -> String {
            self.iter().collect()
        }
    }

    impl Foo for str {
        fn foo(self) -> String {
            self.to_owned()
        }
    }

    impl Foo for dyn FnMut() -> String {
        fn foo(mut self) -> String {
            self()
        }
    }

    let x = *(Box::new(['h', 'e', 'l', 'l', 'o']) as Box<[char]>);
    assert_eq!(&x.foo() as &str, "hello");

    let x = Box::new(['h', 'e', 'l', 'l', 'o']) as Box<[char]>;
    assert_eq!(&x.foo() as &str, "hello");

    let x = "hello".to_owned().into_boxed_str();
    assert_eq!(&x.foo() as &str, "hello");

    let x = *("hello".to_owned().into_boxed_str());
    assert_eq!(&x.foo() as &str, "hello");

    let x = "hello".to_owned().into_boxed_str();
    assert_eq!(&x.foo() as &str, "hello");

    let x = *(Box::new(|| "hello".to_owned()) as Box<dyn FnMut() -> String>);
    assert_eq!(&x.foo() as &str, "hello");

    let x = Box::new(|| "hello".to_owned()) as Box<dyn FnMut() -> String>;
    assert_eq!(&x.foo() as &str, "hello");
}
*/

fn vtable_ptr_eq() {
    use std::{fmt, ptr};

    // We don't always get the same vtable when casting this to a wide pointer.
    let x = &2;
    let x_wide = x as &dyn fmt::Display;
    assert!((0..256).any(|_| !ptr::eq(x as &dyn fmt::Display, x_wide)));
}

fn main() {
    ref_box_dyn();
    box_box_trait();
    vtable_ptr_eq();
}
