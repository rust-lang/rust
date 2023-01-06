#![warn(clippy::unused_self)]
#![allow(clippy::boxed_local, clippy::fn_params_excessive_bools)]

mod unused_self {
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    struct A;

    impl A {
        fn unused_self_move(self) {}
        fn unused_self_ref(&self) {}
        fn unused_self_mut_ref(&mut self) {}
        fn unused_self_pin_ref(self: Pin<&Self>) {}
        fn unused_self_pin_mut_ref(self: Pin<&mut Self>) {}
        fn unused_self_pin_nested(self: Pin<Arc<Self>>) {}
        fn unused_self_box(self: Box<Self>) {}
        fn unused_with_other_used_args(&self, x: u8, y: u8) -> u8 {
            x + y
        }
        fn unused_self_class_method(&self) {
            Self::static_method();
        }

        fn static_method() {}
    }
}

mod unused_self_allow {
    struct A;

    impl A {
        // shouldn't trigger
        #[allow(clippy::unused_self)]
        fn unused_self_move(self) {}
    }

    struct B;

    // shouldn't trigger
    #[allow(clippy::unused_self)]
    impl B {
        fn unused_self_move(self) {}
    }

    struct C;

    #[allow(clippy::unused_self)]
    impl C {
        #[warn(clippy::unused_self)]
        fn some_fn((): ()) {}

        // shouldn't trigger
        fn unused_self_move(self) {}
    }

    pub struct D;

    impl D {
        // shouldn't trigger for public methods
        pub fn unused_self_move(self) {}
    }

    pub struct E;

    impl E {
        // shouldn't trigger if body contains todo!()
        pub fn unused_self_todo(self) {
            let x = 42;
            todo!()
        }
    }
}

pub use unused_self_allow::D;

mod used_self {
    use std::pin::Pin;

    struct A {
        x: u8,
    }

    impl A {
        fn used_self_move(self) -> u8 {
            self.x
        }
        fn used_self_ref(&self) -> u8 {
            self.x
        }
        fn used_self_mut_ref(&mut self) {
            self.x += 1
        }
        fn used_self_pin_ref(self: Pin<&Self>) -> u8 {
            self.x
        }
        fn used_self_box(self: Box<Self>) -> u8 {
            self.x
        }
        fn used_self_with_other_unused_args(&self, x: u8, y: u8) -> u8 {
            self.x
        }
        fn used_in_nested_closure(&self) -> u8 {
            let mut a = || -> u8 { self.x };
            a()
        }

        #[allow(clippy::collapsible_if)]
        fn used_self_method_nested_conditions(&self, a: bool, b: bool, c: bool, d: bool) {
            if a {
                if b {
                    if c {
                        if d {
                            self.used_self_ref();
                        }
                    }
                }
            }
        }

        fn foo(&self) -> u32 {
            let mut sum = 0u32;
            for i in 0..self.x {
                sum += i as u32;
            }
            sum
        }

        fn bar(&mut self, x: u8) -> u32 {
            let mut y = 0u32;
            for i in 0..x {
                y += self.foo()
            }
            y
        }
    }
}

mod not_applicable {
    use std::fmt;

    struct A;

    impl fmt::Debug for A {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "A")
        }
    }

    impl A {
        fn method(x: u8, y: u8) {}
    }

    trait B {
        fn method(&self) {}
    }
}

fn main() {}
