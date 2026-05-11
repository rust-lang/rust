//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This test ensures that at least and at most one of `drop` and `pin_drop`
// are implemented for types that implement `Drop`.

mod drop_only {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn drop(&mut self) {} // ok, only `drop` is implemented
    }

    impl Drop for Bar {
        fn drop(&mut self) {} //~ ERROR `Bar` must implement `pin_drop`
    }
}

mod pin_drop_only {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn pin_drop(&pin mut self) {} // ok, only `pin_drop` is implemented
    }

    impl Drop for Bar {
        fn pin_drop(&pin mut self) {} // ok, non-`#[pin_v2]` can also implement `pin_drop`
    }
}

mod both {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        //~^ ERROR conflicting implementations of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }

    impl Drop for Bar {
        //~^ ERROR conflicting implementations of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }
}

mod neither {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
    impl Drop for Bar {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
}

mod drop_wrong_type {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn drop(&pin mut self) {} //~ ERROR method `drop` has an incompatible type for trait [E0053]
    }
    impl Drop for Bar {
        fn drop(&pin mut self) {}
        //~^ ERROR method `drop` has an incompatible type for trait [E0053]
        //~| ERROR `Bar` must implement `pin_drop`
    }
}

mod pin_drop_wrong_type {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn pin_drop(&mut self) {} //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
    }

    impl Drop for Bar {
        fn pin_drop(&mut self) {} //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
    }
}

mod explicit_call_pin_drop {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn drop(&mut self) {
            Drop::pin_drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
    impl Drop for Bar {
        fn drop(&mut self) {
            //~^ ERROR `Bar` must implement `pin_drop`
            Drop::pin_drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
}

mod explicit_call_drop {
    struct Foo;
    #[pin_v2]
    struct Bar;

    impl Drop for Foo {
        fn pin_drop(&pin mut self) {
            Drop::drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
    impl Drop for Bar {
        fn pin_drop(&pin mut self) {
            Drop::drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
}

fn main() {}
