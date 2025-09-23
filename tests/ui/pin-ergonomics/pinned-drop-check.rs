//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This test ensures that at least and at most one of `drop` and `pin_drop`
// are implemented for types that implement `Drop`.

mod drop_only {
    struct Foo;

    impl Drop for Foo {
        fn drop(&mut self) {} // ok, only `drop` is implemented
    }
}

mod pin_drop_only {
    struct Foo;

    impl Drop for Foo {
        fn pin_drop(&pin mut self) {} // ok, only `pin_drop` is implemented
    }
}

mod both {
    struct Foo;

    impl Drop for Foo {
        //~^ ERROR conflict implementation of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }
}

mod neither {
    struct Foo;

    impl Drop for Foo {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
}

mod drop_wrong_type {
    struct Foo;

    impl Drop for Foo {
        fn drop(&pin mut self) {} //~ ERROR method `drop` has an incompatible type for trait [E0053]
    }
}

mod pin_drop_wrong_type {
    struct Foo;

    impl Drop for Foo {
        fn pin_drop(&mut self) {} //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
    }
}

mod explicit_call_pin_drop {
    struct Foo;

    impl Drop for Foo {
        fn drop(&mut self) {
            Drop::pin_drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
}

mod explicit_call_drop {
    struct Foo;

    impl Drop for Foo {
        fn pin_drop(&pin mut self) {
            Drop::drop(todo!()); //~ ERROR explicit use of destructor method [E0040]
        }
    }
}

fn main() {}
