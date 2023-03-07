#![warn(clippy::forget_non_drop)]

use core::mem::forget;

fn forget_generic<T>(t: T) {
    // Don't lint
    forget(t)
}

fn main() {
    struct Foo;
    // Lint
    forget(Foo);

    struct Bar;
    impl Drop for Bar {
        fn drop(&mut self) {}
    }
    // Don't lint
    forget(Bar);

    struct Baz<T>(T);
    // Lint
    forget(Baz(Foo));
    // Don't lint
    forget(Baz(Bar));
}
