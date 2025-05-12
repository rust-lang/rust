#![warn(clippy::drop_non_drop)]

use core::mem::drop;

fn make_result<T>(t: T) -> Result<T, ()> {
    Ok(t)
}

#[must_use]
fn must_use<T>(t: T) -> T {
    t
}

fn drop_generic<T>(t: T) {
    // Don't lint
    drop(t)
}

fn main() {
    struct Foo;
    // Lint
    drop(Foo);
    //~^ drop_non_drop

    // Don't lint
    drop(make_result(Foo));
    // Don't lint
    drop(must_use(Foo));

    struct Bar;
    impl Drop for Bar {
        fn drop(&mut self) {}
    }
    // Don't lint
    drop(Bar);

    struct Baz<T>(T);
    // Lint
    drop(Baz(Foo));
    //~^ drop_non_drop

    // Don't lint
    drop(Baz(Bar));
}
