// run-pass
macro_rules! foo {
    () => {
        struct Bar;
        struct Baz;
    }
}

macro_rules! grault {
    () => {
        foo!();
        struct Xyzzy;
    }
}

fn static_assert_exists<T>() { }

fn main() {
    grault!();
    static_assert_exists::<Bar>();
    static_assert_exists::<Baz>();
    static_assert_exists::<Xyzzy>();
}
