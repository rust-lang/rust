//@ run-rustfix

// Regression test for https://github.com/rust-lang/rust/issues/144319.
// Function-local constructors cannot be named through the enclosing function
// path. The suggestion must omit path segments that cannot be written in source,
// while preserving real path segments like local modules and enums.

#![allow(dead_code)]

fn direct_tuple_struct() {
    struct Foo(bool);
    struct Bar(Foo);

    _ = Bar(false);
    //~^ ERROR mismatched types
    //~| HELP try wrapping the expression in `Foo`
}

fn enum_variant() {
    enum LocalResult<T> {
        Ok(T),
    }
    struct Bar(LocalResult<bool>);

    _ = Bar(false);
    //~^ ERROR mismatched types
    //~| HELP try wrapping the expression in `LocalResult::Ok`
}

fn local_module() {
    mod inner {
        pub struct Foo(pub bool);
    }
    struct Bar(inner::Foo);

    _ = Bar(false);
    //~^ ERROR mismatched types
    //~| HELP try wrapping the expression in `inner::Foo`
}

fn closure_body() {
    let _ = || {
        struct Foo(bool);
        struct Bar(Foo);

        _ = Bar(false);
        //~^ ERROR mismatched types
        //~| HELP try wrapping the expression in `Foo`
    };
}

fn inline_const_block() {
    const {
        struct Foo(bool);
        struct Bar(Foo);

        _ = Bar(false);
        //~^ ERROR mismatched types
        //~| HELP try wrapping the expression in `Foo`
    };
}

pub fn main() {}
