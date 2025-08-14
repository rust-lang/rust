// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g., if `Add` changes to `Addition`, and
// `no_implicit_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

#[no_implicit_prelude]
mod foo {
    mod baz {
        struct Test;
        impl Add for Test {} //~ ERROR cannot find trait `Add` in this scope
        impl Clone for Test {} //~ ERROR expected trait, found derive macro `Clone`
        impl Iterator for Test {} //~ ERROR cannot find trait `Iterator` in this scope
        impl ToString for Test {} //~ ERROR cannot find trait `ToString` in this scope
        impl Writer for Test {} //~ ERROR cannot find trait `Writer` in this scope

        fn foo() {
            drop(2) //~ ERROR cannot find function `drop` in this scope
        }
    }

    struct Test;
    impl Add for Test {} //~ ERROR cannot find trait `Add` in this scope
    impl Clone for Test {} //~ ERROR expected trait, found derive macro `Clone`
    impl Iterator for Test {} //~ ERROR cannot find trait `Iterator` in this scope
    impl ToString for Test {} //~ ERROR cannot find trait `ToString` in this scope
    impl Writer for Test {} //~ ERROR cannot find trait `Writer` in this scope

    fn foo() {
        drop(2) //~ ERROR cannot find function `drop` in this scope
    }
}

fn qux() {
    #[no_implicit_prelude]
    mod qux_inner {
        struct Test;
        impl Add for Test {} //~ ERROR cannot find trait `Add` in this scope
        impl Clone for Test {} //~ ERROR expected trait, found derive macro `Clone`
        impl Iterator for Test {} //~ ERROR cannot find trait `Iterator` in this scope
        impl ToString for Test {} //~ ERROR cannot find trait `ToString` in this scope
        impl Writer for Test {} //~ ERROR cannot find trait `Writer` in this scope

        fn foo() {
            drop(2) //~ ERROR cannot find function `drop` in this scope
        }
    }
}


fn main() {
    // these should work fine
    drop(2)
}
