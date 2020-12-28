// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g. if `Add` changes to `Addition`, and
// `no_implicit_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

#[no_implicit_prelude]
//~^ WARNING: deprecated
//~^^ WARNING: deprecated
mod foo {
    mod baz {
        struct Test;
        impl Add for Test {} //~ ERROR: not in scope
        impl Clone for Test {} //~ ERROR: not in scope
        impl Iterator for Test {} //~ ERROR: not in scope
        impl ToString for Test {} //~ ERROR: not in scope
        impl Writer for Test {} //~ ERROR: not in scope

        fn foo() {
            drop(2) //~ ERROR: unresolved name
        }
    }

    struct Test;
    impl Add for Test {} //~ ERROR: not in scope
    impl Clone for Test {} //~ ERROR: not in scope
    impl Iterator for Test {} //~ ERROR: not in scope
    impl ToString for Test {} //~ ERROR: not in scope
    impl Writer for Test {} //~ ERROR: not in scope

    fn foo() {
        drop(2) //~ ERROR: unresolved name
    }
}

fn qux() {
    #[no_implicit_prelude]
    //~^ WARNING: deprecated
    //~^^ WARNING: deprecated
    mod qux_inner {
        struct Test;
        impl Add for Test {} //~ ERROR: not in scope
        impl Clone for Test {} //~ ERROR: not in scope
        impl Iterator for Test {} //~ ERROR: not in scope
        impl ToString for Test {} //~ ERROR: not in scope
        impl Writer for Test {} //~ ERROR: not in scope

        fn foo() {
            drop(2) //~ ERROR: unresolved name
        }
    }
}

fn main() {
    // these should work fine
    drop(2)
}
