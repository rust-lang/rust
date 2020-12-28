#![feature(no_prelude)]

// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g. if `Add` changes to `Addition`, and
// `no_prelude` stops working, then the `impl Add` will still
// fail with the same error message).
//
// Unlike `no_implicit_prelude`, `no_prelude` doesn't cascade into nested
// modules, this makes the impl in foo::baz work.

#[no_prelude]
mod foo {
    mod baz {
        struct Test;
        impl From<Test> for Test {
            fn from(t: Test) {
                Test
            }
        }
        impl Clone for Test {
            fn clone(&self) {
                Test
            }
        }
        impl Eq for Test {}

        fn foo() {
            drop(2)
        }
    }

    struct Test;
    impl From for Test {} //~ ERROR: cannot find trait
    impl Clone for Test {} //~ ERROR: expected trait, found derive macro
    impl Iterator for Test {} //~ ERROR: cannot find trait
    impl ToString for Test {} //~ ERROR: cannot find trait
    impl Eq for Test {} //~ ERROR: expected trait, found derive macro

    fn foo() {
        drop(2) //~ ERROR: cannot find function `drop`
    }
}

fn qux() {
    #[no_prelude]
    mod qux_inner {
        struct Test;
        impl From for Test {} //~ ERROR: cannot find trait
        impl Clone for Test {} //~ ERROR: expected trait, found derive macro
        impl Iterator for Test {} //~ ERROR: cannot find trait
        impl ToString for Test {} //~ ERROR: cannot find trait
        impl Eq for Test {} //~ ERROR: expected trait, found derive macro
        fn foo() {
            drop(2) //~ ERROR: cannot find function `drop`
        }
    }
}

fn main() {
    // these should work fine
    drop(2)
}
