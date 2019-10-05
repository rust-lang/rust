// edition:2018
// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

use test_macros::empty_attr as empty_helper;

#[empty_helper] //~ ERROR `empty_helper` is ambiguous
#[derive(Empty)]
struct S {
    #[empty_helper] //~ ERROR `empty_helper` is ambiguous
    field: [u8; {
        use empty_helper; //~ ERROR `empty_helper` is ambiguous

        #[empty_helper] //~ ERROR `empty_helper` is ambiguous
        struct U;

        mod inner {
            // OK, no ambiguity, the non-helper attribute is not in scope here, only the helper.
            #[empty_helper]
            struct V;
        }

        0
    }]
}

fn main() {
    let s = S { field: [] };
}
