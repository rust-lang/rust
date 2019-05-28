// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

use test_macros::empty_attr as empty_helper;

#[empty_helper] //~ ERROR `empty_helper` is ambiguous
#[derive(Empty)]
struct S {
    // FIXME No ambiguity, attributes in non-macro positions are not resolved properly
    #[empty_helper]
    field: [u8; {
        // FIXME No ambiguity, derive helpers are not put into scope for non-attributes
        use empty_helper;

        // FIXME No ambiguity, derive helpers are not put into scope for inner items
        #[empty_helper]
        struct U;

        mod inner {
            #[empty_helper] //~ ERROR attribute `empty_helper` is currently unknown
            struct V;
        }

        0
    }]
}

fn main() {
    let s = S { field: [] };
}
