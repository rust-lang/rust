// aux-build:derive-helper-shadowing.rs

extern crate derive_helper_shadowing;
use derive_helper_shadowing::*;

#[my_attr] //~ ERROR `my_attr` is ambiguous
#[derive(MyTrait)]
struct S {
    // FIXME No ambiguity, attributes in non-macro positions are not resolved properly
    #[my_attr]
    field: [u8; {
        // FIXME No ambiguity, derive helpers are not put into scope for non-attributes
        use my_attr;

        // FIXME No ambiguity, derive helpers are not put into scope for inner items
        #[my_attr]
        struct U;

        mod inner {
            #[my_attr] //~ ERROR attribute `my_attr` is currently unknown
            struct V;
        }

        0
    }]
}

fn main() {
    let s = S { field: [] };
}
