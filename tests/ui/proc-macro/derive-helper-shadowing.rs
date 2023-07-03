// edition:2018
// aux-build:test-macros.rs
// aux-build:derive-helper-shadowing.rs

#[macro_use]
extern crate test_macros;
#[macro_use]
extern crate derive_helper_shadowing;

use test_macros::empty_attr as empty_helper;

macro_rules! gen_helper_use {
    () => {
        #[empty_helper] //~ ERROR cannot find attribute `empty_helper` in this scope
        struct W;
    }
}

#[empty_helper] //~ ERROR `empty_helper` is ambiguous
                //~| WARN derive helper attribute is used before it is introduced
                //~| WARN this was previously accepted
#[derive(Empty)]
struct S {
    #[empty_helper] // OK, no ambiguity, derive helpers have highest priority
    field: [u8; {
        use empty_helper; // OK, no ambiguity, derive helpers have highest priority

        #[empty_helper] // OK, no ambiguity, derive helpers have highest priority
        struct U;

        mod inner {
            // OK, no ambiguity, the non-helper attribute is not in scope here, only the helper.
            #[empty_helper]
            struct V;

            gen_helper_use!();

            #[derive(GenHelperUse)] //~ ERROR cannot find attribute `empty_helper` in this scope
            struct Owo;

            use empty_helper as renamed;
            #[renamed] //~ ERROR cannot use a derive helper attribute through an import
            struct Wow;
        }

        0
    }]
}

// OK, no ambiguity, only the non-helper attribute is in scope.
#[empty_helper]
struct Z;

fn main() {
    let s = S { field: [] };
}
