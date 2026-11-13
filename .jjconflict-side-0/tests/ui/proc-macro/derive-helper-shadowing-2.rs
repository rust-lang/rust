// If a derive macro introduces a helper attribute with the same name as that macro,
// then make sure that it's usable without ambiguities.

//@ check-pass
//@ proc-macro: derive-helper-shadowing-2.rs

#[macro_use]
extern crate derive_helper_shadowing_2;

#[derive(same_name)]
struct S {
    #[same_name] // OK, no ambiguity, derive helpers have highest priority
    field: u8,
}

fn main() {}
