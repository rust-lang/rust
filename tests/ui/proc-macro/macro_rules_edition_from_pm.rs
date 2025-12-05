// Tests how edition hygiene works for macro_rules macros generated from a
// proc-macro.
// See https://github.com/rust-lang/rust/issues/132906

//@ proc-macro: macro_rules_edition_pm.rs
//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@ check-pass
//@ ignore-backends: gcc

// This checks how the expr fragment specifier works.
macro_rules_edition_pm::make_edition_macro!{}

const _: () = {
    assert!(edition!(const {}) == 2021);
};

// This checks how the expr fragment specifier from a nested macro.
macro_rules_edition_pm::make_nested_edition_macro!{}
make_inner!{}

const _: () = {
    assert!(edition_inner!(const {}) == 2021);
};

fn main() {}
