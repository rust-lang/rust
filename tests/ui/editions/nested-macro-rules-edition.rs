// This checks the behavior of how nested macro_rules definitions are handled
// with regards to edition spans. Prior to https://github.com/rust-lang/rust/pull/133274,
// the compiler would compile the inner macro with the edition of the local crate.
// Afterwards, it uses the edition where the macro was *defined*.
//
// Unfortunately macro_rules compiler discards the edition of any *input* that
// was used to generate the macro. This is possibly not the behavior that we
// want. If we want to keep with the philosophy that code should follow the
// edition rules of the crate where it is written, then presumably we would
// want the input tokens to retain the edition of where they were written.
//
// See https://github.com/rust-lang/rust/issues/135669 for more.
//
// This has two revisions, one where local=2021 and the dep=2024. The other
// revision is vice-versa.

//@ revisions: e2021 e2024
//@[e2021] edition:2021
//@[e2024] edition:2024
//@[e2021] aux-crate: nested_macro_rules_dep=nested_macro_rules_dep_2024.rs
//@[e2024] aux-crate: nested_macro_rules_dep=nested_macro_rules_dep_2021.rs
//@[e2024] check-pass

mod with_input {
    // If we change the macro_rules input behavior, then this should pass when
    // local edition is 2021 because `gen` is written in a context with 2021
    // behavior. For local edition 2024, the reverse would be true and this
    // should fail.
    nested_macro_rules_dep::make_macro_with_input!{gen}
    macro_inner_input!{}
    //[e2021]~^ ERROR found reserved keyword
}
mod no_input {
    nested_macro_rules_dep::make_macro!{}
    macro_inner!{}
    //[e2021]~^ ERROR found reserved keyword
}

fn main() {}
