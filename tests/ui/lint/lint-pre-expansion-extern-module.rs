//@ check-pass
//@ compile-flags: -W rust-2018-compatibility

fn main() {}

mod lint_pre_expansion_extern_module_aux;

//~? WARN `try` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
