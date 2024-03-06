//@ check-pass
//@ compile-flags: -W rust-2018-compatibility
//@ error-pattern: `try` is a keyword in the 2018 edition

fn main() {}

mod lint_pre_expansion_extern_module_aux;
