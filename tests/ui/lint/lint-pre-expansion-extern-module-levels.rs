//! Lint level attributes on the ancestors of an out-of-line module apply to the
//! pre-expansion lints of that module, the same as they do for an inline module.

//@ revisions: crate_warn crate_allow parent_warn
//@ check-pass
//@ edition: 2015
//@[crate_allow] compile-flags: -W keyword-idents-2018

#![cfg_attr(crate_warn, warn(keyword_idents_2018))]
#![cfg_attr(crate_allow, allow(keyword_idents_2018))]

fn main() {}

#[cfg_attr(parent_warn, warn(keyword_idents_2018))]
mod lint_pre_expansion_extern_module_levels_aux;

//[crate_warn,parent_warn]~? WARN `try` is a keyword in the 2018 edition
//[crate_warn,parent_warn]~? WARN this is accepted in the current edition
//[crate_warn,parent_warn]~? WARN `async` is a keyword in the 2018 edition
//[crate_warn,parent_warn]~? WARN this is accepted in the current edition
