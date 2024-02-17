//@ revisions: e2024 none
//@[e2024] compile-flags: --edition 2024 -Zunstable-options

gen fn foo() {}
//[none]~^ ERROR: expected one of `#`, `async`, `const`, `default`, `extern`, `fn`, `pub`, `unsafe`, or `use`, found `gen`
//[e2024]~^^ ERROR: gen blocks are experimental

fn main() {}
