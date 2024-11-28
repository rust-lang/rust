//@ revisions: e2024 none
//@[e2024] edition: 2024

gen fn foo() {}
//[none]~^ ERROR: expected one of `#`, `async`, `const`, `default`, `extern`, `fn`, `pub`, `safe`, `unsafe`, or `use`, found `gen`
//[e2024]~^^ ERROR: gen blocks are experimental

fn main() {}
