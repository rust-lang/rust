//@ revisions: has_support no_support
//@[no_support] ignore-enzyme
//@[has_support] needs-enzyme

#![crate_type = "lib"]

// This checks that without the autodiff feature enabled, we can't use it.

#[autodiff(dfoo, Reverse)]
//[has_support]~^ ERROR cannot find attribute `autodiff` in this scope
//[no_support]~^^ ERROR cannot find attribute `autodiff` in this scope
fn foo() {}
