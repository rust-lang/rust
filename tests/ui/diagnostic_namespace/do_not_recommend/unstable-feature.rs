//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![deny(unknown_or_malformed_diagnostic_attributes)]
trait Foo {}

#[diagnostic::do_not_recommend]
//~^ ERROR unknown diagnostic attribute [unknown_or_malformed_diagnostic_attributes]
impl Foo for i32 {}

fn main() {}
