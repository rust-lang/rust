#[deprcated] //~ ERROR cannot find attribute macro `deprcated` in this scope
fn foo() {}

#[tests] //~ ERROR cannot find attribute macro `tests` in this scope
fn bar() {}

#[rustc_err]
//~^ ERROR cannot find attribute macro `rustc_err` in this scope
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

fn main() {}
