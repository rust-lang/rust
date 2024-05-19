#[deprcated] //~ ERROR cannot find attribute `deprcated` in this scope
fn foo() {}

#[tests] //~ ERROR cannot find attribute `tests` in this scope
fn bar() {}

#[rustc_err]
//~^ ERROR cannot find attribute `rustc_err` in this scope
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

fn main() {}
