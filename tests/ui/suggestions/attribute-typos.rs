#[deprcated] //~ ERROR cannot find attribute `deprcated`
fn foo() {}

#[tests] //~ ERROR cannot find attribute `tests`
fn bar() {}

#[rustc_err]
//~^ ERROR cannot find attribute `rustc_err`
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

fn main() {}
