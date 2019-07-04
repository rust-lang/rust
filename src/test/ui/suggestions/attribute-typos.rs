#[deprcated] //~ ERROR attribute `deprcated` is currently unknown
fn foo() {}

#[tests] //~ ERROR attribute `tests` is currently unknown to the compiler
fn bar() {}

#[rustc_err]
//~^ ERROR attribute `rustc_err` is currently unknown
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

fn main() {}
