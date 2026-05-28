#[deprcated] //~ ERROR cannot find attribute `deprcated` in this scope
fn foo() {}

#[tests] //~ ERROR cannot find attribute `tests` in this scope
fn bar() {}

#[rustc_dumm]
//~^ ERROR cannot find attribute `rustc_dumm` in this scope
//~| ERROR attributes starting with `rustc` are reserved for use by the `rustc` compiler

// Regression test for https://github.com/rust-lang/rust/issues/150566.
#[cfg_trace] //~ ERROR cannot find attribute `cfg_trace` in this scope
fn cfg_trace_attr() {}

#[cfg_attr_trace] //~ ERROR cannot find attribute `cfg_attr_trace` in this scope
fn cfg_attr_trace_attr() {}

fn main() {}
