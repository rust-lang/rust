// FIXME this is only temporary to access the CI state.

use std::fmt::Debug;

// Allowed
fn in_parameters(_: impl Debug) {
    panic!()
}

// Allowed
fn in_return() -> impl Debug {
    panic!()
}

// Allowed
fn in_adt_in_parameters(_: Vec<impl Debug>) {
    panic!()
}

// Disallowed
//~v ERROR `impl Trait` is not allowed in `fn` pointer parameters
fn in_fn_parameter_in_parameters(_: fn(impl Debug)) {
    panic!()
}

fn main() {}
