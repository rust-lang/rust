// build-pass (FIXME(62277): could be check-pass?)
// aux-build:local_inner_macros.rs

extern crate local_inner_macros;

use local_inner_macros::{public_macro, public_macro_dynamic};

public_macro!();

macro_rules! local_helper {
    () => ( struct Z; )
}

public_macro_dynamic!(local_helper);

fn main() {
    let s = S;
    let z = Z;
}
