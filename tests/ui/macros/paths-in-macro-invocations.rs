//@ run-pass
#![allow(dead_code)]
//@ aux-build:two_macros-rpass.rs

extern crate two_macros_rpass as two_macros;

::two_macros::macro_one!();
two_macros::macro_one!();

mod foo { pub use two_macros::macro_one as bar; }

trait T {
    foo::bar!();
    crate::foo::bar!();
}

struct S {
    x: foo::bar!(i32),
    y: crate::foo::bar!(i32),
}

impl S {
    foo::bar!();
    crate::foo::bar!();
}

fn main() {
    foo::bar!();
    crate::foo::bar!();

    let _ = foo::bar!(0);
    let _ = crate::foo::bar!(0);

    let foo::bar!(_) = 0;
    let crate::foo::bar!(_) = 0;
}
