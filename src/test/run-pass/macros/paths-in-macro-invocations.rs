// run-pass
#![allow(dead_code)]
// aux-build:two_macros.rs

extern crate two_macros;

::two_macros::macro_one!();
two_macros::macro_one!();

mod foo { pub use two_macros::macro_one as bar; }

trait T {
    foo::bar!();
    ::foo::bar!();
}

struct S {
    x: foo::bar!(i32),
    y: ::foo::bar!(i32),
}

impl S {
    foo::bar!();
    ::foo::bar!();
}

fn main() {
    foo::bar!();
    ::foo::bar!();

    let _ = foo::bar!(0);
    let _ = ::foo::bar!(0);

    let foo::bar!(_) = 0;
    let ::foo::bar!(_) = 0;
}
