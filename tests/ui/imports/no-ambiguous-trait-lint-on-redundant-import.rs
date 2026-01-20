//@ check-pass
// The AMBIGUOUS_GLOB_IMPORTED_TRAITS lint is reported on uses of traits that are
// ambiguously glob imported. This test checks that we don't report this lint
// when the same trait is glob imported multiple times.

mod t {
    pub trait Trait {
        fn method(&self) {}
    }

    impl Trait for i8 {}
}

mod m1 {
    pub use t::Trait;
}

mod m2 {
    pub use t::Trait;
}

use m1::*;
use m2::*;

fn main() {
    0i8.method();
}
