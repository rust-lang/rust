//@aux-build:proc_macros.rs
#![warn(clippy::manual_ilog2)]
#![allow(clippy::unnecessary_operation)]

use proc_macros::{external, with_span};

fn foo(a: u32, b: u64) {
    31 - a.leading_zeros(); //~ manual_ilog2
    a.ilog(2); //~ manual_ilog2

    63 - b.leading_zeros(); //~ manual_ilog2
    64 - b.leading_zeros(); // No lint because manual ilog2 is `BIT_WIDTH - 1 - x.leading_zeros()`

    // don't lint when macros are involved
    macro_rules! two {
        () => {
            2
        };
    };

    macro_rules! thirty_one {
        () => {
            31
        };
    };

    a.ilog(two!());
    thirty_one!() - a.leading_zeros();

    external!($a.ilog(2));
    with_span!(span; a.ilog(2));
}

fn wrongly_unmangled_macros() {
    struct S {
        inner: u32,
    }

    let x = S { inner: 42 };
    macro_rules! access {
        ($s:expr) => {
            $s.inner
        };
    }
    let log = 31 - access!(x).leading_zeros();
    //~^ manual_ilog2
    let log = access!(x).ilog(2);
    //~^ manual_ilog2
}
