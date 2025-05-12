//@ run-pass
//@ compile-flags: -C codegen-units=3
//@ aux-build:sepcomp_cci_lib.rs

// Test accessing cross-crate inlined items from multiple compilation units.


extern crate sepcomp_cci_lib;
use sepcomp_cci_lib::{cci_fn, CCI_CONST};

fn call1() -> usize {
    cci_fn() + CCI_CONST
}

mod a {
    use sepcomp_cci_lib::{cci_fn, CCI_CONST};
    pub fn call2() -> usize {
        cci_fn() + CCI_CONST
    }
}

mod b {
    use sepcomp_cci_lib::{cci_fn, CCI_CONST};
    pub fn call3() -> usize {
        cci_fn() + CCI_CONST
    }
}

fn main() {
    assert_eq!(call1(), 1234);
    assert_eq!(a::call2(), 1234);
    assert_eq!(b::call3(), 1234);
}
