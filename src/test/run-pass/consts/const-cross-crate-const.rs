// run-pass
// aux-build:cci_const.rs
#![allow(non_upper_case_globals)]

extern crate cci_const;
static foo: &'static str = cci_const::foopy;
static a: usize = cci_const::uint_val;
static b: usize = cci_const::uint_expr + 5;

pub fn main() {
    assert_eq!(a, 12);
    let foo2 = a;
    assert_eq!(foo2, cci_const::uint_val);
    assert_eq!(b, cci_const::uint_expr + 5);
    assert_eq!(foo, cci_const::foopy);
}
