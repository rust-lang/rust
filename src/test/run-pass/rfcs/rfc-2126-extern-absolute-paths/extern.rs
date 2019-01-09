// run-pass
#![allow(dead_code)]
// aux-build:xcrate.rs
// compile-flags:--extern xcrate

#![feature(extern_in_paths)]

use extern::xcrate::Z;

type A = extern::xcrate::S;
type B = for<'a> extern::xcrate::Tr<'a>;

fn f() {
    use extern::xcrate;
    use extern::xcrate as ycrate;
    let s = xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = ycrate::Z;
    assert_eq!(format!("{:?}", z), "Z");
}

fn main() {
    let s = extern::xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = Z;
    assert_eq!(format!("{:?}", z), "Z");
    assert_eq!(A {}, extern::xcrate::S {});
}
