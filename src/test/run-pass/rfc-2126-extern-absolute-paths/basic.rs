// aux-build:xcrate.rs
// edition:2018

use xcrate::Z;

fn f() {
    use xcrate;
    use xcrate as ycrate;
    let s = xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = ycrate::Z;
    assert_eq!(format!("{:?}", z), "Z");
}

fn main() {
    let s = ::xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = Z;
    assert_eq!(format!("{:?}", z), "Z");
}
