#![feature(crate_in_paths)]

use crate::m::f;
use crate as root;

mod m {
    pub fn f() -> u8 { 1 }
    pub fn g() -> u8 { 2 }
    pub fn h() -> u8 { 3 }

    // OK, visibilities are implicitly absolute like imports
    pub(in crate::m) struct S;
}

mod n
{
    use crate::m::f;
    use crate as root;
    pub fn check() {
        assert_eq!(f(), 1);
        assert_eq!(crate::m::g(), 2);
        assert_eq!(root::m::h(), 3);
    }
}

fn main() {
    assert_eq!(f(), 1);
    assert_eq!(crate::m::g(), 2);
    assert_eq!(root::m::h(), 3);
    n::check();
}
