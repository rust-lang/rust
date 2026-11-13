// Issue #7526: lowercase static constants in patterns look like bindings

#![allow(dead_code)]
#![deny(non_upper_case_globals)]

#[allow(non_upper_case_globals)]
pub const a : isize = 97;

fn f() {
    let r = match (0,0) {
        (0, a) => 0,
        //~^ ERROR constant in pattern `a` should have an upper case name
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 1);
}

mod m {
    #[allow(non_upper_case_globals)]
    pub const aha : isize = 7;
}

fn g() {
    use self::m::aha;
    let r = match (0,0) {
        (0, aha) => 0,
        //~^ ERROR constant in pattern `aha` should have an upper case name
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
}

mod n {
    pub const OKAY : isize = 8;
}

fn h() {
    use self::n::OKAY as not_okay;
    let r = match (0,0) {
        (0, not_okay) => 0,
//~^ ERROR constant in pattern `not_okay` should have an upper case name
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
}

fn main () {
    f();
    g();
    h();
}
