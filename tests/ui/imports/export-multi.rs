//@ run-pass

use m::f;
use m::g;

mod m {
    pub fn f() { }
    pub fn g() { }
}

pub fn main() { f(); g(); m::f(); m::g(); }
