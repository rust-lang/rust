// run-pass
// pretty-expanded FIXME #23616

use m::f;
use m::g;

mod m {
    pub fn f() { }
    pub fn g() { }
}

pub fn main() { f(); g(); m::f(); m::g(); }
