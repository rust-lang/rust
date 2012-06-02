// xfail-win32 - not sure why this is busted
// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_borrow_lib.rs

use cci_borrow_lib;
import cci_borrow_lib::foo;

fn main() {
    let p = @22u;
    let r = foo(p);
    #debug["r=%u", r];
    assert r == 22u;
}
