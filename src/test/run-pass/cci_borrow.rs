// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_borrow_lib.rs

extern mod cci_borrow_lib;
use cci_borrow_lib::foo;

fn main() {
    let p = @22u;
    let r = foo(p);
    debug!("r=%u", r);
    assert r == 22u;
}
