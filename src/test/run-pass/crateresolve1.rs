// xfail-fast
// aux-build:crateresolve1-1.rs
// aux-build:crateresolve1-2.rs
// aux-build:crateresolve1-3.rs

use crateresolve1(vers = "0.2");

fn main() {
    assert crateresolve1::f() == 20;
}
