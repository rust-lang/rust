// xfail-fast
// aux-build:crateresolve-1.rs
// aux-build:crateresolve-2.rs
// aux-build:crateresolve-3.rs

use crateresolve(vers = "0.2");

fn main() {
    assert crateresolve::f() == 20;
}
