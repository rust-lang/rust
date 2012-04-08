// xfail-fast
// aux-build:crateresolve6-1.rs
// aux-build:crateresolve6-2.rs
// aux-build:crateresolve7x.rs

use crateresolve7x;

fn main() {
    assert crateresolve7x::a::f() == 100;
    assert crateresolve7x::b::f() == 200;
}
