// xfail-fast
// aux-build:crateresolve_calories-1.rs
// aux-build:crateresolve_calories-2.rs
// aux-build:crateresolve7x.rs

extern mod crateresolve7x;

fn main() {
    assert crateresolve7x::a::f() == 100;
    assert crateresolve7x::b::f() == 200;
}
