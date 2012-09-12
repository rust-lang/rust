// xfail-fast
// aux-build:crateresolve_calories-1.rs
// aux-build:crateresolve_calories-2.rs
// error-pattern:mismatched types

// These both have the same version but differ in other metadata
extern mod cr6_1 (name = "crateresolve_calories", vers = "0.1", calories="100");
extern mod cr6_2 (name = "crateresolve_calories", vers = "0.1", calories="200");

fn main() {
    assert cr6_1::f() == 100;
    assert cr6_2::f() == 200;
}
