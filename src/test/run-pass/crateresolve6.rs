// xfail-fast
// aux-build:crateresolve6-1.rs
// aux-build:crateresolve6-2.rs
// error-pattern:mismatched types

// These both have the same version but differ in other metadata
use cr6_1 (name = "crateresolve6", vers = "0.1", calories="100");
use cr6_2 (name = "crateresolve6", vers = "0.1", calories="200");

fn main() {
    assert cr6_1::f() == 100;
    assert cr6_2::f() == 200;
}
