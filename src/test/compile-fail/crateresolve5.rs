// xfail-fast
// aux-build:crateresolve5-1.rs
// aux-build:crateresolve5-2.rs
// error-pattern:mismatched types

extern mod cr5_1 (name = "crateresolve5", vers = "0.1");
extern mod cr5_2 (name = "crateresolve5", vers = "0.2");

fn main() {
    // Nominal types from two multiple versions of a crate are different types
    assert cr5_1::nominal() == cr5_2::nominal();
}
