// xfail-fast
// aux-build:crateresolve5-1.rs
// aux-build:crateresolve5-2.rs

use cr5_1 (name = "crateresolve5", vers = "0.1");
use cr5_2 (name = "crateresolve5", vers = "0.2");

fn main() {
    // Structural types can be used between two versions of the same crate
    assert cr5_1::structural() == cr5_2::structural();
    // Make sure these are actually two different crates
    assert cr5_1::f() == 10 && cr5_2::f() == 20;
}
