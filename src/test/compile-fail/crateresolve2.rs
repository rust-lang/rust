// aux-build:crateresolve2-1.rs
// aux-build:crateresolve2-2.rs
// aux-build:crateresolve2-3.rs
// error-pattern:using multiple versions of crate `crateresolve2`

use crateresolve2(vers = "0.1");

mod m {
    use crateresolve2(vers = "0.2");
}

fn main() {
    let x: int = false;
}
