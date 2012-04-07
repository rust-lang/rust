// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
#[link(name = "crateresolve4b", vers = "0.1")];
#[crate_type = "lib"];

use crateresolve4a(vers="0.2");

fn f() -> int { crateresolve4a::g() }
