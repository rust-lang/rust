// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
#[link(name = "crateresolve4b", vers = "0.2")];
#[crate_type = "lib"];

extern mod crateresolve4a(vers="0.1");

fn g() -> int { crateresolve4a::f() }
