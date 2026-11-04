#![crate_type = "lib"]

#[lint::mac] //~ ERROR cannot find module or crate `lint` in this scope
struct Foo;
