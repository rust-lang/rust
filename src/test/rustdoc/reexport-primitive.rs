// aux-build: reexport-primitive.rs
// compile-flags:--extern foo --edition 2018

#![crate_name = "bar"]

// @has bar/p/index.html
// @has - 'bool'
pub mod p {
    pub use foo::bar::*;
}
