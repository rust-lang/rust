// aux-build: reexport-primitive.rs
// compile-flags:--extern foo --edition 2018

pub mod p {
    pub use foo::bar::*;
}
