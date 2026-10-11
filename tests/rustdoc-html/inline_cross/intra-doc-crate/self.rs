//@ aux-build:self.rs
//@ build-aux-docs

extern crate cross_crate_self;

//@ has self/type.S.html '//a[@href="type.S.html#method.f"]' "Self::f"
//@ has self/type.S.html '//a[@href="type.S.html"]' "Self"
//@ has self/type.S.html '//a[@href="../cross_crate_self/index.html"]' "crate"
pub use cross_crate_self::S;
