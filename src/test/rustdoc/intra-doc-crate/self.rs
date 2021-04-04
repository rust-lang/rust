// aux-build:self.rs
// build-aux-docs

extern crate cross_crate_self;

// @has self/struct.S.html '//a[@href="../self/struct.S.html#method.f"]' "Self::f"
// @has self/struct.S.html '//a[@href="../self/struct.S.html"]' "Self"
// @has self/struct.S.html '//a[@href="../cross_crate_self/index.html"]' "crate"
pub use cross_crate_self::S;
