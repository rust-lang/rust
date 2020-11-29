// aux-build:self.rs

extern crate cross_crate_self;

// @has self/struct.S.html '//a[@href="../self/struct.S.html#method.f"]' "Self::f"
pub use cross_crate_self::S;
