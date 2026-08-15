//! Regression test to check that literal expressions in a struct field can be coerced to the
//! expected field type, including block expressions.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/31260>

//@ check-pass

pub struct Struct<K: 'static> {
    pub field: K,
}

static STRUCT: Struct<&'static [u8]> = Struct { field: { &[1] } };

static STRUCT2: Struct<&'static [u8]> = Struct { field: &[1] };

fn main() {}
