// check-pass

// compile-flags: --crate-type lib

#![warn(unconditional_panic)]

pub struct Fixed64(i64);

// HACK: this test passes only because this is a const fn that is written to metadata
pub const fn div(f: Fixed64) {
    f.0 / 0; //~ WARN will panic at runtime
}
