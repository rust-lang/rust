// build-pass

// compile-flags: --crate-type lib

#![warn(unconditional_panic)]

pub struct Fixed64(i64);

pub fn div(f: Fixed64) {
    f.0 / 0; //~ WARN will panic at runtime
}
