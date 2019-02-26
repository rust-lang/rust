// Test that cfg_attr doesn't emit any attributes when the
// configuration variable is false. This mirrors `cfg-attr-multi-true.rs`

// compile-pass

#![warn(unused_must_use)]

#[cfg_attr(any(), deprecated, must_use)]
struct Struct {}

impl Struct {
    fn new() -> Struct {
        Struct {}
    }
}

fn main() {
    Struct::new();
}
