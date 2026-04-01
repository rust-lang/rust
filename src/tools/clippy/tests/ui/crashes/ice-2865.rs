//@ check-pass

#![allow(dead_code, clippy::extra_unused_lifetimes)]

// Test for https://github.com/rust-lang/rust-clippy/issues/2865

struct Ice {
    size: String,
}

impl<'a> From<String> for Ice {
    fn from(_: String) -> Self {
        let text = || "iceberg".to_string();
        Self { size: text() }
    }
}

fn main() {}
