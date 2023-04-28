// check-pass

// Make sure we don't have any false positives about the "struct is never constructed" lint.

#![deny(dead_code)]

struct Foo {
    #[allow(dead_code)]
    inner: u32,
}

impl From<u32> for Foo {
    fn from(inner: u32) -> Self {
        Self { inner }
    }
}

fn main() {}
