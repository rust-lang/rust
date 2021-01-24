// check-pass

#![allow(dead_code, unused)]

///! This isn't supposed to work, but it accidentally did, so unfortunately we need to support this.

struct Tricky;

impl<T: std::fmt::Debug> From<T> for Tricky {
    fn from(_: T) -> Tricky { Tricky }
}

fn foo() -> Result<(), Tricky> {
    None?;
    Ok(())
}

fn main() {
    assert!(matches!(foo(), Err(Tricky)));
}
