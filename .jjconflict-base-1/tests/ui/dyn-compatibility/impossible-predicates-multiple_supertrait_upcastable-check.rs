//@ check-pass
// issue: rust-lang/rust#106247

pub trait Trait {
    fn method(&self) where Self: Sync;
}

fn main() {}
