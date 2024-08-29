//@ check-pass

pub trait Trait {
    fn method(&self) where Self: Sync;
}

fn main() {}
