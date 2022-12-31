// check-pass

#![deny(where_clauses_object_safety)]

pub trait Trait {
    fn method(&self) where Self: Sync;
}

fn main() {}
