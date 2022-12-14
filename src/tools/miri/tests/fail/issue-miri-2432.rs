#![allow(where_clauses_object_safety)]

trait Trait {}

trait X {
    fn foo(&self)
    where
        Self: Trait;
}

impl X for () {
    fn foo(&self) {}
}

impl Trait for dyn X {}

pub fn main() {
    <dyn X as X>::foo(&()); //~ERROR: trying to call something that is not a method
}
