#![crate_type="lib"]

pub trait Remote {
    fn foo(&self) { }
}

pub trait Remote1<T> {
    fn foo(&self, _t: T) { }
}

pub trait Remote2<T, U> {
    fn foo(&self, _t: T, _u: U) { }
}

pub struct Pair<T,U>(T,U);
