//@ check-pass
#![allow(dead_code)]

trait Trait {
        fn method(self) -> isize;
}

struct Wrapper<T> {
        field: T
}

impl<'a, T> Trait for &'a Wrapper<T> where &'a T: Trait {
    fn method(self) -> isize {
        let r: &'a T = &self.field;
        Trait::method(r); // these should both work
        r.method()
    }
}

fn main() {}
