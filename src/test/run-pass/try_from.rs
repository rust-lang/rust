// This test relies on `TryFrom` being blanket impl for all `T: Into`
// and `TryInto` being blanket impl for all `U: TryFrom`

// This test was added to show the motivation for doing this
// over `TryFrom` being blanket impl for all `T: From`

#![feature(never_type)]

use std::convert::{TryInto, Infallible};

struct Foo<T> {
    t: T,
}

// This fails to compile due to coherence restrictions
// as of Rust version 1.32.x, therefore it could not be used
// instead of the `Into` version of the impl, and serves as
// motivation for a blanket impl for all `T: Into`, instead
// of a blanket impl for all `T: From`
/*
impl<T> From<Foo<T>> for Box<T> {
    fn from(foo: Foo<T>) -> Box<T> {
        Box::new(foo.t)
    }
}
*/

impl<T> Into<Vec<T>> for Foo<T> {
    fn into(self) -> Vec<T> {
        vec![self.t]
    }
}

pub fn main() {
    let _: Result<Vec<i32>, Infallible> = Foo { t: 10 }.try_into();
}
