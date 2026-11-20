//@ check-pass
//@ compile-flags: -Z parse-crate-root-only

use std::ops::Deref;
use std::fmt::Debug;

trait Foo {
    type Bar<'a>;
    type Bar<'a, 'b>;
    type Bar<'a, 'b,>;
    type Bar<'a, 'b, T>;
    type Bar<'a, 'b, T, U>;
    type Bar<'a, 'b, T, U,>;
    type Bar<'a, 'b, T: Debug, U,>;
    type Bar<'a, 'b, T: Debug, U,>: Debug;
    type Bar<'a, 'b, T: Debug, U,>: Deref<Target = T> + Into<U>;
    type Bar<'a, 'b, T: Debug, U,> where T: Deref<Target = U>, U: Into<T>;
    type Bar<'a, 'b, T: Debug, U,>: Deref<Target = T> + Into<U>
        where T: Deref<Target = U>, U: Into<T>;
}

fn main() {}
