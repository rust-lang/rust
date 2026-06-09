#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct GenericStruct<const T: usize> { val: i64 }

impl<const T: usize> From<GenericStruct<T>> for GenericStruct<{T + 1}> {
//~^ ERROR: conflicting implementations of trait `From<GenericStruct<_>>` for type `GenericStruct<_>`
    fn from(other: GenericStruct<T>) -> Self {
        Self { val: other.val }
    }
}

impl<const T: usize> From<GenericStruct<{T + 1}>> for GenericStruct<T> {
//~^ ERROR: conflicting implementations of trait `From<GenericStruct<_>>` for type `GenericStruct<_>`
    fn from(other: GenericStruct<{T + 1}>) -> Self {
        Self { val: other.val }
    }
}

fn main() {}
