#![feature(type_alias_impl_trait)]

// check-pass

use std::fmt::Debug;

// No need to report the `type_alias_bounds` lint, as
// the moment an opaque type is mentioned, we actually do check
// type alias bounds.
type Foo<T: Debug> = (impl Debug, usize);

fn foo<U: Debug>() -> Foo<U> {
    (Vec::<U>::new(), 1234)
}

fn main() {}
