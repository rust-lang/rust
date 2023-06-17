#![feature(type_alias_impl_trait)]

use std::fmt::{Debug, Display};

struct Struct<V: Display>(Option<V>);

// Make sure that, in contrast to type aliases without opaque types,
// we actually do a wf check for the aliased type.
type Foo<T: Debug> = (impl Debug, Struct<T>);
//~^ ERROR: `T` doesn't implement `std::fmt::Display`

fn foo<U: Debug + Display>() -> Foo<U> {
    (Vec::<U>::new(), Struct(None))
}

fn main() {}
