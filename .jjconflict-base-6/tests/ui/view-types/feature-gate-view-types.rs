//@ compile-flags: -Zno-analysis

use std::view::view_type;

struct Foo {
    bar: (),
    baz: (),
}

type FooBar = view_type!(Foo.{ bar });
//~^ ERROR use of unstable library feature `view_type_macro`
type FooBaz = view_type!(Foo.{ baz });
//~^ ERROR use of unstable library feature `view_type_macro`
