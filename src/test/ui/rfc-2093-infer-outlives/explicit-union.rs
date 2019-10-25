#![feature(rustc_attrs)]
#![feature(untagged_unions)]
#![allow(unions_with_drop_fields)]

#[rustc_outlives]
union Foo<'b, U> { //~ ERROR rustc_outlives
    bar: Bar<'b, U>
}

union Bar<'a, T> where T: 'a {
    x: &'a (),
    y: T,
}

fn main() {}
