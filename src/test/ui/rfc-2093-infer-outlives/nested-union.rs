#![feature(rustc_attrs)]
#![feature(untagged_unions)]
#![allow(unions_with_drop_fields)]

#[rustc_outlives]
union Foo<'a, T> { //~ ERROR rustc_outlives
    field1: Bar<'a, T>
}

// Type U needs to outlive lifetime 'b
union Bar<'b, U> {
    field2: &'b U
}

fn main() {}
