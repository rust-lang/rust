#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]
#![feature(untagged_unions)]
#![allow(unions_with_drop_fields)]


#[rustc_outlives]
union Foo<'a, T> { //~ ERROR 18:1: 20:2: rustc_outlives
    field1: Bar<'a, T>
}

// Type U needs to outlive lifetime 'b
union Bar<'b, U> {
    field2: &'b U
}

fn main() {}
