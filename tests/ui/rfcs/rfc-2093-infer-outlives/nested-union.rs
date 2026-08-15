#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
union Foo<'a, T: Copy> { //~ ERROR rustc_dump_inferred_outlives
    field1: Bar<'a, T>
}

// Type U needs to outlive lifetime 'b
#[derive(Clone, Copy)]
union Bar<'b, U: Copy> {
    field2: &'b U
}

fn main() {}
