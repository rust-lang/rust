#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
union Foo<'b, U: Copy> { //~ ERROR rustc_dump_inferred_outlives
    bar: Bar<'b, U>
}

#[derive(Clone, Copy)]
union Bar<'a, T: Copy> where T: 'a {
    x: &'a (),
    y: T,
}

fn main() {}
