#![feature(rustc_attrs)]
#![feature(untagged_unions)]

#[rustc_outlives]
union Foo<'b, U: Copy> { //~ ERROR rustc_outlives
    bar: Bar<'b, U>
}

union Bar<'a, T: Copy> where T: 'a {
    x: &'a (),
    y: T,
}

fn main() {}
