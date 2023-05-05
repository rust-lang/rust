// revisions: full min

#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

enum Foo<const N: isize> {
    Variant = N,
    //~^ ERROR: generic parameters may not be used in enum discriminant values
}

enum Owo<const N: isize> {
    Variant = { N + 1 },
    //~^ ERROR: generic parameters may not be used in enum discriminant values
}

#[repr(isize)]
enum Bar<T> {
    Variant = { std::mem::size_of::<T>() as isize },
    Other(T), //~^ ERROR: generic parameters may not be used in enum discriminant values
}

#[repr(isize)]
enum UwU<'a> {
    Variant = {
        let a: &'a ();
        //~^ ERROR: generic parameters may not be used in enum discriminant values
        10_isize
    },
    Other(&'a ()),
}

fn main() {}
