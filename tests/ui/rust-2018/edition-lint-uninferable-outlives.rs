//@ build-pass (FIXME(62277): could be check-pass?)

#![allow(unused)]
#![deny(explicit_outlives_requirements)]

// A case where we can't infer the outlives requirement. Example copied from
// RFC 2093.
// (https://rust-lang.github.io/rfcs/2093-infer-outlives.html
// #where-explicit-annotations-would-still-be-required)


trait MakeRef<'a> {
    type Type;
}

impl<'a, T> MakeRef<'a> for Vec<T>
    where T: 'a  // still required
{
    type Type = &'a T;
}


struct Foo<'a, T>
    where T: 'a  // still required, not inferred from `field`
{
    field: <Vec<T> as MakeRef<'a>>::Type
}


fn main() {}
