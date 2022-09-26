// check-fail
// known-bug: #88460

// This should pass, but has a missed normalization due to HRTB.

pub trait Marker {}

pub trait Trait {
    type Assoc<'a>;
}

fn test<T>(value: T)
where
    T: Trait,
    for<'a> T::Assoc<'a>: Marker,
{
}

impl Marker for () {}

struct Foo;

impl Trait for Foo {
    type Assoc<'a> = ();
}

fn main() {
    test(Foo);
}
