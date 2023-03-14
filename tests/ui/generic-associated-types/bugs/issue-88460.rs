// check-pass

// Used to fail due to a missed normalization due to HRTB.

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
