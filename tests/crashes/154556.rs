//@ known-bug: #154556
#![feature(generic_const_exprs)]

pub trait Foo {
    // Has to take self or a reference to self to ICE.
    fn eq(self);
}

pub trait Bar {
    const NUMBER: usize;
}

impl<T: Bar> Foo for T
where
    [(); T::NUMBER]: Sized, // Bound required for ICE
{
    fn eq(self) {}
}

// As long as the method called below has the same name as the one defined in Foo, the ICE happens.
fn baz() {
    ().eq(&());
}

fn main() {}
