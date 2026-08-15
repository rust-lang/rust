// Along with the other tests in this series, illustrates the
// "projection gap": in this test, we know that `T: 'x`, and that is
// enough to conclude that `T::Foo: 'x`.

//@ check-pass
#![allow(dead_code)]
#![allow(unused_variables)]

trait Trait1<'x> {
    type Foo;
}

// calling this fn should trigger a check that the type argument
// supplied is well-formed.
fn wf<T>() { }

fn func<'x, T:Trait1<'x>>(t: &'x T)
{
    wf::<&'x T::Foo>();
}


fn main() { }
