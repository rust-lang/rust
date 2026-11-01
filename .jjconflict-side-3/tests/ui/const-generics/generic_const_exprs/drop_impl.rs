//@check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Foo<const N: usize>
where
    [(); N + 1]: ;

impl<const N: usize> Drop for Foo<N>
where
    [(); N + 1]: ,
{
    fn drop(&mut self) {}
}

fn main() {}
