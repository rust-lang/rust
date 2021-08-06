//check-pass
#![feature(const_generics, const_evaluatable_checked)]
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
