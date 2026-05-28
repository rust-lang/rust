//@ build-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Target {
    const LENGTH: usize;
}


pub struct Container<T: Target>
where
    [(); T::LENGTH]: Sized,
{
    _target: T,
}

impl<T: Target> Container<T>
where
    [(); T::LENGTH]: Sized,
{
    pub fn start(
        _target: T,
    ) -> Container<T> {
        Container { _target }
    }
}

fn main() {}
