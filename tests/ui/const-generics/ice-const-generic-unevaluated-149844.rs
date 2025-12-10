//@ build-pass
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

#[derive(Clone, Copy, Debug)]
pub struct A<T: const From<u8>, const N: usize>
where
    [(); N / 8]:,
{
    b: [u8; N / 8],
}

impl<T: const From<u8>, const N: usize> A<T, N>
where
    [(); N / 8]:,
{
    pub const fn f(&self) -> T
    where
        [(); N / 8]:,
    {
        self.b[0].into()
    }
}

fn main() {}
