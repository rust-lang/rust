pub struct Struct<const N: usize>(());

impl<const N: usize> Struct<N> {
    pub fn new() -> Self {
        Struct(())
    }

    pub fn same_ty<const M: usize>(&self) -> (usize, usize) {
        (N, M)
    }

    pub fn different_ty<const M: u8>(&self) -> (usize, u8) {
        (N, M)
    }

    pub fn containing_ty<T, const M: u8>(&self) -> (usize, u8) {
        (std::mem::size_of::<T>() +  N, M)
    }

    pub fn we_have_to_go_deeper<const M: usize>(&self) -> Struct<M> {
        Struct(())
    }
}

pub trait Foo {
    fn foo<const M: usize>(&self) -> usize;
}

impl Foo for Struct<7> {
    fn foo<const M: usize>(&self) -> usize {
        M
    }
}
