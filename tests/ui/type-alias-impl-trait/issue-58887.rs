//@ check-pass

#![feature(impl_trait_in_assoc_type)]

trait UnwrapItemsExt {
    type Iter;
    fn unwrap_items(self) -> Self::Iter;
}

impl<I, T, E> UnwrapItemsExt for I
where
    I: Iterator<Item = Result<T, E>>,
    E: std::fmt::Debug,
{
    type Iter = impl Iterator<Item = T>;

    fn unwrap_items(self) -> Self::Iter {
        self.map(|x| x.unwrap())
    }
}

fn main() {}
