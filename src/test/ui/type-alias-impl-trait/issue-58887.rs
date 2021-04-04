// run-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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
