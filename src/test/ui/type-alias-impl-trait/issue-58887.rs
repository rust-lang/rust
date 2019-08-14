#![feature(type_alias_impl_trait)]

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
    //~^ ERROR: could not find defining uses

    fn unwrap_items(self) -> Self::Iter {
    //~^ ERROR: type parameter `T` is part of concrete type
    //~| ERROR: type parameter `E` is part of concrete type
        self.map(|x| x.unwrap())
    }
}

fn main() {}
