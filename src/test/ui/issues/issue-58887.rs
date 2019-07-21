#![feature(existential_type)]

trait UnwrapItemsExt {
    type II;
    fn unwrap_items(self) -> Self::II;
}

impl<I, T, E> UnwrapItemsExt for I
where
    I: Iterator<Item = Result<T, E>>,
    E: std::fmt::Debug,
{
    existential type II: Iterator<Item = T>;
    //~^ ERROR: could not find defining uses

    fn unwrap_items(self) -> Self::II {
    //~^ ERROR: type parameter `T` is part of concrete type
    //~| ERROR: type parameter `E` is part of concrete type
        self.map(|x| x.unwrap())
    }
}
