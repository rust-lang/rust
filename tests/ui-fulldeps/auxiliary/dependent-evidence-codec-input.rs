pub trait Family {
    type Item;
}

impl Family for bool {
    type Item = u8;
}

impl<T: Family> Family for (T, T) {
    type Item = (T::Item, T::Item);
}

pub trait Other {
    type Item;
}

fn main() {}
