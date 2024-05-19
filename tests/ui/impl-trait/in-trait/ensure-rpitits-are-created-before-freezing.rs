trait Iterable {
    type Item;
    fn iter(&self) -> impl Sized;
}

// `ty::Error` in a trait ref will silence any missing item errors, but will also
// prevent the `associated_items` query from being called before def ids are frozen.
impl Iterable for Missing {
//~^ ERROR cannot find type `Missing` in this scope
    fn iter(&self) -> Self::Item {}
}

fn main() {}
