pub trait ToNbt<T> {
    fn new(val: T) -> Self;
}

impl dyn ToNbt<Self> {}
//~^ ERROR `Self` is not valid at this location

fn main() {}
