pub trait ToNbt<T> {
    fn new(val: T) -> Self;
}

impl dyn ToNbt<Self> {}
//~^ ERROR `Self` is not valid in the self type of an impl block

fn main() {}
