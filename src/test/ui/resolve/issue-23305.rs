pub trait ToNbt<T> {
    fn new(val: T) -> Self;
}

impl dyn ToNbt<Self> {}
//~^ ERROR cycle detected
//~| ERROR `Self` is only available in impls, traits, and type definitions

fn main() {}
