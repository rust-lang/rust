pub trait ToNbt<T> {
    fn new(val: T) -> Self;
}

impl dyn ToNbt<Self> {}
//~^ ERROR cycle detected

fn main() {}
