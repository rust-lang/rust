pub trait ToNbt<T> {
    fn new(val: T) -> Self;
}

impl ToNbt<Self> {}
//~^ ERROR cycle detected

fn main() {}
