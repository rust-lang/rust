trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

struct Struct {
    x: isize,
}

impl Struct {
    fn uhoh<T>(foo: <T as Get>::Value) {}
    //~^ ERROR trait `Get` is not implemented for `T`
    //~| ERROR trait `Get` is not implemented for `T`
}

fn main() {}
