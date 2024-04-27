trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

struct Struct {
    x: isize,
}

impl Struct {
    fn uhoh<T>(foo: <T as Get>::Value) {}
    //~^ ERROR the trait bound `T: Get` is not satisfied
    //~| ERROR the trait bound `T: Get` is not satisfied
}

fn main() {}
