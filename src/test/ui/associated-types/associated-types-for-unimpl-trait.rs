trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

trait Other {
    fn uhoh<U:Get>(&self, foo: U, bar: <Self as Get>::Value) {}
    //~^ ERROR the trait bound `Self: Get` is not satisfied
}

fn main() {
}
