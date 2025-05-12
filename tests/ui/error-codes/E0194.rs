trait Foo<T> {
    fn do_something(&self) -> T;
    fn do_something_else<T: Clone>(&self, bar: T);
    //~^ ERROR E0403
}

fn main() {
}
