// Check that we are not producing long error messages

trait Trait {
    type Output;
    fn process<T>(&mut self, input: T) -> T;
}

struct MyImpl;

impl Trait for MyImpl {
    type Output = &'static str;
    fn process<T>(&mut self, input: T) -> T { input }
}

fn make() -> impl Trait<Output = <dyn Trait<Output = &'static str> as Trait>::Output> {
    //~^ ERROR the trait `Trait` is not dyn compatible
    MyImpl
}

fn main() {}
