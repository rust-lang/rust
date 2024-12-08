use std::fmt::Debug;

trait Any {}
impl<T> Any for T {}

// Check that type parameters are captured and not considered 'static
fn foo<T>(x: T) -> impl Any + 'static {
    x
    //~^ ERROR the parameter type `T` may not live long enough
}

fn main() {}
