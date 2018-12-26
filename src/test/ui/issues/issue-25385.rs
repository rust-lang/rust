macro_rules! foo {
    ($e:expr) => { $e.foo() }
    //~^ ERROR no method named `foo` found for type `i32` in the current scope
}

fn main() {
    let a = 1i32;
    foo!(a);

    foo!(1i32.foo());
    //~^ ERROR no method named `foo` found for type `i32` in the current scope
}
