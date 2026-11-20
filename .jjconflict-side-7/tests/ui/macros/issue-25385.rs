macro_rules! foo {
    ($e:expr) => { $e.foo() }
    //~^ ERROR no method named `foo` found
}

fn main() {
    let a = 1i32;
    foo!(a);

    foo!(1i32.foo());
    //~^ ERROR no method named `foo` found
}
