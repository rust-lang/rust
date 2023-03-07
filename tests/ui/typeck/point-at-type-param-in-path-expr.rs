fn foo<T: std::fmt::Display>() {}

fn main() {
    let x = foo::<()>;
    //~^ ERROR `()` doesn't implement `std::fmt::Display`
}
