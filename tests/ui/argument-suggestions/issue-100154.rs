fn foo(i: impl std::fmt::Display) {}

fn main() {
    foo::<()>(());
    //~^ ERROR function takes 0 generic arguments but 1 generic argument was supplied
    //~| ERROR `()` doesn't implement `std::fmt::Display`
}
