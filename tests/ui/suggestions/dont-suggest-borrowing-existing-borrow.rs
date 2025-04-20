fn main() {
    let _ = &str::from("value");
    //~^ ERROR the trait bound `str: From<_>` is not satisfied
    //~| ERROR the size for values of type `str` cannot be known at compilation time
}
