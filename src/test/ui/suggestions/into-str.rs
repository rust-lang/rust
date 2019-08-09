fn foo<'a, T>(_t: T) where T: Into<&'a str> {}

fn main() {
    foo(String::new());
    //~^ ERROR the trait bound `&str: std::convert::From<std::string::String>` is not satisfied
}
