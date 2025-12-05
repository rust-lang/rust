fn foo<'a, T>(_t: T) where T: Into<&'a str> {}

fn main() {
    foo(String::new());
    //~^ ERROR the trait bound `&str: From<String>` is not satisfied
}
