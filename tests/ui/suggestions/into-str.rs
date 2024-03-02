fn foo<'a, T>(_t: T) where T: Into<&'a str> {}

fn main() {
    foo(String::new());
    //~^ ERROR trait `From<String>` is not implemented for `&str`
}
