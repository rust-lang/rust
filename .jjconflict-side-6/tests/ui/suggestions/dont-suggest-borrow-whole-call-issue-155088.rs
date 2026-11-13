use std::borrow::Borrow;

fn foo(_v: impl IntoIterator<Item = impl Borrow<str>>) {}

fn main() {
    foo(&[String::from("a")]);
    //~^ ERROR the trait bound `&String: Borrow<str>` is not satisfied
}
