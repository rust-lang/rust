#![feature(reborrow)]

struct Wrap<'a>(&'a mut (), String);
//~^ ERROR the trait bound `String: Copy` is not satisfied

impl<'a> std::marker::Reborrow for Wrap<'a> {}

fn get(x: Wrap<'_>) -> String {
    x.1
}

fn main() {
    let x = Wrap(&mut (), String::from("a"));
    let _y = get(x);
    let _z = get(x);
}
