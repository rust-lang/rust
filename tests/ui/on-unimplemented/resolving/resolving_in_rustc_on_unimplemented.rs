#![feature(rustc_attrs)]

pub struct Foo;

#[diagnostic::rustc_on_unimplemented(
    on(Self = "Foo", message = "the specialized message"),
    message = "the message"
)]
pub trait FromIterator<A>: Sized {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
}
fn main() {
    let iter = 0..42_8;
    let x: Foo = FromIterator::from_iter(iter);
    //~^ ERROR the specialized message
}
