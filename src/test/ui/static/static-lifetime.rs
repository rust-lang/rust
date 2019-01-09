pub trait Arbitrary: Sized + 'static {}

impl<'a, A: Clone> Arbitrary for ::std::borrow::Cow<'a, A> {} //~ ERROR lifetime bound

fn main() {
}
