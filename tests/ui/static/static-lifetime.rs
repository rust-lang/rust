pub trait Arbitrary: Sized + 'static {}

impl<'a, A: Clone> Arbitrary for ::std::borrow::Cow<'a, A> {} //~ ERROR lifetime bound
//~^ ERROR cannot infer an appropriate lifetime for lifetime parameter `'a`

fn main() {
}
