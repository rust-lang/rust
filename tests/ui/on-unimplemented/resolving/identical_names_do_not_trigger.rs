#![feature(rustc_attrs)]

mod module {
    pub struct Foo;

    #[diagnostic::rustc_on_unimplemented(
        on(Self = "Foo", message = "the specialized message"),
        message = "the regular message"
    )]
    pub trait FromIterator<A>: Sized {
        fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
    }
}

struct Foo;

fn main() {
    let iter = 0..42_8;
    let x: Foo = module::FromIterator::from_iter(iter);
    //~^ ERROR the regular message
}
