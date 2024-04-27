//@ check-pass

pub trait Fooey: Sized {
    type Context<'c> where Self: 'c;
}

pub struct Handle<E: Fooey>(Option<Box<dyn for<'c> Fn(&mut E::Context<'c>)>>);

fn tuple<T>() -> (Option<T>,) { (Option::None,) }

pub struct FooImpl {}
impl Fooey for FooImpl {
    type Context<'c> = &'c ();
}

impl FooImpl {
    pub fn fail1() -> Handle<Self> {
        let (tx,) = tuple();
        Handle(tx)
    }
}

fn main() {}
