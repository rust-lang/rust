use std::fmt::Debug;

trait Str {}

trait Something: Sized {
    fn yay<T: Debug>(_: Option<Self>, thing: &[T]);
}

struct X { data: u32 }

impl Something for X {
    fn yay<T: Str>(_:Option<X>, thing: &[T]) {
    //~^ ERROR E0276
    }
}

fn main() {
    let arr = &["one", "two", "three"];
    println!("{:?}", Something::yay(None::<X>, arr));
}
