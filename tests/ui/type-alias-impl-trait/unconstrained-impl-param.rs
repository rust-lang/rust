#![feature(type_alias_impl_trait)]

use std::fmt::Display;

type Opaque<X> = impl Sized + 'static;
#[define_opaque(Opaque)]
fn define<X>() -> Opaque<X> {}

trait Trait {
    type Assoc: Display;
}
impl<'a> Trait for Opaque<&'a str> {
    //~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait, self type, or predicates
    type Assoc = &'a str;
}

// ======= Exploit =======

fn extend<T: Trait + 'static>(s: T::Assoc) -> Box<dyn Display> {
    Box::new(s)
}

fn main() {
    let val = extend::<Opaque<&'_ str>>(&String::from("blah blah blah"));
    println!("{}", val);
}
