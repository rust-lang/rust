// known-bug: #108425
// check-pass
#![feature(type_alias_impl_trait)]
use std::fmt::Display;

type Opaque<'a> = impl Sized + 'static;
fn define<'a>() -> Opaque<'a> {}

trait Trait {
    type Assoc: Display;
}
impl<'a> Trait for Opaque<'a> {
    type Assoc = &'a str;
}

// ======= Exploit =======

fn extend<T: Trait + 'static>(s: T::Assoc) -> Box<dyn Display> {
    Box::new(s)
}

fn main() {
    let val = extend::<Opaque<'_>>(&String::from("blah blah blah"));
    println!("{}", val);
}
