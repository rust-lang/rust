#![crate_name = "foo"]
#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait Notable {}

pub struct Wrapper<T: ?Sized>(Box<T>);

pub struct Foo;
pub type Alias = Foo;

impl Notable for Wrapper<dyn std::error::Error> {}
impl Notable for Wrapper<u8> {}

//@ has 'foo/fn.returns_alias.html'
pub fn returns_alias() -> Wrapper<Alias> {
    Wrapper(Box::new(Foo))
}
