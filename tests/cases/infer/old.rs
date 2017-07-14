mod a {
    pub trait Abc {

    }

    pub struct Klm;
}

pub fn a<A: a::Abc>(a: A) -> A {
    a
}

pub fn b() -> a::Klm {
    a::Klm
}

pub struct Hij<'a> {
    pub field: &'a a::Abc,
    pub field2: ::std::rc::Rc<a::Abc>,
}
