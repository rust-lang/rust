mod a {
    pub trait Def {

    }

    pub struct Opq;
}

pub fn a<A: a::Def>(a: A) -> A {
    a
}

pub fn b() -> a::Opq {
    a::Opq
}

pub struct Hij<'a> {
    pub field: &'a dyn a::Def,
    pub field2: ::std::rc::Rc<dyn a::Def>,
}
