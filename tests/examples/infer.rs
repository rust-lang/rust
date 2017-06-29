pub mod old {
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
}

pub mod new {
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
        pub field: &'a a::Def,
        pub field2: ::std::rc::Rc<a::Def>,
    }
}
