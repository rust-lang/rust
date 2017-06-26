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

    pub struct Hij {
        pub field: Box<a::Abc>,
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

    pub struct Hij {
        pub field: Box<a::Def>,
    }
}
