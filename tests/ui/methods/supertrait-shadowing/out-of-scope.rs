//@ run-pass

#![allow(dead_code)]

mod out_of_scope {
    pub trait Subtrait: super::Supertrait {
        fn hello(&self) -> &'static str {
            "subtrait"
        }
    }
    impl<T> Subtrait for T {}
}

trait Supertrait {
    fn hello(&self) -> &'static str {
        "supertrait"
    }
}
impl<T> Supertrait for T {}

fn main() {
    assert_eq!(().hello(), "supertrait");
}
