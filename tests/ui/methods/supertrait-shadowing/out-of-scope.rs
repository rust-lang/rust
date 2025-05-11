//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

mod out_of_scope {
    pub trait Subtrait: super::Supertrait {
        fn hello(&self) {
            println!("subtrait");
        }
    }
    impl<T> Subtrait for T {}
}

trait Supertrait {
    fn hello(&self) {
        println!("supertrait");
    }
}
impl<T> Supertrait for T {}

fn main() {
    ().hello();
}
