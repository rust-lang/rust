//@ edition: 2021
//@ check-pass
// https://github.com/rust-lang/rust/issues/105235#issue-1474295873

mod abc {
    pub struct Beeblebrox;
    pub struct Zaphod;
}

mod foo {
    pub mod bar {
        use crate::abc::*;

        #[derive(Debug)]
        pub enum Zaphod {
            Whale,
            President,
        }
    }
    pub use bar::*;
}

mod baz {
    pub fn do_something() {
        println!("{:?}", crate::foo::Zaphod::Whale);
    }
}

fn main() {
    baz::do_something();
}
