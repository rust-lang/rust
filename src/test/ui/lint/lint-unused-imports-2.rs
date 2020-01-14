#![deny(unused_imports)]
#![allow(dead_code)]

// Be sure that if we just bring some methods into scope that they're also
// counted as being used.
use test::B;
// But only when actually used: do not get confused by the method with the same name.
use test::B2; //~ ERROR unused import: `test::B2`

mod test {
    pub trait B {
        fn b(&self) {}
    }
    pub trait B2 {
        fn b(&self) {}
    }
    pub struct C;
    impl B for C {}
}

fn main() {
    test::C.b();
}
