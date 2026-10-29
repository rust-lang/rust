// https://github.com/rust-lang/rust/issues/7663
//@ run-pass

#![allow(unused_imports, dead_code)]

mod test1 {

    mod foo { pub fn p() -> isize { 1 } }
    mod bar { pub fn p() -> isize { 2 } }

    pub mod baz {
        use crate::test1::bar::p;

        pub fn my_main() { assert_eq!(p(), 2); }
    }
}

mod test2 {

    mod foo { pub fn p() -> isize { 1 } }
    mod bar { pub fn p() -> isize { 2 } }

    pub mod baz {
        use crate::test2::bar::p;

        pub fn my_main() { assert_eq!(p(), 2); }
    }
}

fn main() {
    test1::baz::my_main();
    test2::baz::my_main();
}
