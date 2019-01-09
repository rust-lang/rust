// run-pass
#![allow(unused_must_use)]
#![feature(decl_macro)]

pub macro create_struct($a:ident) {
    struct $a;
    impl Clone for $a {
        fn clone(&self) -> Self {
            $a
        }
    }
}

fn main() {
    create_struct!(Test);
    Test.clone();
}
