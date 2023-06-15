// check-pass

#![warn(unused_imports)] // Warning explanation here, it's OK

mod test {
    pub trait A {
        fn a();
    }

    impl A for () {
        fn a() { }
    }

    pub trait B {
        fn b(self);
    }

    impl B for () {
        fn b(self) { }
    }

    pub trait Unused {
    }
}

use test::Unused;   // This is really unused, so warning is OK
                    //~^ WARNING unused import
use test::A;        // This is used by the test2::func() through import of super::*
use test::B;        // This is used by the test2::func() through import of super::*

mod test2 {
    use super::*;
    pub fn func() {
        <()>::a();
        ().b();
        test3::inner_func();
    }
    mod test3 {
        use super::*;
        pub fn inner_func() {
            <()>::a();
            ().b();
        }
    }
}

fn main() {
    test2::func();
}
