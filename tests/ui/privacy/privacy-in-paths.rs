mod foo {
    pub use self::bar::S;
    mod bar {
        pub struct S;
        pub use crate::baz;
    }

    trait T {
        type Assoc;
    }
    impl T for () {
        type Assoc = S;
    }
}

impl foo::S {
    fn f() {}
}

pub mod baz {
    fn f() {}

    fn g() {
        crate::foo::bar::baz::f(); //~ERROR module `bar` is private
        crate::foo::bar::S::f(); //~ERROR module `bar` is private
        <() as crate::foo::T>::Assoc::f(); //~ERROR trait `T` is private
    }
}

fn main() {}
