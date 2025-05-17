#![allow(unused_imports, dead_code)]

mod test1 {
    mod foo {
        pub struct P;
    }

    mod bar {
        pub struct P;
    }

    pub mod baz {
        use test1::foo::*;
        use test1::bar::*;

        pub fn f() {
            let _ = P; //~ ERROR `P` is ambiguous
        }
    }
}

mod test2 {
    mod foo {
        pub struct P;
    }

    mod bar {
        pub struct P;
    }

    pub mod baz {
        use test2::foo::P;
        use test2::bar::P; //~ ERROR the name `P` is defined multiple times
    }
}

fn main() {
}
