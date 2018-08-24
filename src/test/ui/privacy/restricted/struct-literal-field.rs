#![allow(warnings)]

mod foo {
    pub mod bar {
        pub struct S {
            pub(in foo) x: i32,
        }
    }

    fn f() {
        use foo::bar::S;
        S { x: 0 }; // ok
    }
}

fn main() {
    use foo::bar::S;
    S { x: 0 }; //~ ERROR private
}
