//! Regression test for <https://github.com/rust-lang/rust/issues/138225>

pub struct A {
    name: NestedOption<Option<String>>,
    //~^ ERROR cannot find type `NestedOption` in this scope
}

impl A {
    pub async fn func1() -> &'static A {
        //~^ ERROR `async fn` is not permitted in Rust 2015
        static RES: A = A { name: None };
        &RES
    }
}

fn main() {}
