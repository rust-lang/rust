//! Regression test for issue <https://github.com/rust-lang/rust/issues/53300>
//! Tests that an undefined type (Wrapper) used with impl Trait correctly gives E0412.

pub trait A {
    fn add(&self, b: i32) -> i32;
}

fn addition() -> Wrapper<impl A> {}
//~^ ERROR cannot find type `Wrapper` in this scope [E0425]

fn main() {
    let res = addition();
}
