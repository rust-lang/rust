//@ compile-flags: -Znext-solver
//@ check-pass

pub trait With {
    type F;
}

impl With for i32 {
    type F = fn(&str);
}

fn f(_: &str) {}

fn main() {
    let _: V<i32> = V(f);
    pub struct V<T: With>(<T as With>::F);

    pub enum E3<T: With> {
        Var(<T as With>::F),
    }
    let _: E3<i32> = E3::Var(f);
}
