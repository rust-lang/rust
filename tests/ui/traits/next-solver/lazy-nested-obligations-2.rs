//@ compile-flags: -Znext-solver

// FIXME: This fails on the old solver. The next solver used to accept it thanks to
// lazy normalization, but that behavior regressed when output expectation fudge
// handling was updated in #149320 and #150316.

pub trait With {
    type F;
}

impl With for i32 {
    type F = fn(&str);
}

fn f(_: &str) {}

fn main() {
    let _: V<i32> = V(f);
    //~^ ERROR: type mismatch
    pub struct V<T: With>(<T as With>::F);

    pub enum E3<T: With> {
        Var(<T as With>::F),
    }
    let _: E3<i32> = E3::Var(f);
    //~^ ERROR: type mismatch
}
