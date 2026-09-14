//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This was previously accepted by the next-solver, thanks to its
// lazy normalization. But it started failing after #150316 due to changes in
// output expectation fudging in #149320 and #150316.

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
