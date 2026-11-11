//@ compile-flags: -Znext-solver
//@ check-pass

trait Mirror {
    type Assoc;
}

impl<T> Mirror for T {
    type Assoc = T;
}

trait Test {}
impl Test for i64 {}
impl Test for u64 {}

fn mirror_me<T: Mirror>(t: T, s: <T as Mirror>::Assoc) where <T as Mirror>::Assoc: Test {}

fn main() {
    let mut x = 0;
    mirror_me(x, 1);
    x = 1i64;
}
