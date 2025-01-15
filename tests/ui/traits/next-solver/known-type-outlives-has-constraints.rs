//@ compile-flags: -Znext-solver
//@ check-pass

trait Norm {
    type Out;
}
impl<'a, T: 'a> Norm for &'a T {
    type Out = T;
}

fn hello<'a, T: 'a>() where <&'a T as Norm>::Out: 'a {}

fn main() {}
