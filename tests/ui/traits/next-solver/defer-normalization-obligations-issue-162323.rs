//@ check-pass
//@ compile-flags: -Znext-solver=globally

struct Input;

struct MyRhs;

trait Jac {
    type V;
    fn jacobian(&self, x: Self::V);
}

impl Jac for MyRhs {
    type V = Input;

    fn jacobian(&self, _: Self::V) {}
}

struct MyEqns;

trait EqnsRef<'a, ImplicitBounds = &'a ()> {
    type Rhs: Jac<V = Input>;

    fn rhs(&'a self) -> Self::Rhs;
}

impl<'a> EqnsRef<'a> for MyEqns {
    type Rhs = MyRhs;

    fn rhs(&'a self) -> Self::Rhs {
        MyRhs
    }
}

fn inner() -> impl for<'a> EqnsRef<'a, Rhs: Jac<V = Input>> {
    MyEqns
}

fn main() {
    inner().rhs().jacobian(Input);
}
