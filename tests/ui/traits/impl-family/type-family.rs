// A family of impls that differ only in the self type's type argument. Integer and float
// inference variables, and impls written with an alias, must still find their impls.
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Tr {}
struct W<T>(T);
mod m {
    pub struct A;
    pub struct B;
}

macro_rules! family {
    ($tr:ident: $($t:ty),*) => { $(impl $tr for W<$t> {})* };
}
family!(Tr: u8, i32, f64, bool, m::A, m::B);

trait Id {
    type Out;
}
impl<T> Id for T {
    type Out = T;
}

// One impl written through an alias keeps this family from being split at all.
trait Tr2 {}
family!(Tr2: u8, i32, f64, bool, m::A);
impl Tr2 for W<<m::B as Id>::Out> {}

fn takes<T: Tr>(_: T) {}
fn requires2<T: Tr2>() {}

fn main() {
    takes(W(1));
    takes(W(1.0));
    takes(W(m::A));
    requires2::<W<m::B>>();
}
