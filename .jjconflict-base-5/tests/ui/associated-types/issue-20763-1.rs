//@ check-pass
#![allow(dead_code)]

trait T0 {
    type O;
    fn dummy(&self) { }
}

struct S<A>(A);
impl<A> T0 for S<A> { type O = A; }

trait T1: T0 {
    // this looks okay but as we see below, `f` is unusable
    fn m0<F: Fn(<Self as T0>::O) -> bool>(self, f: F) -> bool;
}

// complains about the bounds on F here not being required by the trait
impl<A> T1 for S<A> {
    fn m0<F: Fn(A) -> bool>(self, f: F) -> bool { f(self.0) }
}

// // complains about mismatched types: <S<A> as T0>::O vs. A
// impl<A> T1 for S<A>
// {
//     fn m0<F: Fn(<Self as T0>::O) -> bool>(self, f: F) -> bool { f(self.0) }
// }

fn main() { }
