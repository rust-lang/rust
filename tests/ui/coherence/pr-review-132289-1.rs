// This is a regression test for issues that came up during review of the (closed)
// PR #132289; this single-crate test case is
// the first example from @steffahn during review.
// https://github.com/rust-lang/rust/pull/132289#issuecomment-2564492153

//@ check-pass

type A = &'static [usize; 1];
type B = &'static [usize; 100];

type DynSomething = dyn Something<Assoc = A>;

trait Super {
    type Assoc;
}
impl Super for Foo {
    type Assoc = A;
}

trait IsDynSomething {}
impl IsDynSomething for DynSomething {}

impl<T: ?Sized> Super for T
where
    T: IsDynSomething,
{
    type Assoc = B;
}

trait Something: Super {
    fn method(&self) -> Self::Assoc;
}

struct Foo;
impl Something for Foo {
    fn method(&self) -> Self::Assoc {
        &[1337]
    }
}

fn main() {
    let x = &Foo;
    let y: &DynSomething = x;

    // no surprises here
    let _arr1: A = x.method();

    // this (`_arr2`) can't ever become B either, soundly
    let _arr2: A = y.method();
    // there aren't any other arrays being defined anywhere in this
    // test case, besides the length-1 one containing [1337]
}
