//@ check-pass
// This test checks that the compiler propagates outlives requirements for both
// non-local lower bounds ['a, 'b] of '_, instead of conservatively finding a post-dominiting one
// from those 2.

struct MyTy<'a, 'b, 'x>(std::cell::Cell<(&'a &'x str, &'b &'x str)>);
fn wf<T>(_: T) {}
fn test<'a, 'b, 'x>() {
    |x: MyTy<'a, 'b, '_>| wf(x);
}

fn main() {}
