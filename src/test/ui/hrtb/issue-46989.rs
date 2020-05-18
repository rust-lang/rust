// Regression test for #46989:
//
// In the move to universes, this test started passing.
// It is not necessarily WRONG to do so, but it was a bit
// surprising. The reason that it passed is that when we were
// asked to prove that
//
//     for<'a> fn(&'a i32): Foo
//
// we were able to use the impl below to prove
//
//     fn(&'empty i32): Foo
//
// and then we were able to prove that
//
//     fn(&'empty i32) = for<'a> fn(&'a i32)
//
// This last fact is somewhat surprising, but essentially "falls out"
// from handling variance correctly. In particular, consider the subtyping
// relations. First:
//
//     fn(&'empty i32) <: for<'a> fn(&'a i32)
//
// This holds because -- intuitively -- a fn that takes a reference but doesn't use
// it can be given a reference with any lifetime. Similarly, the opposite direction:
//
//     for<'a> fn(&'a i32) <: fn(&'empty i32)
//
// holds because 'a can be instantiated to 'empty.

trait Foo {

}

impl<A> Foo for fn(A) { }

fn assert_foo<T: Foo>() {}

fn main() {
    assert_foo::<fn(&i32)>();
    //~^ ERROR the trait bound `for<'r> fn(&'r i32): Foo` is not satisfied
}
