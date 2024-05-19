// Check that we don't reject non-escaping late-bound vars in the type of assoc const bindings.
// There's no reason why we should disallow them.
//
//@ check-pass

#![feature(associated_const_equality)]

trait Trait<T> {
    const K: T;
}

fn take(
    _: impl Trait<
        <for<'a> fn(&'a str) -> &'a str as Discard>::Out,
        K = { () }
    >,
) {}

trait Discard { type Out; }
impl<T: ?Sized> Discard for T { type Out = (); }

fn main() {}
