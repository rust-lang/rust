//@ check-pass
//@ compile-flags: -Zvalidate-mir

// This test checks that bivariant parameters are handled correctly
// in the mir.
#![allow(coherence_leak_check)]
trait Trait {
    type Assoc;
}

struct Foo<T, U>(T)
where
    T: Trait<Assoc = U>;

impl Trait for for<'a> fn(&'a ()) {
    type Assoc = u32;
}
impl Trait for fn(&'static ()) {
    type Assoc = String;
}

fn foo(x: Foo<for<'a> fn(&'a ()), u32>) -> Foo<fn(&'static ()), String> {
    x
}

fn main() {}
