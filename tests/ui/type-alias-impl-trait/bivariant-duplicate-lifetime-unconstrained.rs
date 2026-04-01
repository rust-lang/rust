// The defining use below has an unconstrained lifetime argument.
// Opaque<'{empty}, 'a> := ();
// Make sure we accept it because the lifetime parameter in such position is
// irrelevant - it is an artifact of how we internally represent opaque
// generics.
// See issue #122307 for details.

//@ check-pass
#![feature(type_alias_impl_trait)]
#![allow(unconditional_recursion)]

type Opaque<'a> = impl Sized + 'a;

#[define_opaque(Opaque)]
fn test<'a>() -> Opaque<'a> {
    let _: () = test::<'a>();
}

fn main() {}
