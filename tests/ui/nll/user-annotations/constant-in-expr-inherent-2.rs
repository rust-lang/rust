// Test that we still check constants are well-formed, even when we there's no
// type annotation to check.

const FUN: fn(&'static ()) = |_| {};
struct A;
impl A {
    const ASSOCIATED_FUN: fn(&'static ()) = |_| {};
}

struct B<'a>(&'a ());
impl B<'static> {
    const ALSO_ASSOCIATED_FUN: fn(&'static ()) = |_| {};
}

trait Z: 'static {
    const TRAIT_ASSOCIATED_FUN: fn(&'static Self) = |_| ();
}

impl Z for () {}

fn main() {
    let x = ();
    FUN(&x);                        //~ ERROR `x` does not live long enough
    A::ASSOCIATED_FUN(&x);          //~ ERROR `x` does not live long enough
    B::ALSO_ASSOCIATED_FUN(&x);     //~ ERROR `x` does not live long enough
    <_>::TRAIT_ASSOCIATED_FUN(&x);  //~ ERROR `x` does not live long enough
}
