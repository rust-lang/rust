// run-pass
#![allow(dead_code)]
// Check that we do not report ambiguities when the same predicate
// appears in the environment twice. Issue #21965.

// pretty-expanded FIXME #23616

trait Foo {
    type B;

    fn get() -> Self::B;
}

fn foo<T>() -> ()
    where T : Foo<B=()>, T : Foo<B=()>
{
    <T as Foo>::get()
}

fn main() {
}
