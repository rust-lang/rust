// A regression test for https://github.com/rust-lang/rust/issues/152663
// Previously triggered an ICE when checking whether the param-env
// shadows a global impl. The crash occurred due to calling
// `TyCtxt::type_of` on an erroneous associated type in a trait impl
// that had no corresponding value.

trait Iterable {
    type Iter;
}

impl<T> Iterable for [T] {
    //~^ ERROR: not all trait items implemented
    fn iter() -> Self::Iter {}
    //~^ ERROR: method `iter` is not a member of trait `Iterable`
    //~| ERROR: mismatched types
}

fn main() {}
