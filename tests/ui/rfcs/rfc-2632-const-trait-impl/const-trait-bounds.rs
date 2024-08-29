//@ known-bug: #110395
//@ compile-flags: -Znext-solver
// FIXME(effects): check-pass
#![feature(const_trait_impl, effects, generic_const_exprs)]
#![allow(incomplete_features)]

fn main() {
    let _ = process::<()>([()]);
    let _ = Struct::<(), 4> { field: [1, 0] };
}

fn process<T: const Trait>(input: [(); T::make(2)]) -> [(); T::make(2)] {
    input
}

struct Struct<T: const Trait, const P: usize>
where
    [u32; T::make(P)]:,
{
    field: [u32; T::make(P)],
}

#[const_trait]
trait Trait {
    fn make(input: usize) -> usize;
}

impl const Trait for () {
    fn make(input: usize) -> usize {
        input / 2
    }
}
