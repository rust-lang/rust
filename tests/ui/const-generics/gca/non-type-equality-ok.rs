//@ check-pass
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![feature(generic_const_items)]
#![expect(incomplete_features)]

trait Trait {
    const PROJECTED_A: usize;
    const PROJECTED_B: usize;
}

struct StructImpl;
struct GenericStructImpl<const N: usize>;

impl Trait for StructImpl {
    const PROJECTED_A: usize = 1;
    const PROJECTED_B: usize = 1;
}

impl<const N: usize> Trait for GenericStructImpl<N> {
    const PROJECTED_A: usize = N;
    const PROJECTED_B: usize = N;
}

const FREE_A: usize = 1;
const FREE_B: usize = 1;

struct Struct<const N: usize>;

fn f<const N: usize>() {
    let _: Struct<{ <GenericStructImpl<N> as Trait>::PROJECTED_A }> =
        Struct::<{ <GenericStructImpl<N> as Trait>::PROJECTED_A }>;
}

fn g<T: Trait>() {
    let _: Struct<{ T::PROJECTED_A }> = Struct::<{ T::PROJECTED_A }>;
}

fn main() {
    let _: Struct<FREE_A> = Struct::<FREE_B>;
    let _: Struct<{ <StructImpl as Trait>::PROJECTED_A }> =
        Struct::<{ <StructImpl as Trait>::PROJECTED_B }>;
    let _: Struct<{ <GenericStructImpl<2> as Trait>::PROJECTED_A }> =
        Struct::<{ <GenericStructImpl<2> as Trait>::PROJECTED_B }>;
}
