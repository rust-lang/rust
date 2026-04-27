//@ check-pass
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    const PROJECTED: usize;
}

struct StructImpl;
struct GenericStructImpl<const A: usize>;

const FREE: usize = 1;

impl Trait for StructImpl {
    const PROJECTED: usize = 1;
}

impl<const A: usize> Trait for GenericStructImpl<A> {
    const PROJECTED: usize = A;
}

struct Struct<const N: usize>;

fn f<T: Trait>() {
    let _ = Struct::<{ T::PROJECTED }>;
}

fn main() {
    let _ = Struct::<FREE>;
    let _ = Struct::<{ <StructImpl as Trait>::PROJECTED }>;
    let _ = Struct::<{ <GenericStructImpl<2> as Trait>::PROJECTED }>;
}
