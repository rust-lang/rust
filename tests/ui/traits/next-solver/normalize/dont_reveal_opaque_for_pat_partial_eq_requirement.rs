//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Previously we check the `PartialEq` requirement for pattern types in
// post analysis typing mode which would reveal opaques types so this code
// compiles with the old solver.
// With the next solver, we previously forgot to set opaques to non-rigid when
// entering post-analysis mode which results in an ICE when computing layout.
//
// Now we check the `PartialEq` requirement in the correct typing env and this
// code properly fails with both solvers.

#![feature(type_alias_impl_trait)]
#![feature(generic_const_items)]
#![allow(incomplete_features)]

const A<T>: Option<T> = None;

type Opaque = impl Sized;

#[define_opaque(Opaque)]
fn value() -> Opaque {
    0usize
}

fn defines() {
    let a = value();
    match Some(a) {
        A::<_> => {}
        //~^ ERROR constant of non-structural type `Option<Opaque>` in a pattern
        _ => {}
    }
}
fn main() {}
