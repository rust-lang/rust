//! This test checks that when checking for opaque types that
//! only differ in lifetimes, we handle the case of non-generic
//! regions correctly.
#![feature(type_alias_impl_trait)]

type Opq<'a> = impl Sized;

// Two defining uses: Opq<'{empty}> and Opq<'a>.
// This used to ICE.
// issue: #122782
#[define_opaque(Opq)]
fn build<'a>() -> Opq<'a> {
    let _: Opq<'_> = ();
    //~^ ERROR expected generic lifetime parameter, found `'_`
}

fn main() {}
