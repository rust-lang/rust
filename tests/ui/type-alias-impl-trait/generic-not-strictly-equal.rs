// `Opaque<'?1> := u8` is not a valid defining use here.
//
// Due to fundamental limitations of the member constraints algorithm,
// we require '?1 to be *equal* to some universal region.
//
// While '?1 is eventually inferred to be equal to 'x because of the constraint '?1: 'x,
// we don't consider them equal in the strict sense because they lack the bidirectional outlives
// constraints ['?1: 'x, 'x: '?1]. In NLL terms, they are not part of the same SCC.
//
// See #113971 for details.

//@ revisions: basic member_constraints
#![feature(type_alias_impl_trait)]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn ensure_outlives<'a, X: 'a>(_: X) {}
fn relate<X>(_: X, _: X) {}

type Opaque<'a> = impl Copy + Captures<'a>;

#[define_opaque(Opaque)]
fn test<'x>(_: Opaque<'x>) {
    let opaque = None::<Opaque<'_>>; // let's call this lifetime '?1

    #[cfg(basic)]
    let hidden = None::<u8>;

    #[cfg(member_constraints)]
    let hidden = None::<&'x u8>;

    ensure_outlives::<'x>(opaque); // outlives constraint: '?1: 'x
    relate(opaque, hidden); // defining use: Opaque<'?1> := u8
    //~^ ERROR expected generic lifetime parameter, found `'_`
}

fn main() {}
