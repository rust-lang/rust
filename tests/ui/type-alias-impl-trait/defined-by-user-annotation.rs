// User type annotation in fn bodies is a a valid defining site for opaque types.
// check-pass
#![feature(type_alias_impl_trait)]

trait Equate { type Proj; }
impl<T> Equate for T { type Proj = T; }

trait Indirect { type Ty; }
impl<A, B: Equate<Proj = A>> Indirect for (A, B) { type Ty = (); }

type Opq = impl Sized;
fn define_1(_: Opq) {
    let _ = None::<<(Opq, u8) as Indirect>::Ty>;
}
fn define_2() -> Opq {
    0u8
}

fn main() {}
