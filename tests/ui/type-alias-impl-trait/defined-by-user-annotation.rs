// User type annotation in fn bodies is a valid defining site for opaque types.
//@ check-pass
#![feature(type_alias_impl_trait)]

trait Equate { type Proj; }
impl<T> Equate for T { type Proj = T; }

trait Indirect { type Ty; }
impl<A, B: Equate<Proj = A>> Indirect for (A, B) { type Ty = (); }

mod basic {
    use super::*;
    type Opq = impl Sized;
    #[define_opaque(Opq)]
    fn define_1(_: Opq) {
        let _ = None::<<(Opq, u8) as Indirect>::Ty>;
    }
    #[define_opaque(Opq)]
    fn define_2() -> Opq {
        0u8
    }
}

// `Opq<'a> == &'b u8` shouldn't be an error because `'a == 'b`.
mod lifetime {
    use super::*;
    type Opq<'a> = impl Sized + 'a;
    #[define_opaque(Opq)]
    fn define<'a: 'b, 'b: 'a>(_: Opq<'a>) {
        let _ = None::<<(Opq<'a>, &'b u8) as Indirect>::Ty>;
    }
}

fn main() {}
