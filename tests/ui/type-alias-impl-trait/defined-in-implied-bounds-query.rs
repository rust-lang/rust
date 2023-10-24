// Test that we correctly handle implied bounds for tait1::test
// even when the computation requires defining `Tait2 := Tait1`.
//
// Currently wfcheck does not allow item signature to define opaques,
// but if this is ever allowed then this test should pass.
// We are currenly using DefiningAnchor::Bubble for implied_bounds query
// but if we keep doing so by then, this test will ICE because of the ambiguity
// when relating two opaques from different scopes `Tait1 == Tait2`.
//
// Much of the complexity is due to the uncertainty around the definig scope
// rules for TAIT.

#![feature(type_alias_impl_trait)]

trait Id { type Assoc; }
impl<X> Id for X { type Assoc = X; }

trait Equate { type Ty; }
// equate X==Y and pass Z unchanged
impl<X, Y, Z> Equate for (X, Y, Z)
where
    X: Id<Assoc = Y>,
{
    type Ty = Z;
}

mod tait1 {
    use super::*;

    pub type Tait1 = impl Clone;
    pub fn def_tait1() -> Tait1 {}

    pub struct EqualToTait1Rt<X, Z>(X, Z)
    where
        (X, Tait1, Z): Equate,
        <(X, Tait1, Z) as Equate>::Ty: 'static;
    pub struct EqualToTait1Lt<X, Z>(X, Z)
    where
        (Tait1, X, Z): Equate,
        <(Tait1, X, Z) as Equate>::Ty: 'static;
}

mod tait2 {
    use super::tait1::*;

    type Tait2 = impl Sized;

    fn test<Z1, Z2>(
        _: EqualToTait1Lt<Tait2, Z1>,
        //~^ ERROR type mismatch resolving `<Tait1 as Id>::Assoc == Tait2`
        _: EqualToTait1Rt<Tait2, Z2>,
        //~^ ERROR type mismatch resolving `<Tait2 as Id>::Assoc == Tait1`
    ) -> Tait2 {
        statik::<Z1>();
        statik::<Z2>();
        def_tait1()
    }

    fn statik<T: 'static>() {}
}

fn main() {}
