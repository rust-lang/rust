trait Super {
    type Assoc;
}
impl Super for () {
    type Assoc = u8;
}
trait Sub: Super<Assoc = u16> {}

trait BoundOnSelf: Sub {}
impl BoundOnSelf for () {}
//~^ ERROR the trait bound `(): Sub` is not satisfied
//~| ERROR type mismatch resolving `<() as Super>::Assoc == u16`

trait BoundOnParam<T: Sub> {}
impl BoundOnParam<()> for () {}
//~^ ERROR the trait bound `(): Sub` is not satisfied
//~| ERROR type mismatch resolving `<() as Super>::Assoc == u16`

trait BoundOnAssoc {
    type Assoc: Sub;
}
impl BoundOnAssoc for () {
    type Assoc = ();
    //~^ ERROR the trait bound `(): Sub` is not satisfied
    //~| ERROR type mismatch resolving `<() as Super>::Assoc == u16`
}

trait BoundOnGat where Self::Assoc<u8>: Sub {
    type Assoc<T>;
}
impl BoundOnGat for u8 {
    //~^ ERROR type mismatch resolving `<() as Super>::Assoc == u16`
    type Assoc<T> = ();
    //~^ ERROR the trait bound `(): Sub` is not satisfied
}

fn trivial_bound() where (): Sub {}
//~^ ERROR the trait bound `(): Sub` is not satisfied
//~| ERROR type mismatch resolving `<() as Super>::Assoc == u16`

fn main() {}
