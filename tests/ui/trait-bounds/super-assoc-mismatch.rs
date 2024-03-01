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

trait BoundOnParam<T: Sub> {}
impl BoundOnParam<()> for () {}
//~^ ERROR the trait bound `(): Sub` is not satisfied

trait BoundOnAssoc {
    type Assoc: Sub;
}
impl BoundOnAssoc for () {
    type Assoc = ();
    //~^ ERROR the trait bound `(): Sub` is not satisfied
}

trait BoundOnGat where Self::Assoc<u8>: Sub {
    type Assoc<T>;
}
impl BoundOnGat for u8 {
    type Assoc<T> = ();
    //~^ ERROR the trait bound `(): Sub` is not satisfied
}

fn trivial_bound() where (): Sub {}
//~^ ERROR the trait bound `(): Sub` is not satisfied

fn main() {}
