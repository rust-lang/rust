trait Trait {} //~ HELP this trait has no implementations, consider adding one
trait TraitWithParam<T> {} //~ HELP this trait has no implementations, consider adding one

mod m {
    pub trait Trait {}
    pub trait TraitWithParam<T> {}
    pub struct St;  //~ HELP the trait `Trait` is not implemented for `St`
    //~| HELP the trait `TraitWithParam<St>` is not implemented for `St`
    impl Trait for St {}
    impl<T> TraitWithParam<T> for St {}
}

fn func<T: Trait>(_: T) {} //~ NOTE required by a bound in `func`
//~^ NOTE required by this bound in `func`

fn func2<T: TraitWithParam<T>> (_: T) {} //~ NOTE required by a bound in `func2`
//~^ NOTE required by this bound in `func2`

fn main() {
    func(m::St); //~ ERROR the trait bound `St: Trait` is not satisfied
    //~^ NOTE unsatisfied trait bound
    //~| NOTE required by a bound introduced by this call
    //~| NOTE `St` implements similarly named trait `m::Trait`, but not `Trait`
    func2(m::St); //~ ERROR the trait bound `St: TraitWithParam<St>` is not satisfied
    //~^ NOTE unsatisfied trait bound
    //~| NOTE required by a bound introduced by this call
    //~| NOTE `St` implements similarly named trait `m::TraitWithParam`, but not `TraitWithParam<St>`
}
