trait Trait {} //~ HELP this trait has no implementations, consider adding one

mod m {
    pub trait Trait {}
    pub struct St;
    impl Trait for St {}
}

fn func<T: Trait>(_: T) {} //~ NOTE required by a bound in `func`
//~^ NOTE required by this bound in `func`

fn main() {
    func(m::St); //~ ERROR the trait bound `St: Trait` is not satisfied
    //~^ NOTE the trait `Trait` is not implemented for `St`
    //~| NOTE required by a bound introduced by this call
    //~| NOTE `St` implements similarly named `m::Trait`, but not `Trait`
}
