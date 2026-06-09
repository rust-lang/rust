// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ ERROR `?` is not allowed
    //~| ERROR `?` is not allowed
    //~| ERROR `Try` is not yet stable as a const trait
    //~| ERROR `FromResidual` is not yet stable as a const trait
    None
}

fn main() {}
