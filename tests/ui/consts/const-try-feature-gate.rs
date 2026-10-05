// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ ERROR `?` is not allowed
    //~| ERROR `?` is not allowed
    //~| ERROR `?` is not yet stable in const contexts
    //~| ERROR `?` is not yet stable in const contexts
    None
}

fn main() {}
