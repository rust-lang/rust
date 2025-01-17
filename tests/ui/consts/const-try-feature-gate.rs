// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ ERROR `?` is not allowed
    //~| ERROR `?` is not allowed
    None
}

fn main() {}
