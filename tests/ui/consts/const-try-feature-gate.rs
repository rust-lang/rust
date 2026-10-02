// gate-test-const_try

const fn t() -> Option<()> {
    Some(())?;
    //~^ ERROR cannot call conditionally-const method `<Option<()> as ops::try_trait_old::Try>::branch` in constant functions [E0658]
    //~| ERROR `ops::try_trait_old::Try` is not yet stable as a const trait
    //~| ERROR cannot call conditionally-const associated function `<Option<()> as ops::try_trait_old::FromResidual<Option<!>>>::from_residual` in constant functions [E0658]
    //~| ERROR `ops::try_trait_old::FromResidual` is not yet stable as a const trait
    None
}

fn main() {}
