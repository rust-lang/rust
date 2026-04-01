// Test that we give suitable error messages when the user attempts to
// impl a trait `Trait` for its own object type.

// If the trait is dyn-incompatible, we give a more tailored message
// because we're such schnuckels:
trait DynIncompatible { fn eq(&self, other: Self); }
impl DynIncompatible for dyn DynIncompatible { }
//~^ ERROR E0038
//~| ERROR E0046

fn main() { }
