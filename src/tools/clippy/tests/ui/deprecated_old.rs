#[warn(unstable_as_slice)]
//~^ ERROR: lint `unstable_as_slice` has been removed: `Vec::as_slice` has been stabilized
//~| NOTE: `-D renamed-and-removed-lints` implied by `-D warnings`
#[warn(unstable_as_mut_slice)]
//~^ ERROR: lint `unstable_as_mut_slice` has been removed: `Vec::as_mut_slice` has been st
#[warn(misaligned_transmute)]
//~^ ERROR: lint `misaligned_transmute` has been removed: this lint has been split into ca

fn main() {}
