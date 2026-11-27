//@ compile-flags:-Zforce-unstable-if-unmarked

#[unstable()]
//~^ ERROR stability attributes may not be used
//~| ERROR missing 'feature'
//~| ERROR missing 'issue'
#[stable()]
//~^ ERROR stability attributes may not be used
//~| ERROR missing 'feature'
//~| ERROR missing 'since'
fn main() {}
