//@ check-fail

static STATIC_VAR_FIVE: &One();
//~^ ERROR cannot find type
//~| ERROR free static item without body

fn main() {}
