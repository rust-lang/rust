//@ check-fail

static STATIC_VAR_FIVE: &One();
//~^ cannot find type
//~| free static item without body

fn main() {}
