#[ignore(clippy::single_match)]
//~^ ERROR valid forms for the attribute are `#[ignore = "reason"]` and `#[ignore]`
//~| HELP if you meant to silence a warning, consider using #![allow(clippy::single_match)] or #![expect(clippy::single_match)]
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main() {}
