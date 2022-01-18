// compile-flags:-Zforce-unstable-if-unmarked

#[unstable()] //~ ERROR: stability attributes may not be used
#[stable()] //~ ERROR: stability attributes may not be used
#[rustc_deprecated()] //~ ERROR: stability attributes may not be used
//~^ ERROR missing 'since'
fn main() {}
