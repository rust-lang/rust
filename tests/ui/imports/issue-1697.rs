// Testing that we don't fail abnormally after hitting the errors

use unresolved::*;
//~^ ERROR unresolved import `unresolved` [E0432]
//~| NOTE you might be missing crate `unresolved`
//~| HELP consider adding `extern crate unresolved` to use the `unresolved` crate

fn main() {}
