// Testing that we don't fail abnormally after hitting the errors

use unresolved::*; //~ ERROR unresolved import `unresolved` [E0432]
                   //~^ you might be missing a crate named `unresolved`

fn main() {}
