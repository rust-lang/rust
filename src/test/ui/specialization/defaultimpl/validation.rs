#![feature(negative_impls)]
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

struct S;
struct Z;

default impl S {} //~ ERROR inherent impls cannot be `default`

default unsafe impl Send for S {} //~ ERROR impls of auto traits cannot be default
default impl !Send for Z {} //~ ERROR `std::marker::Send` impl requires `Z: std::marker::Send` but
                            //~^ ERROR impls of auto traits cannot be default

trait Tr {}
default impl !Tr for S {} //~ ERROR `Tr` impl requires `S: Tr` but

fn main() {}
