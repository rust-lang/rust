#![feature(optin_builtin_traits)]
#![feature(specialization)]

struct S;
struct Z;

default impl S {} //~ ERROR inherent impls cannot be default

default unsafe impl Send for S {} //~ ERROR impls of auto traits cannot be default
default impl !Send for Z {} //~ ERROR impls of auto traits cannot be default

trait Tr {}
default impl !Tr for S {} //~ ERROR negative impls are only allowed for auto traits

fn main() {}
