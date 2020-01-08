#![feature(optin_builtin_traits)]
#![feature(specialization)]

trait MyTrait {
    type Foo;
}

default impl !MyTrait for u32 {} //~ ERROR negative impls cannot be default impls

fn main() {}
