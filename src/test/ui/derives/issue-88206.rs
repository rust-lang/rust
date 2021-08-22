// compile-flags: -Z deduplicate-diagnostics=yes

#![warn(unused_imports)]
//~^ NOTE lint level

mod hey {
    pub trait Serialize {}
}

use hey::Serialize;
//~^ WARNING unused import
//~| NOTE `Serialize` is imported here

#[derive(Serialize)]
//~^ ERROR cannot find derive macro `Serialize`
struct A;

fn main() {}
