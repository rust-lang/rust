#![feature(final_associated_functions)]
#![feature(share_trait)]

use std::clone::Share;

struct Alias;

impl Clone for Alias {
    fn clone(&self) -> Self {
        Alias
    }
}

impl Share for Alias {
    fn share(&self) -> Self {
        //~^ ERROR cannot override `share` because it already has a `final` definition in the trait
        Alias
    }
}

fn main() {}
