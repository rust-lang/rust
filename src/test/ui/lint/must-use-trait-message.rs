#![feature(must_use_trait)]
#![deny(unused_must_use)] //~ NOTE the lint level is defined here

use std::marker::MustUse;

struct St;

impl St {
    fn new() -> Self {
        St
    }
}

impl MustUse for St {
    const REASON: &'static str = "because I said so";
}

fn main() {
    St::new();
    //~^ ERROR unused `St` that must be used
    //~| NOTE because I said so
}
