#![feature(must_use_trait)]
#![deny(unused_must_use)]

use std::marker::MustUse;

struct St;

impl St {
    fn new() -> Self {
        St
    }
}

impl MustUse for St {}

fn main() {
    St::new(); //~ ERROR unused `St` that must be used
}
