//@ edition: 2024

#[attr]
//~^ ERROR cannot find attribute `attr` in this scope
extern crate core as std;
//~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`

mod inner {
    use std::str;

    use crate::*;
}

fn main() {}
