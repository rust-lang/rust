//@ edition: 2018

use clippy::time::Instant;
//~^ ERROR: cannot find `clippy`
//~| NOTE: `clippy` is a tool module

fn main() {
    Instant::now();
}
