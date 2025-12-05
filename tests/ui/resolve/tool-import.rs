//@ edition: 2018

use clippy::time::Instant;
//~^ ERROR `clippy` is a tool module

fn main() {
    Instant::now();
}
