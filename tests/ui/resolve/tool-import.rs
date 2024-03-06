//@ edition: 2018

use clippy::time::Instant;
//~^ `clippy` is a tool module

fn main() {
    Instant::now();
}
