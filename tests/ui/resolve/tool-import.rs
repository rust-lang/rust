//@ edition: 2018

use clippy::time::Instant;
//~^ ERROR: cannot find module or crate `clippy`

fn main() {
    Instant::now();
}
