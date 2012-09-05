// xfail-test -- #2378 unfixed 
// aux-build:issue2378a.rs
// aux-build:issue2378b.rs

use issue2378a;
use issue2378b;

use issue2378a::{just, methods};
use issue2378b::{methods};

fn main() {
    let x = {a: just(3), b: just(5)};
    assert x[0u] == (3, 5);
}
