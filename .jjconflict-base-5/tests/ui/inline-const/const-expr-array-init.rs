//@ build-pass

use std::cell::Cell;

fn main() {
    let _x = [const { Cell::new(0) }; 20];
}
