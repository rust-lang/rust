extern crate bar;

use bar::{Bar, make_bar};

fn main() {
    let b = make_bar(3);
    b.print();
}
