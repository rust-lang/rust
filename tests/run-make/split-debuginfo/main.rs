extern crate bar;

use bar::{make_bar, Bar};

fn main() {
    let b = make_bar(3);
    b.print();
}
