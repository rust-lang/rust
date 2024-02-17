//@ check-pass
#![deny(dead_code)]

// We previously incorrectly linted `L` as unused here.
const L: usize = 3;

fn main() {
    let p = Printer {};
    p.print();
}

trait Print<const N: usize> {
    fn print(&self) -> usize {
        3
    }
}

struct Printer {}
impl Print<L> for Printer {}
