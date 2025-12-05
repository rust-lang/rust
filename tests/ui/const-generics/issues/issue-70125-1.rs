//@ run-pass

const L: usize = 4;

pub trait Print<const N: usize> {
    fn print(&self) -> usize {
        N
    }
}

pub struct Printer;
impl Print<L> for Printer {}

fn main() {
    let p = Printer;
    assert_eq!(p.print(), 4);
}
