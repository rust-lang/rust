mod circ1 {
    pub use crate::circ2::f2;
    pub fn f1() { println!("f1"); }
    pub fn common() -> usize { return 0; }
}

mod circ2 {
    pub use crate::circ1::f1;
    pub fn f2() { println!("f2"); }
    pub fn common() -> usize { return 1; }
}

mod test {
    use crate::circ1::*;

    fn test() { f1066(); } //~ ERROR cannot find function `f1066` in this scope
}

fn main() {}
