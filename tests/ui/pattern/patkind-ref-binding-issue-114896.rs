//@ run-rustfix
#![allow(dead_code)]

fn main() {
    fn x(a: &char) {
        let &b = a;
        b.make_ascii_uppercase();
        //~^ ERROR cannot borrow `b` as mutable, as it is not declared as mutable
    }
}
