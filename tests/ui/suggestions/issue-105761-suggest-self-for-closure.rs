//@run-rustfix
#![allow(unused)]

struct S;
impl S {
    fn foo(&mut self) {
        let x = |v: i32| {
            self.bar();
            self.hel();
        };
        self.qux(); //~ ERROR cannot borrow `*self` as mutable because it is also borrowed as immutable
        x(1);
        x(3);
    }
    fn bar(&self) {}
    fn hel(&self) {}
    fn qux(&mut self) {}

    fn hello(&mut self) {
        let y = || {
            self.bar();
        };
        self.qux(); //~ ERROR cannot borrow `*self` as mutable because it is also borrowed as immutable
        y();
    }
}

fn main() {}
