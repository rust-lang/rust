#![allow(unconditional_recursion)]

struct Struct;

impl Struct {
    fn bar(self: &mut Self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
        //~^^ HELP try removing `&mut` here
    }

    fn imm(self) { //~ HELP consider changing this to be mutable
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
    }

    fn mtbl(mut self) {
        (&mut self).bar();
    }

    fn immref(&self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
        //~^^ ERROR cannot borrow data in a `&` reference as mutable [E0596]
    }

    fn mtblref(&mut self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
        //~^^ HELP try removing `&mut` here
    }
}

fn main() {}
