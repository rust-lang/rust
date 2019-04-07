struct Struct;

impl Struct {
    fn bar(self: &mut Self) {
        //~^ WARN function cannot return without recursing
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
    }

    fn imm(self) {
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
    }
}

fn main () {}
