struct Foo {}

impl Foo {
    pub fn foo(&mut self) {
        || bar(&mut self);
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable
    }

    pub fn baz(&self) {
        || bar(&mut self);
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable
        //~| ERROR cannot borrow data in a `&` reference as mutable
    }

    pub fn qux(mut self) {
        || bar(&mut self);
        // OK
    }

    pub fn quux(self) {
        || bar(&mut self);
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable
    }
}

fn bar(_: &mut Foo) {}

fn main() {}
