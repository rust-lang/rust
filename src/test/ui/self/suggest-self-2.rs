struct Foo {}

impl Foo {
    fn foo(&self) {
        bar(self);
        //~^ ERROR cannot find function `bar` in this scope
        //~| HELP try calling method instead of passing `self` as parameter


        bar(&self);
        //~^ ERROR cannot find function `bar` in this scope
        //~| HELP try calling method instead of passing `self` as parameter

        bar();
        //~^ ERROR cannot find function `bar` in this scope

        self.bar();
        //~^ ERROR no method named `bar` found for type
    }
}

fn main() {}
