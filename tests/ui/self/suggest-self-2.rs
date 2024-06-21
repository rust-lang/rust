struct Foo {}

impl Foo {
    fn foo(&self) {
        bar(self);
        //~^ ERROR cannot find function `bar`
        //~| HELP try calling `bar` as a method

        bar(&&self, 102);
        //~^ ERROR cannot find function `bar`
        //~| HELP try calling `bar` as a method

        bar(&mut self, 102, &"str");
        //~^ ERROR cannot find function `bar`
        //~| HELP try calling `bar` as a method

        bar();
        //~^ ERROR cannot find function `bar`

        self.bar();
        //~^ ERROR no method named `bar` found for reference
    }
}

fn main() {}
