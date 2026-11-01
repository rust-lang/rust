#[derive(Eq, PartialEq)]
struct Test {
    a: &'b str,
    //~^ ERROR use of undeclared lifetime name `'b`
    //~| ERROR use of undeclared lifetime name `'b`
}

trait T {
    fn foo(&'static self) {}
}

impl T for Test {
    fn foo(&'b self) {} //~ ERROR use of undeclared lifetime name `'b`
}

fn main() {}
