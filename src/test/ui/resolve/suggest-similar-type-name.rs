struct FooType {}

impl FooType {
    fn bar() {}
}

fn main() {
    FooTyp::bar()
    //~^ ERROR failed to resolve: use of undeclared type `FooTyp`
}
