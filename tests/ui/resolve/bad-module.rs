fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR failed to resolve: use of undeclared crate or module `thing`

    let foo = foo::bar::baz();
    //~^ ERROR failed to resolve: use of undeclared crate or module `foo`
}
