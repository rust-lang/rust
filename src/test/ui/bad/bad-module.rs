fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR failed to resolve. Use of undeclared type or module `thing`

    let foo = foo::bar::baz();
    //~^ ERROR failed to resolve. Use of undeclared type or module `foo`
}
