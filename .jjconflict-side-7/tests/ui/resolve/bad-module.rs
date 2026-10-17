fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR cannot find module or crate `thing`

    let foo = foo::bar::baz();
    //~^ ERROR cannot find module or crate `foo`
}
