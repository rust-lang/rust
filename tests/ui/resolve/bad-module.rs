fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR cannot find item `thing`

    let foo = foo::bar::baz();
    //~^ ERROR cannot find item `foo`
}
