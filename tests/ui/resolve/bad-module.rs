fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR cannot find `thing`

    let foo = foo::bar::baz();
    //~^ ERROR cannot find `foo`
}
