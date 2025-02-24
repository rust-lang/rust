fn main() {
    let foo = thing::len(Vec::new());
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `thing`

    let foo = foo::bar::baz();
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `foo`
}
