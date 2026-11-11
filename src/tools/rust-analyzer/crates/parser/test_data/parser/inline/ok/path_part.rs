fn foo() {
    let foo::Bar = ();
    let ::Bar = ();
    let Bar { .. } = ();
    let Bar(..) = ();
}
