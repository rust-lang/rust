fn foo() {
    let S() = ();
    let S(_) = ();
    let S(_,) = ();
    let S(_, .. , x) = ();
    let S(| a) = ();
}
