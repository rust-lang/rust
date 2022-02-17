// This test case checks the behavior of typeck::check::method::suggest::is_fn on Ty::Error.
fn main() {
    let arc = std::sync::Arc::new(oops);
    //~^ ERROR cannot find value `oops` in this scope
    //~| NOTE not found
    // The error "note: `arc` is a function, perhaps you wish to call it" MUST NOT appear.
    arc.blablabla();
    //~^ ERROR no method named `blablabla`
    //~| NOTE method not found
    let arc2 = std::sync::Arc::new(|| 1);
    // The error "note: `arc2` is a function, perhaps you wish to call it" SHOULD appear
    arc2.blablabla();
    //~^ ERROR no method named `blablabla`
    //~| NOTE method not found
    //~| NOTE `arc2` is a function, perhaps you wish to call it
}
