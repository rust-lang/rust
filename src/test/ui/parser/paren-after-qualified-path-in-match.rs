// compile-flags: -Z parse-only

fn foo() {
    match x {
        <T as Trait>::Type(2) => (),
        //~^ ERROR unexpected `(` after qualified path
        _ => (),
    }
}
