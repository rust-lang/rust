// compile-flags: -Z parse-only

fn foo() {
    match x {
        <T as Trait>::Type{key: value} => (),
        //~^ ERROR unexpected `{` after qualified path
        _ => (),
    }
}
