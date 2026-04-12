fn f<A>() -> A { unimplemented!() }
fn foo() {
    let _ = f;
    //~^ ERROR type annotations needed
    //~| HELP consider specifying a concrete type for the generic type `A`
}
fn main() {}
