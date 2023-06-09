fn f<A>() -> A { unimplemented!() }
fn foo() {
    let _ = f;
    //~^ ERROR type annotations needed
    //~| HELP consider specifying the generic argument
}
fn main() {}
