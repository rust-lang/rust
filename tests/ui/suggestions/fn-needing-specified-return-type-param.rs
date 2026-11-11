fn f<A>() -> A {
    unimplemented!()
}
fn foo() {
    let _ = f;
    //~^ ERROR type annotations needed
    //~| HELP consider specifying a concrete type for the type parameter `A`
}
fn main() {}
