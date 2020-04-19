fn f<A>() -> A { unimplemented!() }
fn foo() {
    let _ = f; //~ ERROR type annotations needed for `fn() -> A`
}
fn main() {}
