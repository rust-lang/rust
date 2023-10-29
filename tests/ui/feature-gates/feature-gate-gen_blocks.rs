// revisions: e2024 none
//[e2024] compile-flags: --edition 2024 -Zunstable-options

fn main() {
    gen {};
    //[none]~^ ERROR: cannot find struct, variant or union type `gen`
    //[e2024]~^^ ERROR: gen blocks are experimental
    //[e2024]~| ERROR: type annotations needed
}

#[cfg(FALSE)]
fn foo() {
    gen {};
    //[e2024]~^ ERROR: gen blocks are experimental
}
