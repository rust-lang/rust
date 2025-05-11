//@ revisions: e2024 none
//@[e2024] edition: 2024

fn test_gen() {
    gen {};
    //[none]~^ ERROR: cannot find struct, variant or union type `gen`
    //[e2024]~^^ ERROR: gen blocks are experimental
    //[e2024]~| ERROR: type annotations needed
}

fn test_async_gen() {
    async gen {};
    //[none]~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `gen`
    //[e2024]~^^ ERROR: gen blocks are experimental
    //[e2024]~| ERROR: type annotations needed
}

fn main() {}

#[cfg(false)]
fn foo() {
    gen {};
    //[e2024]~^ ERROR: gen blocks are experimental

    async gen {};
    //[e2024]~^ ERROR: gen blocks are experimental
    //[none]~^^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `gen`
}
