//@ revisions: e2015 middle e2024
//@[e2015] edition: 2015
//@[middle] edition: 2018..2024
//@[e2024] edition: 2024

fn test_gen() {
    gen {};
    //[e2015]~^ ERROR: cannot find struct, variant or union type `gen`
    //[middle]~^^ ERROR: cannot find struct, variant or union type `gen` in this scope
    //[e2024]~^^^ ERROR: gen blocks are experimental
    //[e2024]~| ERROR: type annotations needed
}

fn test_async_gen() {
    async gen {};
    //[e2015]~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `gen`
    //[middle]~^^ ERROR: expected one of `move`, `use`, `{`, `|`, or `||`, found `gen`
    //[e2024]~^^^ ERROR: gen blocks are experimental
    //[e2024]~| ERROR: type annotations needed
}

fn main() {}

#[cfg(false)]
fn foo() {
    gen {};
    //[e2024]~^ ERROR: gen blocks are experimental

    async gen {};
    //[e2015]~^ ERROR expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `gen`
    //[middle]~^^ ERROR: expected one of `move`, `use`, `{`, `|`, or `||`, found `gen`
    //[e2024]~^^^ ERROR: gen blocks are experimental
}
