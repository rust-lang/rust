#[derive(unsafe(Debug))]
//~^ ERROR: expected identifier, found keyword `unsafe`
//~| ERROR: traits in `#[derive(...)]` don't accept arguments
//~| ERROR: expected identifier, found keyword `unsafe`
//~| ERROR: expected identifier, found keyword `unsafe`
//~| ERROR: cannot find derive macro `r#unsafe` in this scope
//~| ERROR: cannot find derive macro `r#unsafe` in this scope
struct Foo;

#[unsafe(derive(Debug))] //~ ERROR: is not an unsafe attribute
struct Bar;

fn main() {}
