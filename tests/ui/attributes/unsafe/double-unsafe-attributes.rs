#[unsafe(unsafe(no_mangle))]
//~^ ERROR expected identifier, found keyword `unsafe`
//~| ERROR cannot find attribute `r#unsafe` in this scope
//~| ERROR unnecessary `unsafe`
fn a() {}

fn main() {}
