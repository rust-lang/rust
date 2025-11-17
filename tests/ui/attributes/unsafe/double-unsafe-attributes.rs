#[unsafe(unsafe(no_mangle))]
//~^ ERROR expected identifier, found keyword `unsafe`
//~| ERROR cannot find attribute `r#unsafe` in this scope
//~| ERROR `r#unsafe` is not an unsafe attribute
fn a() {}

fn main() {}
