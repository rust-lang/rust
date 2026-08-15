fn foo<T: impl Trait>() {}
//~^ ERROR expected trait bound, found `impl Trait` type
fn main() {}
