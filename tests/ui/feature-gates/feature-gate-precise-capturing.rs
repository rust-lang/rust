fn hello() -> impl Sized + use<> {}
//~^ ERROR precise captures on `impl Trait` are experimental

fn main() {}
