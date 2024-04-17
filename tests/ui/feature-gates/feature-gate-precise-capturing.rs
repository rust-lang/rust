fn hello() -> impl use<> Sized {}
//~^ ERROR precise captures on `impl Trait` are experimental

fn main() {}
