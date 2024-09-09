fn constant<const C: usize>() -> impl Sized + use<> {}
//~^ ERROR `impl Trait` must mention all const parameters in scope

fn main() {}
