fn foo<T>() -> impl Sized + use<> {}
//~^ ERROR `impl Trait` must mention all type parameters in scope in `use<...>`

fn main() {}
