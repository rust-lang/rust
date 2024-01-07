fn f() -> impl Fn() -> impl Sized { || () }
//~^ ERROR `impl Trait` only allowed in function and inherent method argument and return types, not in the return types of `Fn` trait bounds
fn g() -> &'static dyn Fn() -> impl Sized { &|| () }
//~^ ERROR `impl Trait` only allowed in function and inherent method argument and return types, not in the return types of `Fn` trait bounds

fn main() {}
