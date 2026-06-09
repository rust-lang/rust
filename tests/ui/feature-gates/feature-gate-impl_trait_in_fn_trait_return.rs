fn f() -> impl Fn() -> impl Sized { || () }
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds
fn g() -> &'static dyn Fn() -> impl Sized { &|| () }
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds

fn main() {}
