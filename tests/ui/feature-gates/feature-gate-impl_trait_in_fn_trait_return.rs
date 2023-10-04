fn f() -> impl Fn() -> impl Sized { || () }
//~^ ERROR `impl Trait` only allowed in function and inherent method argument and return types, not in `Fn` trait return
fn g() -> &'static dyn Fn() -> impl Sized { &|| () }
//~^ ERROR `impl Trait` only allowed in function and inherent method argument and return types, not in `Fn` trait return

fn main() {}
