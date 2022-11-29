fn f() -> impl Fn() -> impl Sized { || () }
//~^ ERROR `impl Trait` not allowed within `Fn` trait return
fn g() -> &'static dyn Fn() -> impl Sized { &|| () }
//~^ ERROR `impl Trait` not allowed within `Fn` trait return

fn main() {}
