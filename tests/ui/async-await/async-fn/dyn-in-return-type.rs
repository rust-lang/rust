//@ edition:2024
//@ run-rustfix

async fn f() -> dyn core::fmt::Debug {
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| HELP consider returning an `impl Trait` instead of a `dyn Trait`
//~| HELP alternatively, box the return type, and wrap all of the returned values in `Box::new`
    Box::new("")
}
trait T {
    async fn f(&self) -> dyn core::fmt::Debug {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| HELP consider returning an `impl Trait` instead of a `dyn Trait`
    //~| HELP alternatively, box the return type, and wrap all of the returned values in `Box::new`
    //~| ERROR: the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time [E0277]
    //~| HELP: the trait `Sized` is not implemented for `(dyn Debug + 'static)`
        Box::new("")
    }
}
impl T for () {
    async fn f(&self) -> dyn core::fmt::Debug {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| HELP consider returning an `impl Trait` instead of a `dyn Trait`
    //~| HELP alternatively, box the return type, and wrap all of the returned values in `Box::new`
        Box::new("")
    }
}

fn main() {
    let _ = f();
    let _ = ().f();
}
