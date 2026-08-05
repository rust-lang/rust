use std::future::Future;

pub fn dyn_func<T>(
    executor: impl FnOnce(T) -> dyn Future<Output = ()>,
) -> Box<dyn FnOnce(T) -> dyn Future<Output = ()>> {
    Box::new(executor) //~ ERROR may not live long enough
}

trait Trait {
    fn method(&self) {}
}

impl Trait for fn() {}

pub fn in_ty_param<T: Fn() -> dyn std::fmt::Debug> (t: T) {
    t.method();
    //~^ ERROR no method named `method` found for type parameter `T`
}

fn with_sized<T: Fn() -> &'static (dyn std::fmt::Debug) + ?Sized>() {
    without_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
}

fn without_sized<T: Fn() -> &'static dyn std::fmt::Debug>() {}

fn main() {}
