#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait Foo: for<T> Bar<T> {}

trait Bar<T: PointeeSized>: PointeeSized {
    fn method(&self) {}
}

fn needs_bar(x: &(impl Bar<i32> + PointeeSized)) {
    x.method();
}

impl Foo for () {}

impl<T: PointeeSized> Bar<T> for () {}

fn main() {
    let x: &dyn Foo = &();
    //~^ ERROR the trait `Foo` is not dyn compatible
    needs_bar(x);
}
