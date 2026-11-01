//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// `foo` has a `opaque: 'static` implied bound. This can be proven
// by the caller via the `+ 'static` item bound. However, we must not
// use this implied bound to assume `T: 'static` inside of `foo` as doing
// so would be non-productive cyclic reasoning.

use std::fmt::Display;
fn foo<T: Display>(x: T) -> &'static (impl Display + 'static) {
    //[next]~^ ERROR the parameter type `T` may not live long enough
    Box::leak(Box::new(x))
    //~^ ERROR the parameter type `T` may not live long enough
    //[current]~| ERROR the parameter type `T` may not live long enough
}

fn main() {
    let temp = foo(String::from("temp").as_str());
    println!("{temp}");
}
