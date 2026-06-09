//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::fmt::Display;

trait Static: 'static {}
impl<T> Static for &'static T {}

fn foo<S: Display>(x: S) -> Box<dyn Display>
where
    &'static S: Static,
{
    Box::new(x)
}

fn main() {
    let s = foo(&String::from("blah blah blah"));
    //~^ ERROR temporary value dropped while borrowed
    println!("{}", s);
}
