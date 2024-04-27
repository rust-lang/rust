#![feature(dyn_star, trait_upcasting)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

trait A: B {}
trait B {}
impl A for usize {}
impl B for usize {}

fn main() {
    let x: Box<dyn* A> = Box::new(1usize as dyn* A);
    let y: Box<dyn* B> = x;
    //~^ ERROR mismatched types
}
