#![feature(trait_alias)]

// Although we *elaborate* `T: Alias` to `i32: B`, we should
// not consider `B` to be a supertrait of the type.
trait Alias = A where i32: B;

trait A {}

trait B {
    fn test(&self);
}

trait C: Alias {}

impl A for () {}

impl C for () {}

impl B for i32 {
    fn test(&self) {
        println!("hi {self}");
    }
}

fn test(x: &dyn C) -> &dyn B {
    x
    //~^ ERROR mismatched types
}

fn main() {
    let x: &dyn C = &();
}
