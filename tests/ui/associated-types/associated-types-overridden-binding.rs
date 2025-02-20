#![feature(trait_alias)]

trait Foo: Iterator<Item = i32> {}
trait Bar: Foo<Item = u32> {} //~ ERROR type annotations needed

trait I32Iterator = Iterator<Item = i32>;
trait U32Iterator = I32Iterator<Item = u32>; //~ ERROR type annotations needed

fn main() {
    let _: &dyn I32Iterator<Item = u32>;
    //~^ ERROR conflicting associated type bounds
}
