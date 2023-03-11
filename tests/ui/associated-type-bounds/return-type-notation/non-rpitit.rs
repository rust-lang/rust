#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Trait {
    fn method() {}
}

fn test<T: Trait<method(): Send>>() {}
//~^ ERROR  return type notation used on function that is not `async` and does not return `impl Trait`

fn main() {}
