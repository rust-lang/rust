#![feature(return_type_notation)]

trait Trait {
    fn method() {}
}

fn bound<T: Trait<method(..): Send>>() {}
//~^ ERROR return type notation used on function that is not `async` and does not return `impl Trait`

fn path<T>() where T: Trait, T::method(..): Send {}
//~^ ERROR return type notation used on function that is not `async` and does not return `impl Trait`

fn main() {}
