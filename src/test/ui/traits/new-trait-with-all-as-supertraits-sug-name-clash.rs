// edition:2021

trait NewTrait {}
trait NewTrait2 {}

fn foo() -> NewTrait + NewTrait2 {}
//~^ trait objects must include the `dyn` keyword
//~| only auto traits can be used as additional traits in a trait object

fn main() {}
