//@ edition:2021
trait Trait {}

async fn fun() -> Trait {  //~ ERROR expected a type, found a trait
    todo!()
}

fn main() {}
