#[doc(notable_trait)] //~ ERROR the `doc(notable_trait)` attribute is experimental
trait SomeTrait {}

fn main() {
    #[doc(notable_trait)]
    //~^ ERROR the `doc(notable_trait)` attribute is experimental [E0658]
    //~| WARN `#![doc(notable_trait)]` must be a trait attribute
    //~| WARN this was previously accepted
    println!();
}
