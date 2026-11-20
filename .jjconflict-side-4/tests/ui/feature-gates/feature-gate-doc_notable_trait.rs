#[doc(notable_trait)] //~ ERROR the `doc(notable_trait)` attribute is experimental
trait SomeTrait {}

fn main() {
    #[doc(notable_trait)]
    //~^ ERROR the `doc(notable_trait)` attribute is experimental [E0658]
    println!();
}
