//@ edition: 2021

trait Trait {}

impl dyn Trait {
    const CONST: () = ();
}

fn main() {
    match () {
        Trait::CONST => {}
        //~^ ERROR expected a type, found a trait
        //~| HELP you can add the `dyn` keyword if you want a trait object
    }
}
