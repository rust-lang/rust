//@ edition: 2021

trait Trait {}

impl dyn Trait {
    const CONST: () = ();
}

fn main() {
    match () {
        Trait::CONST => {}
        //~^ ERROR trait objects must include the `dyn` keyword
    }
}
