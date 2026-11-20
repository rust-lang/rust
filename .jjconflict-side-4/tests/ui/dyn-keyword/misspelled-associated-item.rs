//@ edition: 2021

trait Trait {
    fn typo() -> Self;
}

fn main() {
    let () = Trait::typoe();
    //~^ ERROR expected a type, found a trait
    //~| HELP you can add the `dyn` keyword if you want a trait object
    //~| HELP you may have misspelled this associated item
}
