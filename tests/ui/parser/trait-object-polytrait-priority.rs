#![allow(bare_trait_objects)]

trait Trait<'a> {}

fn main() {
    let _: &for<'a> Trait<'a> + 'static;
    //~^ ERROR expected a path on the left-hand side of `+`
    //~| HELP try adding parentheses
}
