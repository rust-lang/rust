//@ edition: 2015
#![allow(bare_trait_objects)]

fn main() {
    let _: &Copy + 'static; //~ ERROR expected a path
    //~^ ERROR is not dyn compatible
    let _: &'static Copy + 'static; //~ ERROR expected a path
    //~^ ERROR is not dyn compatible
}
