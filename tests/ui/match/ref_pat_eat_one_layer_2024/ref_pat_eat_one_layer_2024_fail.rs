//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

pub fn main() {
    if let Some(&mut Some(&_)) = &Some(&Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&_)) = &Some(&mut Some(0)) {
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &mut u32 = x;
        //~^ ERROR: mismatched types
    }
    if let Some(&Some(&x)) = Some(&Some(&mut 0)) {
        //~^ ERROR: mismatched types
    }

    let &mut x = &&0;
    //~^ ERROR: mismatched types
    let _: &u32 = x;

    let &mut x = &&&&&&&&&&&&&&&&&&&&&&&&&&&&0;
    //~^ ERROR: mismatched types
    let _: &u32 = x;
}
