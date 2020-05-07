#![allow(bare_trait_objects)]

fn main() {
    let a: i8 += 1;
    //~^ ERROR expected trait, found builtin type `i8`
    let _ = a;
}
