// Issue: 101797, Suggest associated const for incorrect use of let in traits
//@ run-rustfix
#![allow(dead_code)]
trait Trait {
    let _X: i32;
    //~^ ERROR `let` statements are not allowed outside of functions or const blocks
}

fn main() {

}
