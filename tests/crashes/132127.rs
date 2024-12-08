//@ known-bug: #132127
#![feature(dyn_star)]

trait Trait {}

fn main() {
    let x: dyn* Trait + Send = 1usize;
    x as dyn* Trait;
}
