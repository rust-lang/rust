#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {}

fn anything<T>() -> T {
    let x: Void;
    match x { ! }
    //~^ ERROR used binding `x` isn't initialized
}
