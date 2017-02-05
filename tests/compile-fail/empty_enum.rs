#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![deny(empty_enum)]

enum Empty {} //~ ERROR enum with no variants
    //~^ HELP consider using the uninhabited type `!` or a wrapper around it

fn main() {
}
