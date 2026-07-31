//@ run-rustfix
#![allow(unused_features)]
#![allow(unused_mut)]
#![feature(final_associated_functions)]

fn visibility() {
    pub let s: &str = "hello world";
    //~^ ERROR a `let` statement cannot have a visibility

    println!("{s}");
}

// FIXME: Make this case produce a good error like the others
// fn default() {
//     default let s: &str = "hello world";
//     println!("{s}");
// }

fn final_imm() {
    final let s: &str = "hello world";
    //~^ ERROR a `let` statement cannot be `final`

    println!("{s}");
}

fn final_mut() {
    final let mut s: &str = "hello world";
    //~^ ERROR a `let` statement cannot be `final`

    println!("{s}");
}

fn final_ref() {
    final let ref s: &str = "hello world";
    //~^ ERROR a `let` statement cannot be `final`

    println!("{s}");
}

fn main() {
    visibility();
    // default();
    final_imm();
    final_mut();
    final_ref();
}
