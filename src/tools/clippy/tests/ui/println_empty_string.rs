// run-rustfix
#![allow(clippy::match_single_binding)]

fn main() {
    println!();
    println!("");

    match "a" {
        _ => println!(""),
    }

    eprintln!();
    eprintln!("");

    match "a" {
        _ => eprintln!(""),
    }
}
