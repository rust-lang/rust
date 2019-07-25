#![feature(non_ascii_idents)]
#![deny(non_ascii_idents)]

const חלודה: usize = 2; //~ ERROR identifier contains non-ASCII characters

fn coöperation() {} //~ ERROR identifier contains non-ASCII characters

fn main() {
    let naïveté = 2; //~ ERROR identifier contains non-ASCII characters
    println!("{}", naïveté); //~ ERROR identifier contains non-ASCII characters
}
