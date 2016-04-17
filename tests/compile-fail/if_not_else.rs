#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy)]
#![deny(if_not_else)]

fn bla() -> bool { unimplemented!() }

fn main() {
    if !bla() { //~ ERROR: Unnecessary boolean `not` operation
        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if 4 != 5 { //~ ERROR: Unnecessary `!=` operation
        println!("Bugs");
    } else {
        println!("Bunny");
    }
}
