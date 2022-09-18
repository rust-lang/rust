#![deny(unused_variables)]

fn main() {
    let mut s = String::from("a");
    //~^ ERROR unused variable: `s` [unused_variables]
}
